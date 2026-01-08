//! CEF Offscreen Renderer
//!
//! This module provides CEF-based web content rendering using offscreen rendering (OSR).
//! Web content is rendered to GPU textures that can be imported into Slint.
//!
//! Based on the cef-rs OSR example:
//! https://github.com/tauri-apps/cef-rs/tree/main/examples/osr

use crate::error::{AppError, AppResult};
use anyhow::anyhow;
use cef::{self, rc::Rc, *};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::sync::{Arc, Mutex};

/// CEF offscreen browser instance with texture import capabilities
pub struct CefOffscreenBrowser {
    browser: Option<cef::Browser>,
    render_handler: Arc<Mutex<CefRenderHandler>>,
    wgpu_device: Arc<wgpu::Device>,
    wgpu_queue: Arc<wgpu::Queue>,
    size: (u32, u32),
}

/// Render handler for CEF offscreen rendering
#[derive(Clone)]
pub struct CefRenderHandler {
    device_scale_factor: f32,
    size: Arc<Mutex<(u32, u32)>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    current_texture: Arc<Mutex<Option<wgpu::Texture>>>,
    current_bind_group: Arc<Mutex<Option<wgpu::BindGroup>>>,
}

impl CefRenderHandler {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        device_scale_factor: f32,
        size: (u32, u32),
    ) -> (Self, Arc<Mutex<(u32, u32)>>) {
        let size_rc = Arc::new(Mutex::new(size));
        let handler = Self {
            size: size_rc.clone(),
            device_scale_factor,
            device,
            queue,
            current_texture: Arc::new(Mutex::new(None)),
            current_bind_group: Arc::new(Mutex::new(None)),
        };
        (handler, size_rc)
    }

    pub fn get_current_bind_group(&self) -> Option<wgpu::BindGroup> {
        self.current_bind_group.lock().ok()?.clone()
    }

    pub fn get_current_texture(&self) -> Option<wgpu::Texture> {
        self.current_texture.lock().ok()?.clone()
    }
}

// Wrapper for CEF RenderHandler trait
wrap_render_handler! {
    pub struct RenderHandlerWrapper {
        handler: CefRenderHandler,
    }

    impl RenderHandler {
        fn view_rect(&self, _browser: Option<&mut Browser>, rect: Option<&mut Rect>) {
            if let Some(rect) = rect {
                let size = self.handler.size.lock().unwrap();
                if size.0 > 0 && size.1 > 0 {
                    rect.width = size.0 as _;
                    rect.height = size.1 as _;
                }
            }
        }

        fn screen_info(
            &self,
            _browser: Option<&mut Browser>,
            screen_info: Option<&mut ScreenInfo>,
        ) -> ::std::os::raw::c_int {
            if let Some(screen_info) = screen_info {
                screen_info.device_scale_factor = self.handler.device_scale_factor;
                return true as _;
            }
            false as _
        }

        fn screen_point(
            &self,
            _browser: Option<&mut Browser>,
            _view_x: ::std::os::raw::c_int,
            _view_y: ::std::os::raw::c_int,
            _screen_x: Option<&mut ::std::os::raw::c_int>,
            _screen_y: Option<&mut ::std::os::raw::c_int>,
        ) -> ::std::os::raw::c_int {
            false as _
        }

        #[cfg(feature = "webview")]
        fn on_accelerated_paint(
            &self,
            _browser: Option<&mut Browser>,
            type_: PaintElementType,
            _dirty_rects_count: usize,
            _dirty_rects: Option<&Rect>,
            info: Option<&AcceleratedPaintInfo>,
        ) {
            let Some(info) = info else { return };

            // Only handle VIEW paint type (not POPUP)
            if type_ != PaintElementType::default() {
                return;
            }

            // Import texture from CEF using platform-specific handle
            let src_texture = {
                use cef::osr_texture_import::shared_texture_handle::SharedTextureHandle;

                let shared_handle = SharedTextureHandle::new(info);
                if let SharedTextureHandle::Unsupported = shared_handle {
                    tracing::error!("Platform does not support accelerated painting");
                    return;
                }

                match shared_handle.import_texture(&self.handler.device) {
                    Ok(texture) => texture,
                    Err(e) => {
                        tracing::error!("Failed to import shared texture: {:?}", e);
                        return;
                    }
                }
            };

            // Create sampler for texture
            let sampler = self
                .handler
                .device
                .create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });

            // Create bind group layout
            let texture_bind_group_layout =
                self.handler
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("CEF Texture Bind Group Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    multisampled: false,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    });

            // Create bind group with texture and sampler
            let bind_group = self
                .handler
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("CEF Texture Bind Group"),
                    layout: &texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&src_texture.create_view(
                                &wgpu::TextureViewDescriptor {
                                    label: Some("CEF Texture View"),
                                    ..Default::default()
                                },
                            )),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });

            // Store both the texture and bind group for later use
            if let Ok(mut texture) = self.handler.current_texture.lock() {
                *texture = Some(src_texture);
            }
            if let Ok(mut bind_group_lock) = self.handler.current_bind_group.lock() {
                *bind_group_lock = Some(bind_group);
            }
        }
    }
}

impl RenderHandlerWrapper {
    pub fn build(handler: CefRenderHandler) -> RenderHandler {
        Self::new(handler)
    }
}

// CEF Client wrapper
wrap_client! {
    pub struct ClientWrapper {
        render_handler: RenderHandler,
    }

    impl Client {
        fn render_handler(&self) -> Option<cef::RenderHandler> {
            Some(self.render_handler.clone())
        }
    }
}

impl ClientWrapper {
    pub fn build(render_handler: CefRenderHandler) -> Client {
        Self::new(RenderHandlerWrapper::build(render_handler))
    }
}

impl CefOffscreenBrowser {
    /// Create a new CEF offscreen browser
    pub fn new(
        url: &str,
        width: u32,
        height: u32,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_scale_factor: f32,
    ) -> AppResult<Self> {
        // Create render handler
        let (render_handler, size_rc) = CefRenderHandler::new(
            (*device).clone(),
            (*queue).clone(),
            device_scale_factor,
            (width, height),
        );

        let render_handler_arc = Arc::new(Mutex::new(render_handler.clone()));

        // Check if accelerated OSR is supported
        let accelerated_osr = cfg!(all(
            any(
                target_os = "macos",
                target_os = "windows",
                target_os = "linux"
            ),
            feature = "webview"
        ));

        // Create window info for offscreen rendering
        let window_info = WindowInfo {
            windowless_rendering_enabled: true as _,
            shared_texture_enabled: accelerated_osr as _,
            external_begin_frame_enabled: accelerated_osr as _,
            ..Default::default()
        };

        // Create browser settings
        let browser_settings = BrowserSettings {
            windowless_frame_rate: 60, // Target 60 FPS
            ..Default::default()
        };

        // Create client with render handler
        let mut client = ClientWrapper::build(render_handler);

        // Create browser
        let browser = cef::browser_host_create_browser_sync(
            Some(&window_info),
            Some(&mut client),
            Some(&cef::CefString::from(url)),
            Some(&browser_settings),
            None, // no dictionary value
            None, // no request context
        );

        if browser.is_none() {
            return Err(AppError::Other(anyhow::anyhow!(
                "Failed to create CEF browser"
            )));
        }

        Ok(Self {
            browser,
            render_handler: render_handler_arc,
            wgpu_device: device,
            wgpu_queue: queue,
            size: (width, height),
        })
    }

    /// Get the current rendered texture as a wgpu BindGroup
    pub fn get_texture(&self) -> Option<wgpu::BindGroup> {
        self.render_handler.lock().ok()?.get_current_bind_group()
    }

    /// Resize the browser viewport
    pub fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
        if let Ok(handler) = self.render_handler.lock() {
            *handler.size.lock().unwrap() = (width, height);
        }

        // Notify CEF of resize
        if let Some(ref browser) = self.browser {
            if let Some(host) = browser.host() {
                host.was_resized();
            }
        }
    }

    /// Send an external begin frame signal (for frame updates)
    pub fn send_frame(&self) {
        if let Some(ref browser) = self.browser {
            if let Some(host) = browser.host() {
                host.send_external_begin_frame();
            }
        }
    }

    /// Navigate to a new URL
    pub fn load_url(&self, url: &str) {
        if let Some(ref browser) = self.browser {
            let frame = browser.main_frame();
            if let Some(frame) = frame {
                frame.load_url(Some(&cef::CefString::from(url)));
            }
        }
    }

    /// Get the underlying CEF browser
    pub fn browser(&self) -> Option<&cef::Browser> {
        self.browser.as_ref()
    }

    /// Capture the current frame as a Slint Image
    ///
    /// This reads back the GPU texture to CPU memory and creates a Slint Image.
    /// Note: This involves a GPU-to-CPU copy and should be called sparingly
    /// (typically at display refresh rate, not every CEF render).
    pub async fn capture_frame(&self) -> AppResult<Image> {
        // Get the current texture
        let texture = self
            .render_handler
            .lock()
            .map_err(|e| AppError::Other(anyhow!("Failed to lock render handler: {}", e)))?
            .get_current_texture()
            .ok_or_else(|| AppError::Other(anyhow!("No texture available")))?;

        let (width, height) = self.size;

        // Create a temporary buffer to read pixels to
        let bytes_per_row = width * 4; // RGBA8
        let padded_bytes_per_row = (bytes_per_row + 255) & !255; // Align to 256 bytes
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let output_buffer = self.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CEF Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create command encoder to copy texture to buffer
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CEF Readback Encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.wgpu_queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read the data
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for mapping to complete
        rx.await
            .map_err(|e| AppError::Other(anyhow!("Failed to receive map result: {}", e)))?
            .map_err(|e| AppError::Other(anyhow!("Failed to map buffer: {:?}", e)))?;

        // Read the data
        let data = buffer_slice.get_mapped_range();

        // If padded, we need to remove padding
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        if padded_bytes_per_row != bytes_per_row {
            for y in 0..height {
                let row_start = (y * padded_bytes_per_row) as usize;
                let row_end = row_start + bytes_per_row as usize;
                rgba_data.extend_from_slice(&data[row_start..row_end]);
            }
        } else {
            rgba_data.extend_from_slice(&data[..(width * height * 4) as usize]);
        }

        drop(data);
        output_buffer.unmap();

        // Convert to Slint SharedPixelBuffer
        let pixel_buffer =
            SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&rgba_data, width, height);

        Ok(Image::from_rgba8(pixel_buffer))
    }

    /// Capture the current frame as raw RGBA pixel data
    ///
    /// Returns (pixels, width, height) where pixels is a Vec<u8> of RGBA8 data.
    /// This is useful for sending pixel data across threads, as Vec<u8> is Send.
    pub async fn capture_frame_pixels(&self) -> AppResult<(Vec<u8>, u32, u32)> {
        // Get the current texture
        let texture = self
            .render_handler
            .lock()
            .map_err(|e| AppError::Other(anyhow!("Failed to lock render handler: {}", e)))?
            .get_current_texture()
            .ok_or_else(|| AppError::Other(anyhow!("No texture available")))?;

        let (width, height) = self.size;

        // Create a temporary buffer to read pixels to
        let bytes_per_row = width * 4; // RGBA8
        let padded_bytes_per_row = (bytes_per_row + 255) & !255; // Align to 256 bytes
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let output_buffer = self.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CEF Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create command encoder to copy texture to buffer
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CEF Readback Encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.wgpu_queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read the data
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for mapping to complete
        rx.await
            .map_err(|e| AppError::Other(anyhow!("Failed to receive map result: {}", e)))?
            .map_err(|e| AppError::Other(anyhow!("Failed to map buffer: {:?}", e)))?;

        // Read the data
        let data = buffer_slice.get_mapped_range();

        // If padded, we need to remove padding
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        if padded_bytes_per_row != bytes_per_row {
            for y in 0..height {
                let row_start = (y * padded_bytes_per_row) as usize;
                let row_end = row_start + bytes_per_row as usize;
                rgba_data.extend_from_slice(&data[row_start..row_end]);
            }
        } else {
            rgba_data.extend_from_slice(&data[..(width * height * 4) as usize]);
        }

        drop(data);
        output_buffer.unmap();

        Ok((rgba_data, width, height))
    }
}

impl Drop for CefOffscreenBrowser {
    fn drop(&mut self) {
        // Close the browser
        if let Some(ref browser) = self.browser {
            if let Some(host) = browser.host() {
                host.close_browser(false as _); // Don't force close
            }
        }
    }
}
