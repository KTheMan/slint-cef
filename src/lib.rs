#![warn(missing_docs)]

//! CEF offscreen rendering helpers for Slint.
//!
//! This crate is currently legacy/experimental. Modules are behind features so
//! the crate can exist as a submodule without affecting build times.

#[cfg(feature = "renderer")]
pub mod cef_renderer;

#[cfg(feature = "renderer")]
pub use cef_renderer::*;
