//! Native Split Radiance Cascades implementation.
//!
//! The CPU modules are the executable specification and validation reference.
//! The `gpu` module maps the same constants, keys, intervals, and frame graph to
//! Sokol's compute API when the `native-app` feature is enabled.

pub mod bvh;
pub mod constants;
pub mod interval;
pub mod math;
pub mod profiling;
pub mod rcb;
pub mod sampling;
pub mod scene;
pub mod sparse;
pub mod validation;

#[cfg(feature = "native-app")]
pub mod capture;
#[cfg(feature = "native-app")]
pub mod gpu;

pub use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
