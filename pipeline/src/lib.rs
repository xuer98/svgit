//! `svgit-pipeline` — owned raster-to-vector pipeline stages (Level 2).
//!
//! This crate is where svgit gradually takes ownership of the tracing pipeline
//! described in the project plan: preprocess → color quantization → segmentation
//! → boundary extraction → simplification → curve fitting → layering →
//! serialization.
//!
//! Today it implements the first and highest-leverage stage — **color
//! quantization** (k-means in CIELAB space). It runs as a pre-processing pass
//! before the VTracer core; later stages will replace more of that core.

pub mod color;
pub mod contour;
pub mod curvefit;
pub mod quantize;
pub mod refine;
pub mod segment;
pub mod simplify;
pub mod svg;
pub mod trace;

pub use quantize::{quantize_rgba, QuantizeConfig};
pub use refine::{RefineConfig, Refiner};
pub use trace::{trace_layered, trace_rgba, TraceConfig};
