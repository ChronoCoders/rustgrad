//! # rustgrad
//!
//! Production-grade ML training framework in Rust.
//!
//! ## Architecture
//!
//! - [`tensor`] — strided tensors with device and dtype tracking
//! - [`autograd`] — tape-based automatic differentiation engine
//! - [`backend`] — compute backends (CPU via faer; CUDA in Phase 6)
//! - [`ops`] — differentiable ops (elementwise, reduction, shape, matmul, attention)
//! - [`nn`] — neural network modules
//! - [`optim`] — optimizers

pub mod autograd;
pub mod backend;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tensor;

// Top-level re-exports for ergonomic use.
pub use autograd::{backward, Context, GradFn, Node, NodeId, Tape, TensorStore};
pub use backend::{Backend, CpuBackend};
pub use ops::attention::scaled_dot_product_attention;
pub use ops::binary::{add, div, mul, sub};
pub use ops::embedding::embedding;
pub use ops::matmul::matmul;
pub use ops::norm::layer_norm;
pub use ops::reduction::{mean, sum};
pub use ops::shape::{permute, reshape, squeeze, transpose, unsqueeze};
pub use ops::unary::softmax;
pub use tensor::{DType, Device, Layout, Storage, Tensor};
