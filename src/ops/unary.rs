use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::{Layout, Storage, Tensor};

// ── softmax ───────────────────────────────────────────────────────────────────

struct SoftmaxGrad {
    /// Forward output (the softmax probabilities), stored contiguous.
    out_storage: Arc<Storage>,
    shape: Vec<usize>,
    axis: usize,
    input_id: NodeId,
}

impl GradFn for SoftmaxGrad {
    /// Backward pass for softmax.
    ///
    /// Given softmax output `s` and upstream gradient `g`, the input gradient is:
    ///
    /// ```text
    /// grad_x[i] = s[i] * (g[i] - dot(g, s))
    /// ```
    ///
    /// where `dot(g, s)` is computed independently per slice along `axis`.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let s = &self.out_storage.data;
        let axis_size = self.shape[self.axis];
        let outer_size: usize = self.shape[..self.axis].iter().product();
        let inner_size: usize = self.shape[self.axis + 1..].iter().product();
        let numel: usize = self.shape.iter().product();
        let mut grad_input = vec![0.0f32; numel];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = outer * axis_size * inner_size;

                // dot = Σ_k g[k] * s[k] for this slice.
                let mut dot = 0.0f32;
                for k in 0..axis_size {
                    let idx = base + k * inner_size + inner;
                    dot += grad_output[idx] * s[idx];
                }

                // grad_x[i] = s[i] * (g[i] - dot).
                for k in 0..axis_size {
                    let idx = base + k * inner_size + inner;
                    grad_input[idx] = s[idx] * (grad_output[idx] - dot);
                }
            }
        }
        vec![grad_input]
    }

    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Numerically stable softmax along `axis`.
///
/// For each slice of `a` along `axis` the operation is:
///
/// ```text
/// out[i] = exp(a[i] - max) / Σ exp(a[j] - max)
/// ```
///
/// The output has the same shape as `a`. Every output slice sums to 1.0.
///
/// # Panics
///
/// Panics if `axis >= a.ndim()`.
pub fn softmax(ctx: &Context, tape: &mut Tape, a: &Tensor, axis: usize) -> Tensor {
    assert!(
        axis < a.ndim(),
        "softmax: axis {} out of range for tensor with {} dimensions",
        axis,
        a.ndim()
    );

    let shape = a.shape().to_vec();
    let numel: usize = shape.iter().product();

    // Materialize (handles non-contiguous layouts via strided iteration).
    let a_data = a.to_vec();
    let mut out_data = vec![0.0f32; numel];
    ctx.backend.softmax(&a_data, &mut out_data, &shape, axis);

    let storage = Arc::new(Storage::new(out_data, a.device(), a.dtype()));
    let layout = Layout::contiguous(shape.clone());
    let out = Tensor::from_storage(Arc::clone(&storage), layout, a.requires_grad);

    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(SoftmaxGrad {
                out_storage: storage,
                shape,
                axis,
                input_id: a.node,
            }),
        ));
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::context::Context;
    use crate::autograd::{backward, Node, Tape, TensorStore};
    use crate::backend::CpuBackend;
    use crate::tensor::Device;

    fn ctx() -> Context {
        Context::new(Arc::new(CpuBackend), Device::Cpu)
    }

    fn leaf(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_vec(data, shape, Device::Cpu).with_grad()
    }

    #[test]
    fn softmax_forward_last_axis() {
        // Shape [2,3], axis=1. Each row must sum to 1.
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]);
        let out = softmax(&ctx, &mut tape, &a, 1);
        assert_eq!(out.shape(), &[2, 3]);
        let v = out.to_vec();
        let row0: f32 = v[..3].iter().sum();
        let row1: f32 = v[3..].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-6, "row0 sum={row0}");
        assert!((row1 - 1.0).abs() < 1e-6, "row1 sum={row1}");
        // Uniform row: all 1/3.
        for &val in &v[3..] {
            assert!((val - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {val}");
        }
        // Non-uniform row: monotonically increasing (larger logit → larger prob).
        assert!(v[0] < v[1] && v[1] < v[2]);
    }

    #[test]
    fn softmax_backward_uniform() {
        // Uniform logits → uniform softmax. Upstream grad all-ones.
        // grad_x[i] = s[i] * (1 - Σ s[j]) = s[i] * (1 - 1) = 0.
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 1.0, 1.0], vec![3]);
        tape.push(Node::leaf(a.node));
        let out = softmax(&ctx, &mut tape, &a, 0);
        // Force upstream grad = all ones by using backward on sum-of-outputs.
        // Actually backward() seeds with 1.0, so output must be scalar.
        // Instead, verify via manual reduction: sum output = 1, backward gives zeros.
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // backward seeds out.grad = [1,1,1]. grad_x[i] = s[i]*(1 - 1) = 0.
        let g = a.grad(&store).expect("grad must exist"); // SAFE: requires_grad=true
        for &v in g {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn softmax_backward_peaked() {
        // Large spread: one element dominates. Gradient should be near zero
        // for the dominant element and negative for others.
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![10.0, 0.0, 0.0], vec![3]);
        tape.push(Node::leaf(a.node));
        let out = softmax(&ctx, &mut tape, &a, 0);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Upstream grad is seeded as [1,1,1] by backward().
        // s ≈ [1, 0, 0]. dot(g,s) ≈ 1.
        // grad[0] = 1*(1-1) = 0, grad[1] = 0*(1-1) = 0, grad[2] = 0*(1-1) = 0.
        let g = a.grad(&store).expect("grad must exist"); // SAFE: requires_grad=true
        for &v in g {
            assert!(v.abs() < 1e-3, "expected ~0, got {v}");
        }
    }
}
