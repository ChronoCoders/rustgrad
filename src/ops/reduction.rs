use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::{Layout, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the output shape for a reduction along `axis` (axis is removed).
fn reduce_shape(in_shape: &[usize], axis: usize) -> Vec<usize> {
    in_shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != axis)
        .map(|(_, &s)| s)
        .collect()
}

/// Row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut st = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

/// Expand `grad` (shape = `in_shape` minus `axis`) back to `in_shape` by
/// broadcasting along `axis`. This is the backward pass for sum.
///
/// Every element in `grad` is copied to all `axis_size` positions along
/// the reduced axis.
fn expand_to_axis(grad: &[f32], in_shape: &[usize], axis: usize) -> Vec<f32> {
    let ndim = in_shape.len();
    let numel: usize = in_shape.iter().product();
    let in_strides = row_major_strides(in_shape);

    // Strides for the reduced shape (in_shape with `axis` removed).
    let out_shape = reduce_shape(in_shape, axis);
    let out_strides = row_major_strides(&out_shape);

    let mut result = vec![0.0f32; numel];
    let mut in_indices = vec![0usize; ndim];

    for _ in 0..numel {
        let in_flat: usize =
            in_indices.iter().zip(in_strides.iter()).map(|(&i, &s)| i * s).sum();

        // Map to grad index by dropping the axis coordinate.
        let out_flat: usize = in_indices
            .iter()
            .enumerate()
            .filter(|&(d, _)| d != axis)
            .map(|(d, &i)| {
                let od = if d < axis { d } else { d - 1 };
                i * out_strides[od]
            })
            .sum();

        result[in_flat] = grad[out_flat];

        // Advance in_indices in row-major order.
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                in_indices[d] += 1;
                if in_indices[d] < in_shape[d] {
                    carry = false;
                } else {
                    in_indices[d] = 0;
                }
            }
        }
    }
    result
}

// ── sum ──────────────────────────────────────────────────────────────────────

struct SumGrad {
    in_shape: Vec<usize>,
    axis: usize,
    input_id: NodeId,
}

impl GradFn for SumGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        vec![expand_to_axis(grad_output, &self.in_shape, self.axis)]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Reduce `a` by summing along `axis`. The axis dimension is removed from
/// the output shape.
///
/// # Panics
///
/// Panics if `axis >= a.ndim()`.
pub fn sum(ctx: &Context, tape: &mut Tape, a: &Tensor, axis: usize) -> Tensor {
    assert!(
        axis < a.ndim(),
        "sum: axis {} out of range for tensor with {} dimensions",
        axis,
        a.ndim()
    );
    let in_shape = a.shape().to_vec();
    let out_shape = reduce_shape(&in_shape, axis);
    let out_numel = out_shape.iter().product::<usize>().max(1);

    // Materialize input (handles non-contiguous layouts).
    let a_data = a.to_vec();
    let mut out_data = vec![0.0f32; out_numel];
    ctx.backend.sum(&a_data, &mut out_data, &in_shape, axis);

    let storage = Arc::new(crate::tensor::Storage::new(out_data, a.device(), a.dtype()));
    let layout = if out_shape.is_empty() {
        Layout::contiguous(vec![1]) // scalar result stored as [1]
    } else {
        Layout::contiguous(out_shape)
    };
    let out = Tensor::from_storage(storage, layout, a.requires_grad);

    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(SumGrad { in_shape, axis, input_id: a.node }),
        ));
    }
    out
}

// ── mean ─────────────────────────────────────────────────────────────────────

struct MeanGrad {
    in_shape: Vec<usize>,
    axis: usize,
    axis_size: usize,
    input_id: NodeId,
}

impl GradFn for MeanGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let scale = 1.0f32 / self.axis_size as f32;
        let scaled: Vec<f32> = grad_output.iter().map(|&g| g * scale).collect();
        vec![expand_to_axis(&scaled, &self.in_shape, self.axis)]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Reduce `a` by averaging along `axis`. The axis dimension is removed from
/// the output shape.
///
/// # Panics
///
/// Panics if `axis >= a.ndim()`.
pub fn mean(ctx: &Context, tape: &mut Tape, a: &Tensor, axis: usize) -> Tensor {
    assert!(
        axis < a.ndim(),
        "mean: axis {} out of range for tensor with {} dimensions",
        axis,
        a.ndim()
    );
    let in_shape = a.shape().to_vec();
    let axis_size = in_shape[axis];
    let out_shape = reduce_shape(&in_shape, axis);
    let out_numel = out_shape.iter().product::<usize>().max(1);

    let a_data = a.to_vec();
    let mut sum_data = vec![0.0f32; out_numel];
    ctx.backend.sum(&a_data, &mut sum_data, &in_shape, axis);

    let mean_data: Vec<f32> = sum_data.iter().map(|&s| s / axis_size as f32).collect();

    let storage = Arc::new(crate::tensor::Storage::new(mean_data, a.device(), a.dtype()));
    let layout = if out_shape.is_empty() {
        Layout::contiguous(vec![1])
    } else {
        Layout::contiguous(out_shape)
    };
    let out = Tensor::from_storage(storage, layout, a.requires_grad);

    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(MeanGrad { in_shape, axis, axis_size, input_id: a.node }),
        ));
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{backward, Node, Tape, TensorStore};
    use crate::backend::CpuBackend;
    use crate::autograd::context::Context;
    use crate::tensor::Device;

    fn ctx() -> Context {
        Context::new(Arc::new(CpuBackend), Device::Cpu)
    }

    fn leaf(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_vec(data, shape, Device::Cpu).with_grad()
    }

    // ── sum forward ──────────────────────────────────────────────────────

    #[test]
    fn sum_axis0_2d() {
        // [[1,2,3],[4,5,6]] sum axis=0 → [5,7,9]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = sum(&ctx, &mut tape, &a, 0);
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis1_2d() {
        // [[1,2,3],[4,5,6]] sum axis=1 → [6,15]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = sum(&ctx, &mut tape, &a, 1);
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn sum_axis0_1d() {
        // [1,2,3,4] sum axis=0 → scalar [10]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = sum(&ctx, &mut tape, &a, 0);
        assert_eq!(out.to_vec(), vec![10.0]);
    }

    // ── sum backward ─────────────────────────────────────────────────────

    #[test]
    fn sum_backward_axis0() {
        // Sum along axis 0: grad is broadcast back along that axis.
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = sum(&ctx, &mut tape, &a, 0);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Each output grad (1.0) is copied to both rows.
        assert_eq!(a.grad(&store), Some([1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice()));
    }

    #[test]
    fn sum_backward_axis1() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = sum(&ctx, &mut tape, &a, 1);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Grad [1,1] broadcast along axis 1 → all ones [2,3]
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
    }

    // ── mean forward ─────────────────────────────────────────────────────

    #[test]
    fn mean_axis0_2d() {
        // [[1,2,3],[3,4,5]] mean axis=0 → [2,3,4]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], vec![2, 3]);
        let out = mean(&ctx, &mut tape, &a, 0);
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mean_axis1_2d() {
        // [[1,2,3],[4,5,6]] mean axis=1 → [2,5]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = mean(&ctx, &mut tape, &a, 1);
        assert_eq!(out.to_vec(), vec![2.0, 5.0]);
    }

    // ── mean backward ────────────────────────────────────────────────────

    #[test]
    fn mean_backward_axis0() {
        // mean along axis 0: grad / axis_size broadcast back
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = mean(&ctx, &mut tape, &a, 0);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // grad = 1/2 for all input elements
        let g = a.grad(&store).expect("grad must exist"); // SAFE: requires_grad=true, backward was called
        for &v in g {
            assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {}", v);
        }
    }

    #[test]
    fn mean_backward_axis1() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = mean(&ctx, &mut tape, &a, 1);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // grad = 1/3 for all input elements
        let g = a.grad(&store).expect("grad must exist"); // SAFE: requires_grad=true, backward was called
        for &v in g {
            assert!((v - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {}", v);
        }
    }

    // ── chained ops ──────────────────────────────────────────────────────

    #[test]
    fn sum_then_mean_backward() {
        // sum axis=0, then mean axis=0 on result — gradient flows through both
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let s = sum(&ctx, &mut tape, &a, 0); // [5, 7, 9]
        let m = mean(&ctx, &mut tape, &s, 0); // [7.0]
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &m);
        // grad through mean: 1/3 → sum backward: 1/3 for all 6 elements
        let g = a.grad(&store).expect("grad must exist"); // SAFE: requires_grad=true, backward was called
        for &v in g {
            assert!((v - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {}", v);
        }
    }
}
