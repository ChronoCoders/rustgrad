use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::Tensor;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut st = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

/// Compute the inverse of a permutation.
/// `inv_perm[perm[i]] = i`
fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Materialize `grad` (contiguous, in `out_shape` order) into `in_shape`
/// order by applying the inverse permutation.
///
/// `out_shape = in_shape[perm[0]], in_shape[perm[1]], ...`
///
/// The backward for permute is: `grad_input[i0,i1,...] = grad_output[i_perm[0], i_perm[1], ...]`
fn permute_grad(grad: &[f32], in_shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let numel: usize = in_shape.iter().product();
    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();

    let in_strides = row_major_strides(in_shape);
    let out_strides = row_major_strides(&out_shape);

    let mut result = vec![0.0f32; numel];
    let mut in_indices = vec![0usize; in_shape.len()];

    for _ in 0..numel {
        let in_flat: usize = in_indices
            .iter()
            .zip(in_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        // Apply perm to in_indices to get out_indices.
        let out_flat: usize = perm
            .iter()
            .zip(out_strides.iter())
            .map(|(&p, &s)| in_indices[p] * s)
            .sum();
        result[in_flat] = grad[out_flat];

        // Advance in_indices.
        let mut carry = true;
        for d in (0..in_shape.len()).rev() {
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

// ── reshape ───────────────────────────────────────────────────────────────────

struct ReshapeGrad {
    input_id: NodeId,
}

impl GradFn for ReshapeGrad {
    /// Reshape backward: grad data order is identical to input data order
    /// (reshape never reorders elements, only reinterprets the shape).
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        vec![grad_output.to_vec()]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Reinterpret the tensor as `new_shape` without copying data.
///
/// The tensor must be contiguous. `new_shape` must have the same number
/// of elements as the original shape.
///
/// # Panics
///
/// Panics if the tensor is not contiguous, or if element counts differ.
pub fn reshape(_ctx: &Context, tape: &mut Tape, a: &Tensor, new_shape: Vec<usize>) -> Tensor {
    let new_layout = a.layout.reshape(new_shape); // asserts contiguous + numel match
    let out = Tensor::from_storage(Arc::clone(&a.storage), new_layout, a.requires_grad);
    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(ReshapeGrad { input_id: a.node }),
        ));
    }
    out
}

// ── permute ───────────────────────────────────────────────────────────────────

struct PermuteGrad {
    in_shape: Vec<usize>,
    perm: Vec<usize>,
    input_id: NodeId,
}

impl GradFn for PermuteGrad {
    /// Permute backward: apply the inverse permutation to route each
    /// upstream grad element back to its original position.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let inv = invert_perm(&self.perm);
        vec![permute_grad(grad_output, &self.in_shape, &inv)]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Reorder dimensions according to `dims`. Zero-copy: produces a new
/// `Layout` over the same storage.
///
/// `dims` must be a permutation of `0..a.ndim()`.
///
/// # Panics
///
/// Panics if `dims` is not a valid permutation.
pub fn permute(_ctx: &Context, tape: &mut Tape, a: &Tensor, dims: Vec<usize>) -> Tensor {
    let in_shape = a.shape().to_vec();
    let new_layout = a.layout.permute(&dims);
    let out = Tensor::from_storage(Arc::clone(&a.storage), new_layout, a.requires_grad);
    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(PermuteGrad {
                in_shape,
                perm: dims,
                input_id: a.node,
            }),
        ));
    }
    out
}

// ── transpose ────────────────────────────────────────────────────────────────

struct TransposeGrad {
    in_shape: Vec<usize>,
    dim0: usize,
    dim1: usize,
    input_id: NodeId,
}

impl GradFn for TransposeGrad {
    /// Transpose backward: apply the same transpose. Swapping the same two
    /// dims twice is the identity.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // Build the permutation that swaps dim0 and dim1.
        let ndim = self.in_shape.len();
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(self.dim0, self.dim1);
        // out_shape for the transposed grad:
        let out_shape: Vec<usize> = perm.iter().map(|&p| self.in_shape[p]).collect();
        let _ = out_shape;
        // Apply inverse perm to route grad back to in_shape order.
        // Since transpose is its own inverse, inv_perm == perm.
        vec![permute_grad(grad_output, &self.in_shape, &perm)]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Swap two dimensions. Zero-copy: produces a new `Layout`.
///
/// # Panics
///
/// Panics if `dim0` or `dim1` are out of range.
pub fn transpose(_ctx: &Context, tape: &mut Tape, a: &Tensor, dim0: usize, dim1: usize) -> Tensor {
    let in_shape = a.shape().to_vec();
    let new_layout = a.layout.transpose(dim0, dim1);
    let out = Tensor::from_storage(Arc::clone(&a.storage), new_layout, a.requires_grad);
    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(TransposeGrad {
                in_shape,
                dim0,
                dim1,
                input_id: a.node,
            }),
        ));
    }
    out
}

// ── squeeze ───────────────────────────────────────────────────────────────────

struct SqueezeGrad {
    in_shape: Vec<usize>,
    input_id: NodeId,
}

impl GradFn for SqueezeGrad {
    /// Squeeze backward: grad data is in the same order; just reinterpret
    /// with the original shape.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        debug_assert_eq!(grad_output.len(), self.in_shape.iter().product::<usize>());
        vec![grad_output.to_vec()]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Remove size-1 dimensions. If `axis` is `Some(a)`, removes only that
/// axis (which must have size 1). If `None`, removes all size-1 axes.
/// Zero-copy.
///
/// # Panics
///
/// Panics if `axis` is out of range or its size is not 1.
pub fn squeeze(_ctx: &Context, tape: &mut Tape, a: &Tensor, axis: Option<usize>) -> Tensor {
    let in_shape = a.shape().to_vec();
    let new_layout = a.layout.squeeze(axis);
    let out = Tensor::from_storage(Arc::clone(&a.storage), new_layout, a.requires_grad);
    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(SqueezeGrad {
                in_shape,
                input_id: a.node,
            }),
        ));
    }
    out
}

// ── unsqueeze ────────────────────────────────────────────────────────────────

struct UnsqueezeGrad {
    in_shape: Vec<usize>,
    input_id: NodeId,
}

impl GradFn for UnsqueezeGrad {
    /// Unsqueeze backward: grad data is in the same order; reinterpret with
    /// the original shape.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        debug_assert_eq!(grad_output.len(), self.in_shape.iter().product::<usize>());
        vec![grad_output.to_vec()]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// Insert a size-1 dimension at position `axis`. Zero-copy.
///
/// # Panics
///
/// Panics if `axis > a.ndim()`.
pub fn unsqueeze(_ctx: &Context, tape: &mut Tape, a: &Tensor, axis: usize) -> Tensor {
    let in_shape = a.shape().to_vec();
    let new_layout = a.layout.unsqueeze(axis);
    let out = Tensor::from_storage(Arc::clone(&a.storage), new_layout, a.requires_grad);
    if a.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(UnsqueezeGrad {
                in_shape,
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

    // ── reshape ──────────────────────────────────────────────────────────

    #[test]
    fn reshape_forward_zero_copy() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reshape(&ctx, &mut tape, &a, vec![3, 2]);
        // Same storage pointer — no copy.
        assert!(Arc::ptr_eq(&a.storage, &out.storage));
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = reshape(&ctx, &mut tape, &a, vec![3, 2]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Grad passes through unchanged (same flat data order).
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
    }

    // ── permute ──────────────────────────────────────────────────────────

    #[test]
    fn permute_forward_zero_copy() {
        let ctx = ctx();
        let mut tape = Tape::new();
        // Shape [2,3], permute [1,0] → [3,2]
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = permute(&ctx, &mut tape, &a, vec![1, 0]);
        assert!(Arc::ptr_eq(&a.storage, &out.storage));
        assert_eq!(out.shape(), &[3, 2]);
        // [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]] = [1,4,2,5,3,6]
        assert_eq!(out.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn permute_backward() {
        // grad flows back via inverse permutation
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = permute(&ctx, &mut tape, &a, vec![1, 0]); // [2,3] → [3,2]
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Upstream grad is all-ones in [3,2] order.
        // Inverse permute routes each 1.0 back to its original position.
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
    }

    #[test]
    fn permute_3d_backward() {
        // [2,3,4] permuted [2,0,1] → [4,2,3]
        // Backward: all-ones grad should flow back as all-ones.
        let ctx = ctx();
        let mut tape = Tape::new();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a = leaf(data, vec![2, 3, 4]);
        tape.push(Node::leaf(a.node));
        let out = permute(&ctx, &mut tape, &a, vec![2, 0, 1]);
        assert_eq!(out.shape(), &[4, 2, 3]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0f32; 24].as_slice()));
    }

    // ── transpose ────────────────────────────────────────────────────────

    #[test]
    fn transpose_forward_zero_copy() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = transpose(&ctx, &mut tape, &a, 0, 1);
        assert!(Arc::ptr_eq(&a.storage, &out.storage));
        assert_eq!(out.shape(), &[2, 2]);
        // [[1,2],[3,4]] transposed → [[1,3],[2,4]] = [1,3,2,4]
        assert_eq!(out.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_backward_is_its_own_inverse() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let out = transpose(&ctx, &mut tape, &a, 0, 1);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
    }

    // ── squeeze ──────────────────────────────────────────────────────────

    #[test]
    fn squeeze_all_forward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let out = squeeze(&ctx, &mut tape, &a, None);
        assert!(Arc::ptr_eq(&a.storage, &out.storage));
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn squeeze_axis_forward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = squeeze(&ctx, &mut tape, &a, Some(0));
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn squeeze_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![1, 3]);
        tape.push(Node::leaf(a.node));
        let out = squeeze(&ctx, &mut tape, &a, Some(0));
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0f32; 3].as_slice()));
    }

    // ── unsqueeze ────────────────────────────────────────────────────────

    #[test]
    fn unsqueeze_forward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![3]);
        let out = unsqueeze(&ctx, &mut tape, &a, 0);
        assert!(Arc::ptr_eq(&a.storage, &out.storage));
        assert_eq!(out.shape(), &[1, 3]);
    }

    #[test]
    fn unsqueeze_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![3]);
        tape.push(Node::leaf(a.node));
        let out = unsqueeze(&ctx, &mut tape, &a, 1); // [3] → [3,1]
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0f32; 3].as_slice()));
    }

    // ── chained shape ops ────────────────────────────────────────────────

    #[test]
    fn unsqueeze_then_squeeze_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tape.push(Node::leaf(a.node));
        let u = unsqueeze(&ctx, &mut tape, &a, 0); // [1,2,2]
        let s = squeeze(&ctx, &mut tape, &u, Some(0)); // [2,2]
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &s);
        assert_eq!(a.grad(&store), Some([1.0f32; 4].as_slice()));
    }

    #[test]
    fn transpose_then_sum_backward() {
        // Transpose [2,3] → [3,2], sum axis=1 → [3]
        // grad should flow back through both ops correctly.
        use crate::ops::reduction::sum;
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        tape.push(Node::leaf(a.node));
        let t = transpose(&ctx, &mut tape, &a, 0, 1); // [3,2]
        let s = sum(&ctx, &mut tape, &t, 1); // [3]
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &s);
        // Each element gets gradient 1.0 (sum passes 1 to all, transpose passes it back).
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
    }
}
