use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::{Layout, Storage, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Materialize a tensor into a flat `Vec<f32>` broadcasted to `target_shape`.
///
/// If the tensor is already contiguous with `target_shape`, returns the raw
/// slice directly (zero-copy path via `to_vec`). Otherwise iterates with
/// strides, handling 0-stride broadcast dims.
fn broadcast_collect(tensor: &Tensor, target_shape: &[usize]) -> Vec<f32> {
    let bc_layout = tensor.layout.broadcast_to(target_shape);
    let numel: usize = target_shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let data = &tensor.storage.data;
    let ndim = bc_layout.ndim();

    if ndim == 0 {
        out.push(data[bc_layout.offset]);
        return out;
    }

    let mut indices = vec![0usize; ndim];
    for _ in 0..numel {
        let flat = bc_layout.flat_index(&indices);
        out.push(data[flat]);
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                indices[d] += 1;
                if indices[d] < bc_layout.shape[d] {
                    carry = false;
                } else {
                    indices[d] = 0;
                }
            }
        }
    }
    out
}

/// Reduce `grad` (shape = `out_shape`) back to `src_shape` by summing over
/// any axes that were broadcast or prepended.
///
/// This is the "unbroadcast" operation required in backward passes.
fn reduce_to_shape(grad: &[f32], out_shape: &[usize], src_shape: &[usize]) -> Vec<f32> {
    let src_numel: usize = src_shape.iter().product();
    let out_numel: usize = out_shape.iter().product();
    debug_assert_eq!(grad.len(), out_numel);

    if src_shape == out_shape {
        return grad.to_vec();
    }

    // Work in the padded space where src is left-padded with 1s.
    let ndim = out_shape.len();
    let pad = ndim - src_shape.len();

    // For each output element, accumulate into the src index.
    let mut result = vec![0.0f32; src_numel];

    // Strides for out_shape.
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Strides for src_shape (padded).
    let src_ndim = src_shape.len();
    let mut src_strides = vec![1usize; src_ndim];
    for i in (0..src_ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    let mut out_indices = vec![0usize; ndim];
    for &g in grad.iter() {
        // Compute src flat index: skip prepended dims, collapse broadcast dims.
        let src_flat: usize = (0..ndim)
            .filter(|&d| d >= pad)
            .map(|d| {
                let src_d = d - pad;
                let idx = if src_shape[src_d] == 1 {
                    0
                } else {
                    out_indices[d]
                };
                idx * src_strides[src_d]
            })
            .sum();
        result[src_flat] += g;

        // Advance out_indices.
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                out_indices[d] += 1;
                if out_indices[d] < out_shape[d] {
                    carry = false;
                } else {
                    out_indices[d] = 0;
                }
            }
        }
    }
    result
}

// ── add ──────────────────────────────────────────────────────────────────────

struct AddGrad {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    out_shape: Vec<usize>,
    a_id: NodeId,
    b_id: NodeId,
}

impl GradFn for AddGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        vec![
            reduce_to_shape(grad_output, &self.out_shape, &self.a_shape),
            reduce_to_shape(grad_output, &self.out_shape, &self.b_shape),
        ]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.a_id, self.b_id]
    }
}

/// Element-wise addition with NumPy-style broadcasting.
pub fn add(ctx: &Context, tape: &mut Tape, a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "add: tensors must be on the same device"
    );
    let out_shape = Layout::broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_collect(a, &out_shape);
    let b_bc = broadcast_collect(b, &out_shape);
    let mut data = vec![0.0f32; a_bc.len()];
    ctx.backend.add(&a_bc, &b_bc, &mut data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let storage = Arc::new(crate::tensor::Storage::new(data, a.device(), a.dtype()));
    let layout = Layout::contiguous(out_shape.clone());
    let out = Tensor::from_storage(storage, layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(AddGrad {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
                out_shape,
                a_id: a.node,
                b_id: b.node,
            }),
        ));
    }
    out
}

// ── sub ──────────────────────────────────────────────────────────────────────

struct SubGrad {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    out_shape: Vec<usize>,
    a_id: NodeId,
    b_id: NodeId,
}

impl GradFn for SubGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let neg: Vec<f32> = grad_output.iter().map(|&g| -g).collect();
        vec![
            reduce_to_shape(grad_output, &self.out_shape, &self.a_shape),
            reduce_to_shape(&neg, &self.out_shape, &self.b_shape),
        ]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.a_id, self.b_id]
    }
}

/// Element-wise subtraction with NumPy-style broadcasting.
pub fn sub(ctx: &Context, tape: &mut Tape, a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "sub: tensors must be on the same device"
    );
    let out_shape = Layout::broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_collect(a, &out_shape);
    let b_bc = broadcast_collect(b, &out_shape);
    let mut data = vec![0.0f32; a_bc.len()];
    ctx.backend.sub(&a_bc, &b_bc, &mut data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let storage = Arc::new(crate::tensor::Storage::new(data, a.device(), a.dtype()));
    let layout = Layout::contiguous(out_shape.clone());
    let out = Tensor::from_storage(storage, layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(SubGrad {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
                out_shape,
                a_id: a.node,
                b_id: b.node,
            }),
        ));
    }
    out
}

// ── mul ──────────────────────────────────────────────────────────────────────

struct MulGrad {
    a_data: Arc<Storage>,
    a_layout: Layout,
    b_data: Arc<Storage>,
    b_layout: Layout,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    out_shape: Vec<usize>,
    a_id: NodeId,
    b_id: NodeId,
}

impl GradFn for MulGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // grad_a = grad_out * b (broadcast b to out_shape, then unbroadcast)
        // grad_b = grad_out * a (broadcast a to out_shape, then unbroadcast)
        let b_tmp = Tensor::from_storage(Arc::clone(&self.b_data), self.b_layout.clone(), false);
        let a_tmp = Tensor::from_storage(Arc::clone(&self.a_data), self.a_layout.clone(), false);

        let b_bc = broadcast_collect(&b_tmp, &self.out_shape);
        let a_bc = broadcast_collect(&a_tmp, &self.out_shape);

        let grad_a_full: Vec<f32> = grad_output
            .iter()
            .zip(b_bc.iter())
            .map(|(&g, &b)| g * b)
            .collect();
        let grad_b_full: Vec<f32> = grad_output
            .iter()
            .zip(a_bc.iter())
            .map(|(&g, &a)| g * a)
            .collect();

        vec![
            reduce_to_shape(&grad_a_full, &self.out_shape, &self.a_shape),
            reduce_to_shape(&grad_b_full, &self.out_shape, &self.b_shape),
        ]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.a_id, self.b_id]
    }
}

/// Element-wise multiplication with NumPy-style broadcasting.
pub fn mul(ctx: &Context, tape: &mut Tape, a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "mul: tensors must be on the same device"
    );
    let out_shape = Layout::broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_collect(a, &out_shape);
    let b_bc = broadcast_collect(b, &out_shape);
    let mut data = vec![0.0f32; a_bc.len()];
    ctx.backend.mul(&a_bc, &b_bc, &mut data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let storage = Arc::new(crate::tensor::Storage::new(data, a.device(), a.dtype()));
    let layout = Layout::contiguous(out_shape.clone());
    let out = Tensor::from_storage(storage, layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(MulGrad {
                a_data: Arc::clone(&a.storage),
                a_layout: a.layout.clone(),
                b_data: Arc::clone(&b.storage),
                b_layout: b.layout.clone(),
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
                out_shape,
                a_id: a.node,
                b_id: b.node,
            }),
        ));
    }
    out
}

// ── div ──────────────────────────────────────────────────────────────────────

struct DivGrad {
    a_data: Arc<Storage>,
    a_layout: Layout,
    b_data: Arc<Storage>,
    b_layout: Layout,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    out_shape: Vec<usize>,
    a_id: NodeId,
    b_id: NodeId,
}

impl GradFn for DivGrad {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // grad_a = grad_out / b
        // grad_b = -grad_out * a / b^2
        let b_tmp = Tensor::from_storage(Arc::clone(&self.b_data), self.b_layout.clone(), false);
        let a_tmp = Tensor::from_storage(Arc::clone(&self.a_data), self.a_layout.clone(), false);

        let b_bc = broadcast_collect(&b_tmp, &self.out_shape);
        let a_bc = broadcast_collect(&a_tmp, &self.out_shape);

        let grad_a_full: Vec<f32> = grad_output
            .iter()
            .zip(b_bc.iter())
            .map(|(&g, &b)| g / b)
            .collect();
        let grad_b_full: Vec<f32> = grad_output
            .iter()
            .zip(a_bc.iter())
            .zip(b_bc.iter())
            .map(|((&g, &a), &b)| -g * a / (b * b))
            .collect();

        vec![
            reduce_to_shape(&grad_a_full, &self.out_shape, &self.a_shape),
            reduce_to_shape(&grad_b_full, &self.out_shape, &self.b_shape),
        ]
    }
    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.a_id, self.b_id]
    }
}

/// Element-wise division with NumPy-style broadcasting.
pub fn div(ctx: &Context, tape: &mut Tape, a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "div: tensors must be on the same device"
    );
    let out_shape = Layout::broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_collect(a, &out_shape);
    let b_bc = broadcast_collect(b, &out_shape);
    let mut data = vec![0.0f32; a_bc.len()];
    ctx.backend.div(&a_bc, &b_bc, &mut data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let storage = Arc::new(crate::tensor::Storage::new(data, a.device(), a.dtype()));
    let layout = Layout::contiguous(out_shape.clone());
    let out = Tensor::from_storage(storage, layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(DivGrad {
                a_data: Arc::clone(&a.storage),
                a_layout: a.layout.clone(),
                b_data: Arc::clone(&b.storage),
                b_layout: b.layout.clone(),
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
                out_shape,
                a_id: a.node,
                b_id: b.node,
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
    use crate::autograd::{backward, TensorStore};
    use crate::backend::CpuBackend;

    fn ctx() -> Context {
        Context::new(Arc::new(CpuBackend), crate::tensor::Device::Cpu)
    }

    fn leaf(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_vec(data, shape, crate::tensor::Device::Cpu).with_grad()
    }

    // ── add ──────────────────────────────────────────────────────────────

    #[test]
    fn add_forward_same_shape() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![3]);
        let b = leaf(vec![4.0, 5.0, 6.0], vec![3]);
        let out = add(&ctx, &mut tape, &a, &b);
        assert_eq!(out.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_backward_same_shape() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0], vec![3]);
        let b = leaf(vec![4.0, 5.0, 6.0], vec![3]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = add(&ctx, &mut tape, &a, &b);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0, 1.0, 1.0].as_slice()));
        assert_eq!(b.grad(&store), Some([1.0, 1.0, 1.0].as_slice()));
    }

    #[test]
    fn add_backward_broadcast_row() {
        // a: [2,3], b: [3] — b is broadcast over rows
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0; 6], vec![2, 3]);
        let b = leaf(vec![1.0; 3], vec![3]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = add(&ctx, &mut tape, &a, &b);
        assert_eq!(out.shape(), &[2, 3]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // grad_a: all ones (shape [2,3])
        assert_eq!(a.grad(&store), Some([1.0f32; 6].as_slice()));
        // grad_b: summed over rows → [2,2,2]
        assert_eq!(b.grad(&store), Some([2.0, 2.0, 2.0].as_slice()));
    }

    // ── sub ──────────────────────────────────────────────────────────────

    #[test]
    fn sub_forward_and_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![5.0, 3.0], vec![2]);
        let b = leaf(vec![2.0, 1.0], vec![2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = sub(&ctx, &mut tape, &a, &b);
        assert_eq!(out.to_vec(), vec![3.0, 2.0]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([1.0, 1.0].as_slice()));
        assert_eq!(b.grad(&store), Some([-1.0, -1.0].as_slice()));
    }

    // ── mul ──────────────────────────────────────────────────────────────

    #[test]
    fn mul_forward_and_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![2.0, 3.0], vec![2]);
        let b = leaf(vec![4.0, 5.0], vec![2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = mul(&ctx, &mut tape, &a, &b);
        assert_eq!(out.to_vec(), vec![8.0, 15.0]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(a.grad(&store), Some([4.0, 5.0].as_slice()));
        assert_eq!(b.grad(&store), Some([2.0, 3.0].as_slice()));
    }

    #[test]
    fn mul_backward_broadcast() {
        // a: [2,2], b: [2] (broadcast) — grad_b must be summed over rows
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = leaf(vec![2.0, 3.0], vec![2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = mul(&ctx, &mut tape, &a, &b);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // grad_a = b broadcast: [[2,3],[2,3]]
        assert_eq!(a.grad(&store), Some([2.0, 3.0, 2.0, 3.0].as_slice()));
        // grad_b = sum over rows of a: [1+3, 2+4] = [4, 6]
        assert_eq!(b.grad(&store), Some([4.0, 6.0].as_slice()));
    }

    // ── div ──────────────────────────────────────────────────────────────

    #[test]
    fn div_forward_and_backward() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![6.0, 8.0], vec![2]);
        let b = leaf(vec![2.0, 4.0], vec![2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = div(&ctx, &mut tape, &a, &b);
        assert_eq!(out.to_vec(), vec![3.0, 2.0]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // grad_a = 1/b = [0.5, 0.25]
        assert_eq!(a.grad(&store), Some([0.5, 0.25].as_slice()));
        // grad_b = -a/b^2 = [-6/4, -8/16] = [-1.5, -0.5]
        assert_eq!(b.grad(&store), Some([-1.5, -0.5].as_slice()));
    }
}
