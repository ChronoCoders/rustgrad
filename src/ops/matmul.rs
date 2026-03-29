use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::backend::Backend;
use crate::tensor::{Layout, Storage, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Transpose a row-major `[rows, cols]` matrix into `[cols, rows]`.
fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

// ── GradFn ───────────────────────────────────────────────────────────────────

struct MatMulGrad {
    /// Captured input A storage and layout (needed to compute grad_b).
    a_storage: Arc<Storage>,
    a_layout: Layout,
    /// Captured input B storage and layout (needed to compute grad_a).
    b_storage: Arc<Storage>,
    b_layout: Layout,
    /// Matrix dimensions from the forward pass.
    m: usize,
    k: usize,
    n: usize,
    /// Product of all batch dimensions (1 for plain 2-D matmul).
    batch_size: usize,
    a_id: NodeId,
    b_id: NodeId,
    /// Backend for matmul dispatch in the backward.
    backend: Arc<dyn Backend>,
}

impl GradFn for MatMulGrad {
    /// Backward for `out = A @ B`.
    ///
    /// `grad_A = grad_out @ B^T`  shape: `[..., m, k]`
    /// `grad_B = A^T   @ grad_out` shape: `[..., k, n]`
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let a_tmp =
            Tensor::from_storage(Arc::clone(&self.a_storage), self.a_layout.clone(), false);
        let b_tmp =
            Tensor::from_storage(Arc::clone(&self.b_storage), self.b_layout.clone(), false);
        let a_data = a_tmp.to_vec();
        let b_data = b_tmp.to_vec();

        let mut grad_a = vec![0.0f32; self.batch_size * self.m * self.k];
        let mut grad_b = vec![0.0f32; self.batch_size * self.k * self.n];

        for i in 0..self.batch_size {
            let g = &grad_output[i * self.m * self.n..(i + 1) * self.m * self.n];
            let a_slice = &a_data[i * self.m * self.k..(i + 1) * self.m * self.k];
            let b_slice = &b_data[i * self.k * self.n..(i + 1) * self.k * self.n];

            // grad_a[i] = g @ B^T : [m,n] @ [n,k] = [m,k]
            let b_t = transpose_2d(b_slice, self.k, self.n);
            self.backend.matmul(
                g,
                &b_t,
                &mut grad_a[i * self.m * self.k..(i + 1) * self.m * self.k],
                self.m,
                self.n,
                self.k,
            );

            // grad_b[i] = A^T @ g : [k,m] @ [m,n] = [k,n]
            let a_t = transpose_2d(a_slice, self.m, self.k);
            self.backend.matmul(
                &a_t,
                g,
                &mut grad_b[i * self.k * self.n..(i + 1) * self.k * self.n],
                self.k,
                self.m,
                self.n,
            );
        }

        vec![grad_a, grad_b]
    }

    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.a_id, self.b_id]
    }
}

// ── matmul ────────────────────────────────────────────────────────────────────

/// Matrix multiplication: `out = A @ B`.
///
/// Supports 2-D `[m, k] @ [k, n]` and batched N-D
/// `[..., m, k] @ [..., k, n]` where batch dimensions must match exactly.
/// Faer handles AVX2/AVX-512 dispatch inside the CPU backend.
///
/// For a `[batch, seq, d]` × `[d, d_out]` linear projection, reshape the
/// input to `[batch*seq, d]`, call `matmul`, then reshape the result.
///
/// # Panics
///
/// Panics if either tensor has fewer than 2 dimensions, if the inner
/// (k) dimensions do not match, if ranks differ, or if batch dimensions
/// do not match.
pub fn matmul(ctx: &Context, tape: &mut Tape, a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "matmul: tensors must be on the same device"
    );
    assert!(
        a.ndim() >= 2,
        "matmul: a must have at least 2 dimensions, got {}",
        a.ndim()
    );
    assert!(
        b.ndim() >= 2,
        "matmul: b must have at least 2 dimensions, got {}",
        b.ndim()
    );
    assert_eq!(
        a.ndim(),
        b.ndim(),
        "matmul: a and b must have the same rank ({} vs {})",
        a.ndim(),
        b.ndim()
    );

    let ndim = a.ndim();
    let m = a.shape()[ndim - 2];
    let k = a.shape()[ndim - 1];
    let k2 = b.shape()[ndim - 2];
    let n = b.shape()[ndim - 1];

    assert_eq!(
        k, k2,
        "matmul: inner dimensions must match — A has k={}, B has k={}",
        k, k2
    );

    let batch_shape = &a.shape()[..ndim - 2];
    assert_eq!(
        batch_shape,
        &b.shape()[..ndim - 2],
        "matmul: batch dimensions must match — A batch {:?}, B batch {:?}",
        batch_shape,
        &b.shape()[..ndim - 2]
    );

    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    // Materialize inputs — handles non-contiguous layouts (e.g., transposed K).
    let a_data = a.to_vec();
    let b_data = b.to_vec();

    let mut out_data = vec![0.0f32; batch_size * m * n];
    for i in 0..batch_size {
        ctx.backend.matmul(
            &a_data[i * m * k..(i + 1) * m * k],
            &b_data[i * k * n..(i + 1) * k * n],
            &mut out_data[i * m * n..(i + 1) * m * n],
            m,
            k,
            n,
        );
    }

    let mut out_shape = batch_shape.to_vec();
    out_shape.push(m);
    out_shape.push(n);

    let requires_grad = a.requires_grad || b.requires_grad;
    let storage = Arc::new(Storage::new(out_data, a.device(), a.dtype()));
    let layout = Layout::contiguous(out_shape);
    let out = Tensor::from_storage(storage, layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(MatMulGrad {
                a_storage: Arc::clone(&a.storage),
                a_layout: a.layout.clone(),
                b_storage: Arc::clone(&b.storage),
                b_layout: b.layout.clone(),
                m,
                k,
                n,
                batch_size,
                a_id: a.node,
                b_id: b.node,
                backend: Arc::clone(&ctx.backend),
            }),
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

    // ── forward ──────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x2_forward() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = leaf(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let out = matmul(&ctx, &mut tape, &a, &b);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_2x3_times_3x4() {
        // [2,3] @ [3,4] → [2,4]
        let ctx = ctx();
        let mut tape = Tape::new();
        // A = [[1,0,0],[0,1,0]], B = identity-ish 3x4
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = leaf(
            vec![1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0],
            vec![3, 4],
        );
        let out = matmul(&ctx, &mut tape, &a, &b);
        assert_eq!(out.shape(), &[2, 4]);
        // A @ partial-identity → first 3 cols of A, 4th col zero
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn matmul_batched_3d() {
        // [2,2,2] @ [2,2,2] — two batched 2x2 matmuls
        let ctx = ctx();
        let mut tape = Tape::new();
        // Batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        // Batch 1: [[1,0],[0,1]] @ [[5,6],[7,8]] = [[5,6],[7,8]]
        let a = leaf(
            vec![1.0, 2.0, 3.0, 4.0,  // batch 0
                 1.0, 0.0, 0.0, 1.0], // batch 1 (identity)
            vec![2, 2, 2],
        );
        let b = leaf(
            vec![1.0, 0.0, 0.0, 1.0,  // batch 0 (identity)
                 5.0, 6.0, 7.0, 8.0], // batch 1
            vec![2, 2, 2],
        );
        let out = matmul(&ctx, &mut tape, &a, &b);
        assert_eq!(out.shape(), &[2, 2, 2]);
        assert_eq!(
            out.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    // ── backward ─────────────────────────────────────────────────────────

    #[test]
    fn matmul_backward_2d() {
        // A=[2,3], B=[3,2]. grad_A = G @ B^T, grad_B = A^T @ G
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = matmul(&ctx, &mut tape, &a, &b);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);

        // With all-ones upstream grad (seeded by backward()):
        // grad_A = ones([2,2]) @ B^T = ones([2,2]) @ [[1,3,5],[2,4,6]]
        //        = [[3,7,11],[3,7,11]]
        let ga = a.grad(&store).expect("grad_a must exist"); // SAFE: requires_grad=true
        assert_eq!(ga, &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);

        // grad_B = A^T @ ones([2,2]) = [[1,4],[2,5],[3,6]] @ ones([2,2])
        //        = [[5,5],[7,7],[9,9]]
        let gb = b.grad(&store).expect("grad_b must exist"); // SAFE: requires_grad=true
        assert_eq!(gb, &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn matmul_backward_batched() {
        // [2,2,2] @ [2,2,2] batched — grads should be correct per batch.
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![2, 2, 2]);
        let b = leaf(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![2, 2, 2]);
        tape.push(Node::leaf(a.node));
        tape.push(Node::leaf(b.node));
        let out = matmul(&ctx, &mut tape, &a, &b);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // A and B are identity matrices — grad_A = G @ I^T = G, grad_B = I^T @ G = G
        // upstream G is all-ones → grad_a = [[2,2],[2,2],[2,2],[2,2]] (sums of rows)
        let ga = a.grad(&store).expect("grad_a"); // SAFE: requires_grad=true
        assert_eq!(ga, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn matmul_with_transposed_b() {
        // Simulate Q @ K^T via ops::shape::transpose then matmul.
        use crate::ops::shape::transpose as tr;
        let ctx = ctx();
        let mut tape = Tape::new();
        // Q: [2,3], K: [2,3] (same as Q) → K^T: [3,2]
        let q = leaf(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let k = leaf(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        tape.push(Node::leaf(q.node));
        tape.push(Node::leaf(k.node));
        let kt = tr(&ctx, &mut tape, &k, 0, 1); // [3,2]
        let out = matmul(&ctx, &mut tape, &q, &kt); // [2,2]
        // Q @ Q^T of identity-rows = identity 2x2
        assert_eq!(out.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert!(q.has_grad(&store));
        assert!(k.has_grad(&store));
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn matmul_shape_mismatch_panics() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let a = leaf(vec![1.0; 6], vec![2, 3]);
        let b = leaf(vec![1.0; 8], vec![4, 2]); // k=3 vs k=4 mismatch
        matmul(&ctx, &mut tape, &a, &b);
    }
}
