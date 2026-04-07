use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::backend::Backend;
use crate::tensor::{Layout, Storage, Tensor};

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// ── GradFn ────────────────────────────────────────────────────────────────────

struct SdpaGrad {
    q_storage: Arc<Storage>,
    q_layout: Layout,
    k_storage: Arc<Storage>,
    k_layout: Layout,
    v_storage: Arc<Storage>,
    v_layout: Layout,
    /// Attention weights: `softmax(Q @ K^T / sqrt(d_k))`.
    /// Shape: `[batch_size * num_heads * seq_q * seq_k]`, contiguous.
    attn_weights: Arc<Storage>,
    batch_size: usize,
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
    backend: Arc<dyn Backend>,
}

impl GradFn for SdpaGrad {
    /// Backward pass for scaled dot-product attention.
    ///
    /// Given upstream gradient `dout` of shape `[B, H, Sq, Dv]`:
    ///
    /// ```text
    /// dV          = attn_weights^T  @ dout        [B, H, Sk, Dv]
    /// d_attn      = dout            @ V^T          [B, H, Sq, Sk]
    /// d_scores    = softmax_bwd(attn_weights, d_attn)  [B, H, Sq, Sk]
    /// d_scores   *= 1 / sqrt(d_k)
    /// dQ          = d_scores        @ K            [B, H, Sq, Dk]
    /// dK          = d_scores^T      @ Q            [B, H, Sk, Dk]
    /// ```
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let q_tmp = Tensor::from_storage(Arc::clone(&self.q_storage), self.q_layout.clone(), false);
        let k_tmp = Tensor::from_storage(Arc::clone(&self.k_storage), self.k_layout.clone(), false);
        let v_tmp = Tensor::from_storage(Arc::clone(&self.v_storage), self.v_layout.clone(), false);
        let q_data = q_tmp.to_vec();
        let k_data = k_tmp.to_vec();
        let v_data = v_tmp.to_vec();
        let aw = &self.attn_weights.data;

        let scale = 1.0f32 / (self.d_k as f32).sqrt();
        let nh = self.num_heads;
        let (sq, sk, dk, dv) = (self.seq_q, self.seq_k, self.d_k, self.d_v);

        let mut dq = vec![0.0f32; self.batch_size * nh * sq * dk];
        let mut dk_out = vec![0.0f32; self.batch_size * nh * sk * dk];
        let mut dv_out = vec![0.0f32; self.batch_size * nh * sk * dv];

        for b in 0..self.batch_size {
            for h in 0..nh {
                let bh = b * nh + h;
                let q_off = bh * sq * dk;
                let k_off = bh * sk * dk;
                let v_off = bh * sk * dv;
                let out_off = bh * sq * dv;
                let aw_off = bh * sq * sk;

                let dout_slice = &grad_output[out_off..out_off + sq * dv];
                let aw_slice = &aw[aw_off..aw_off + sq * sk];
                let v_slice = &v_data[v_off..v_off + sk * dv];
                let q_slice = &q_data[q_off..q_off + sq * dk];
                let k_slice = &k_data[k_off..k_off + sk * dk];

                // dV = attn_weights^T @ dout : [sk,sq] @ [sq,dv] → [sk,dv]
                let aw_t = transpose_2d(aw_slice, sq, sk);
                self.backend.matmul(
                    &aw_t,
                    dout_slice,
                    &mut dv_out[v_off..v_off + sk * dv],
                    sk,
                    sq,
                    dv,
                );

                // d_attn = dout @ V^T : [sq,dv] @ [dv,sk] → [sq,sk]
                let v_t = transpose_2d(v_slice, sk, dv);
                let mut d_attn = vec![0.0f32; sq * sk];
                self.backend
                    .matmul(dout_slice, &v_t, &mut d_attn, sq, dv, sk);

                // Softmax backward: d_scores[r,c] = aw[r,c] * (d_attn[r,c] - dot_r)
                // where dot_r = Σ_j d_attn[r,j] * aw[r,j].
                let mut d_scores = vec![0.0f32; sq * sk];
                for r in 0..sq {
                    let mut dot = 0.0f32;
                    for c in 0..sk {
                        dot += d_attn[r * sk + c] * aw_slice[r * sk + c];
                    }
                    for c in 0..sk {
                        d_scores[r * sk + c] = aw_slice[r * sk + c] * (d_attn[r * sk + c] - dot);
                    }
                }

                // Scale by 1/sqrt(d_k).
                for x in d_scores.iter_mut() {
                    *x *= scale;
                }

                // dQ = d_scores @ K : [sq,sk] @ [sk,dk] → [sq,dk]
                self.backend.matmul(
                    &d_scores,
                    k_slice,
                    &mut dq[q_off..q_off + sq * dk],
                    sq,
                    sk,
                    dk,
                );

                // dK = d_scores^T @ Q : [sk,sq] @ [sq,dk] → [sk,dk]
                let ds_t = transpose_2d(&d_scores, sq, sk);
                self.backend.matmul(
                    &ds_t,
                    q_slice,
                    &mut dk_out[k_off..k_off + sk * dk],
                    sk,
                    sq,
                    dk,
                );
            }
        }

        vec![dq, dk_out, dv_out]
    }

    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.q_id, self.k_id, self.v_id]
    }
}

// ── scaled_dot_product_attention ──────────────────────────────────────────────

/// Fused scaled dot-product attention.
///
/// Computes:
///
/// ```text
/// out = softmax(Q @ K^T / sqrt(d_k)) @ V
/// ```
///
/// All three inputs must be 4-D with shape `[batch, heads, seq, dim]`:
///
/// - `q` : `[batch, heads, seq_q, d_k]`
/// - `k` : `[batch, heads, seq_k, d_k]`
/// - `v` : `[batch, heads, seq_k, d_v]`
///
/// Returns a tensor of shape `[batch, heads, seq_q, d_v]`.
///
/// The forward pass is fused: scores, softmax, and the final projection are
/// computed in a single backend call to avoid unnecessary allocations. The
/// attention weights are saved internally for the backward pass.
///
/// # Panics
///
/// Panics if any input is not 4-D, batch/head dimensions don't match across
/// inputs, or the key dimension of Q and K differs.
pub fn scaled_dot_product_attention(
    ctx: &Context,
    tape: &mut Tape,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Tensor {
    assert_eq!(q.ndim(), 4, "sdpa: q must be 4-D, got {}D", q.ndim());
    assert_eq!(k.ndim(), 4, "sdpa: k must be 4-D, got {}D", k.ndim());
    assert_eq!(v.ndim(), 4, "sdpa: v must be 4-D, got {}D", v.ndim());

    let batch = q.shape()[0];
    let heads = q.shape()[1];
    let seq_q = q.shape()[2];
    let d_k = q.shape()[3];

    assert_eq!(
        k.shape()[..2],
        [batch, heads],
        "sdpa: k batch/heads {:?} != q batch/heads [{batch}, {heads}]",
        &k.shape()[..2]
    );
    assert_eq!(
        v.shape()[..2],
        [batch, heads],
        "sdpa: v batch/heads {:?} != q batch/heads [{batch}, {heads}]",
        &v.shape()[..2]
    );
    let seq_k = k.shape()[2];
    assert_eq!(
        k.shape()[3],
        d_k,
        "sdpa: k d_k={} != q d_k={d_k}",
        k.shape()[3]
    );
    assert_eq!(
        v.shape()[2],
        seq_k,
        "sdpa: v seq_k={} != k seq_k={seq_k}",
        v.shape()[2]
    );
    let d_v = v.shape()[3];

    let q_data = q.to_vec();
    let k_data = k.to_vec();
    let v_data = v.to_vec();

    let mut out_data = vec![0.0f32; batch * heads * seq_q * d_v];
    let mut aw_data = vec![0.0f32; batch * heads * seq_q * seq_k];

    ctx.backend.sdpa_forward(
        &q_data,
        &k_data,
        &v_data,
        &mut out_data,
        &mut aw_data,
        batch,
        heads,
        seq_q,
        seq_k,
        d_k,
        d_v,
    );

    let requires_grad = q.requires_grad || k.requires_grad || v.requires_grad;
    let aw_storage = Arc::new(Storage::new(aw_data, q.device(), q.dtype()));
    let out_storage = Arc::new(Storage::new(out_data, q.device(), q.dtype()));
    let layout = Layout::contiguous(vec![batch, heads, seq_q, d_v]);
    let out = Tensor::from_storage(Arc::clone(&out_storage), layout, requires_grad);

    if requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(SdpaGrad {
                q_storage: Arc::clone(&q.storage),
                q_layout: q.layout.clone(),
                k_storage: Arc::clone(&k.storage),
                k_layout: k.layout.clone(),
                v_storage: Arc::clone(&v.storage),
                v_layout: v.layout.clone(),
                attn_weights: aw_storage,
                batch_size: batch,
                num_heads: heads,
                seq_q,
                seq_k,
                d_k,
                d_v,
                q_id: q.node,
                k_id: k.node,
                v_id: v.node,
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
    fn sdpa_output_shape() {
        // [1,2,3,4] × [1,2,5,4] → [1,2,3,6]
        let ctx = ctx();
        let mut tape = Tape::new();
        let q = leaf(vec![0.1f32; 24], vec![1, 2, 3, 4]);
        let k = leaf(vec![0.1f32; 40], vec![1, 2, 5, 4]);
        let v = leaf(vec![0.1f32; 60], vec![1, 2, 5, 6]);
        let out = scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
        assert_eq!(out.shape(), &[1, 2, 3, 6]);
    }

    #[test]
    fn sdpa_attn_rows_sum_to_one() {
        // Verify the softmax step: each query position's attention over keys sums to 1.
        // Use a separate call via the backend, but verify through the op output indirectly
        // by checking V=all-ones → output equals 1 at every position.
        let ctx = ctx();
        let mut tape = Tape::new();
        let q = leaf(vec![1.0f32, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = leaf(vec![1.0f32, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        // V = all-ones: output[q] = sum_k(attn[q,k]) * 1 = 1 for each feature.
        let v = leaf(vec![1.0f32; 4], vec![1, 1, 2, 2]);
        let out = scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
        for &val in &out.to_vec() {
            assert!((val - 1.0).abs() < 1e-5, "expected 1.0, got {val}");
        }
    }

    #[test]
    fn sdpa_backward_grads_exist() {
        // After backward, all three inputs should have gradients.
        let ctx = ctx();
        let mut tape = Tape::new();
        let q = leaf(vec![1.0f32, 0.5, -0.5, 0.2], vec![1, 1, 2, 2]);
        let k = leaf(vec![0.3f32, -0.1, 0.7, 0.4], vec![1, 1, 2, 2]);
        let v = leaf(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        tape.push(Node::leaf(q.node));
        tape.push(Node::leaf(k.node));
        tape.push(Node::leaf(v.node));
        let out = scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert!(q.has_grad(&store), "q grad missing");
        assert!(k.has_grad(&store), "k grad missing");
        assert!(v.has_grad(&store), "v grad missing");
    }

    #[test]
    fn sdpa_backward_grad_shapes() {
        // Grad shapes must match input shapes exactly.
        let ctx = ctx();
        let mut tape = Tape::new();
        let q = leaf(vec![0.1f32; 2 * 3 * 4 * 8], vec![2, 3, 4, 8]);
        let k = leaf(vec![0.1f32; 2 * 3 * 5 * 8], vec![2, 3, 5, 8]);
        let v = leaf(vec![0.1f32; 2 * 3 * 5 * 6], vec![2, 3, 5, 6]);
        tape.push(Node::leaf(q.node));
        tape.push(Node::leaf(k.node));
        tape.push(Node::leaf(v.node));
        let out = scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(q.grad_strict(&store).len(), 2 * 3 * 4 * 8); // SAFE: requires_grad=true
        assert_eq!(k.grad_strict(&store).len(), 2 * 3 * 5 * 8); // SAFE: requires_grad=true
        assert_eq!(v.grad_strict(&store).len(), 2 * 3 * 5 * 6); // SAFE: requires_grad=true
    }

    #[test]
    #[should_panic(expected = "sdpa: q must be 4-D")]
    fn sdpa_wrong_rank_panics() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let q = leaf(vec![1.0f32; 4], vec![2, 2]);
        let k = leaf(vec![1.0f32; 4], vec![1, 1, 2, 2]);
        let v = leaf(vec![1.0f32; 4], vec![1, 1, 2, 2]);
        scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
    }
}
