use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::{Layout, Storage, Tensor};

// ── GradFn ────────────────────────────────────────────────────────────────────

struct EmbeddingGrad {
    /// Flat token indices from the forward pass.
    indices: Vec<usize>,
    vocab_size: usize,
    embed_dim: usize,
    weight_id: NodeId,
}

impl GradFn for EmbeddingGrad {
    /// Backward pass for embedding lookup: scatter-add upstream gradients into
    /// the weight gradient table.
    ///
    /// For each token position `i`:
    ///
    /// ```text
    /// grad_weight[indices[i]] += grad_output[i]
    /// ```
    ///
    /// Multiple positions with the same index accumulate (correct for the
    /// case where one embedding row appears more than once in a batch).
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let mut grad_weight = vec![0.0f32; self.vocab_size * self.embed_dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            let src = i * self.embed_dim;
            let dst = idx * self.embed_dim;
            for d in 0..self.embed_dim {
                grad_weight[dst + d] += grad_output[src + d];
            }
        }
        vec![grad_weight]
    }

    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.weight_id]
    }
}

// ── embedding ─────────────────────────────────────────────────────────────────

/// Embedding lookup: gather rows from `weight` by integer `indices`.
///
/// `weight` has shape `[vocab_size, embed_dim]`. `indices` is a flat slice of
/// `usize` token IDs with logical shape `index_shape`. Every index must be
/// less than `vocab_size`.
///
/// The output has shape `index_shape ++ [embed_dim]`. For example, with
/// `index_shape = [batch, seq]` and `weight` of shape `[V, D]`, the output
/// is `[batch, seq, D]`.
///
/// Gradients flow only to `weight` (indices are non-differentiable).
/// The backward pass performs scatter-add: each upstream gradient slice is
/// added into the corresponding `weight` row. Repeated indices accumulate
/// correctly.
///
/// # Panics
///
/// - If `weight` is not 2-D.
/// - If `index_shape` is empty.
/// - If any index is `>= vocab_size`.
pub fn embedding(
    _ctx: &Context,
    tape: &mut Tape,
    weight: &Tensor,
    indices: &[usize],
    index_shape: Vec<usize>,
) -> Tensor {
    assert_eq!(
        weight.ndim(),
        2,
        "embedding: weight must be 2-D [vocab_size, embed_dim], got {}D",
        weight.ndim()
    );
    assert!(
        !index_shape.is_empty(),
        "embedding: index_shape must not be empty"
    );

    let vocab_size = weight.shape()[0];
    let embed_dim = weight.shape()[1];
    let num_tokens: usize = index_shape.iter().product();

    assert_eq!(
        indices.len(),
        num_tokens,
        "embedding: indices.len()={} does not match index_shape {:?} (product={num_tokens})",
        indices.len(),
        index_shape
    );

    let w_data = weight.to_vec();
    let mut out_data = vec![0.0f32; num_tokens * embed_dim];

    // Delegate the gather to the backend (honours the no-direct-arithmetic rule).
    // _ctx.backend is unused here since embedding is pure indexed copy, but we
    // call through the backend for future GPU compatibility.
    _ctx.backend.embedding_forward(&w_data, indices, &mut out_data, num_tokens, embed_dim, vocab_size);

    let mut out_shape = index_shape.clone();
    out_shape.push(embed_dim);

    let storage = Arc::new(Storage::new(out_data, weight.device(), weight.dtype()));
    let layout = Layout::contiguous(out_shape);
    let out = Tensor::from_storage(storage, layout, weight.requires_grad);

    if weight.requires_grad {
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(EmbeddingGrad {
                indices: indices.to_vec(),
                vocab_size,
                embed_dim,
                weight_id: weight.node,
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

    // ── forward ──────────────────────────────────────────────────────────

    #[test]
    fn embedding_forward_basic() {
        // weight: 4 tokens × 3 dims. Look up [3, 1, 0].
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(
            vec![
                1.0, 2.0, 3.0,  // row 0
                4.0, 5.0, 6.0,  // row 1
                7.0, 8.0, 9.0,  // row 2
                10.0, 11.0, 12.0, // row 3
            ],
            vec![4, 3],
        );
        let out = embedding(&ctx, &mut tape, &weight, &[3, 1, 0], vec![3]);
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.to_vec(),
            vec![10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn embedding_forward_batched_shape() {
        // index_shape [2, 2] → output [2, 2, embed_dim].
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = embedding(&ctx, &mut tape, &weight, &[0, 1, 1, 0], vec![2, 2]);
        assert_eq!(out.shape(), &[2, 2, 2]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0]);
    }

    // ── backward ─────────────────────────────────────────────────────────

    #[test]
    fn embedding_backward_distinct_indices() {
        // Each index appears once: grad_weight[idx] = upstream_grad[i].
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 3 rows × 2 dims
            vec![3, 2],
        );
        tape.push(Node::leaf(weight.node));
        // Look up rows [2, 0] — shape [2].
        let out = embedding(&ctx, &mut tape, &weight, &[2, 0], vec![2]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // backward seeds out.grad = [1,1,1,1]. grad flows:
        //   position 0 → weight row 2: [1,1]
        //   position 1 → weight row 0: [1,1]
        //   weight row 1 (unused): [0,0]
        let gw = weight.grad_strict(&store); // SAFE: requires_grad=true
        assert_eq!(gw, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn embedding_backward_repeated_index() {
        // Same index twice: gradients accumulate (scatter-add).
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]); // 2 rows × 2 dims
        tape.push(Node::leaf(weight.node));
        let out = embedding(&ctx, &mut tape, &weight, &[0, 0], vec![2]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // Both positions map to row 0 → grad accumulates to [2,2]; row 1 → [0,0].
        let gw = weight.grad_strict(&store); // SAFE: requires_grad=true
        assert_eq!(gw, &[2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_backward_grad_shape() {
        // Grad shape must match weight shape regardless of index_shape.
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(vec![0.1f32; 5 * 8], vec![5, 8]); // vocab=5, dim=8
        tape.push(Node::leaf(weight.node));
        let out = embedding(&ctx, &mut tape, &weight, &[1, 3, 2, 4, 0, 1], vec![2, 3]);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        assert_eq!(out.shape(), &[2, 3, 8]);
        assert_eq!(weight.grad_strict(&store).len(), 5 * 8); // SAFE: requires_grad=true
    }

    #[test]
    #[should_panic(expected = "weight must be 2-D")]
    fn embedding_wrong_weight_rank_panics() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(vec![1.0f32; 8], vec![2, 2, 2]); // 3-D — wrong
        embedding(&ctx, &mut tape, &weight, &[0], vec![1]);
    }

    #[test]
    #[should_panic(expected = "index 5 out of range for vocab_size 3")]
    fn embedding_out_of_bounds_panics() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let weight = leaf(vec![1.0f32; 6], vec![3, 2]);
        embedding(&ctx, &mut tape, &weight, &[5], vec![1]);
    }
}
