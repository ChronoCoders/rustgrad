/// Compute backend interface. All tensor ops route through this trait.
///
/// Implementations must be `Send + Sync` — the context is shared across
/// threads. No direct slice arithmetic is allowed in op functions; all
/// compute goes through `ctx.backend`.
pub trait Backend: Send + Sync {
    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Element-wise subtraction: `out[i] = a[i] - b[i]`.
    fn sub(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Element-wise multiplication: `out[i] = a[i] * b[i]`.
    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Element-wise division: `out[i] = a[i] / b[i]`.
    fn div(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Matrix multiplication: `out = A @ B`.
    ///
    /// `a` is `[m, k]`, `b` is `[k, n]`, `out` is `[m, n]`, all in
    /// row-major order.
    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);

    /// Reduction sum along `axis`.
    ///
    /// `a` has the logical `shape`; `out` has all dimensions of `a` except
    /// `axis`, reduced to size 1 (or removed, depending on caller convention).
    fn sum(&self, a: &[f32], out: &mut [f32], shape: &[usize], axis: usize);

    /// Element-wise natural exponential: `out[i] = exp(a[i])`.
    fn exp(&self, a: &[f32], out: &mut [f32]);

    /// Element-wise natural logarithm: `out[i] = ln(a[i])`.
    fn ln(&self, a: &[f32], out: &mut [f32]);

    /// Element-wise square root: `out[i] = sqrt(a[i])`.
    fn sqrt(&self, a: &[f32], out: &mut [f32]);

    /// Gather rows from a weight table by integer indices (embedding lookup).
    ///
    /// For each token position `i` in `0..num_tokens`:
    ///
    /// ```text
    /// out[i * embed_dim .. (i+1) * embed_dim]
    ///     = weight[indices[i] * embed_dim .. (indices[i]+1) * embed_dim]
    /// ```
    ///
    /// - `weight`  : `[vocab_size * embed_dim]`
    /// - `indices` : `[num_tokens]`  — each value must be `< vocab_size`
    /// - `out`     : `[num_tokens * embed_dim]`  (written)
    ///
    /// # Panics
    ///
    /// Panics if any index is `>= vocab_size` or if buffer lengths are wrong.
    fn embedding_forward(
        &self,
        weight: &[f32],
        indices: &[usize],
        out: &mut [f32],
        num_tokens: usize,
        embed_dim: usize,
        vocab_size: usize,
    );

    /// Fused scaled dot-product attention forward pass.
    ///
    /// Computes `softmax(Q @ K^T / sqrt(d_k)) @ V` for each `(batch, head)` slice
    /// independently. All buffers are row-major contiguous.
    ///
    /// - `q`            : `[batch_size * num_heads * seq_q * d_k]`
    /// - `k`            : `[batch_size * num_heads * seq_k * d_k]`
    /// - `v`            : `[batch_size * num_heads * seq_k * d_v]`
    /// - `out`          : `[batch_size * num_heads * seq_q * d_v]`   (written)
    /// - `attn_weights` : `[batch_size * num_heads * seq_q * seq_k]` (written, saved for backward)
    ///
    /// # Panics
    ///
    /// Panics if buffer lengths are inconsistent with the shape parameters.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        attn_weights: &mut [f32],
        batch_size: usize,
        num_heads: usize,
        seq_q: usize,
        seq_k: usize,
        d_k: usize,
        d_v: usize,
    );

    /// Affine layer normalization over the last `norm_size` elements of each
    /// of the `batch_size` outer slices.
    ///
    /// For each slice `b` of length `norm_size`:
    ///
    /// ```text
    /// mu      = mean(x[b])
    /// var     = mean((x[b] - mu)^2)
    /// rstd[b] = 1 / sqrt(var + eps)
    /// xn[b]   = (x[b] - mu) * rstd[b]
    /// out[b]  = xn[b] * weight + bias
    /// ```
    ///
    /// `x`, `out`, and `x_norm` each have length `batch_size * norm_size`.
    /// `weight` and `bias` each have length `norm_size`.
    /// `rstd` has length `batch_size` and is written for use in the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if buffer lengths are inconsistent.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [f32],
        x_norm: &mut [f32],
        rstd: &mut [f32],
        batch_size: usize,
        norm_size: usize,
        eps: f32,
    );

    /// Numerically stable softmax along `axis`.
    ///
    /// For each slice of `a` along `axis`, subtracts the slice maximum before
    /// exponentiating (`exp(x - max)`), then divides by the sum of
    /// exponentials. This prevents overflow for large logit values.
    ///
    /// `a` and `out` must have the same length equal to `shape.iter().product()`.
    /// `shape` is the logical row-major shape of `a` (already contiguous).
    ///
    /// # Panics
    ///
    /// Panics if `axis >= shape.len()`.
    fn softmax(&self, a: &[f32], out: &mut [f32], shape: &[usize], axis: usize);
}
