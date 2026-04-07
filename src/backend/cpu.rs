use faer::Mat;

use crate::backend::traits::Backend;

/// CPU backend. All linear algebra is dispatched through `faer`,
/// which auto-selects AVX2 / AVX-512 SIMD at runtime.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        for ((x, y), z) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *z = x + y;
        }
    }

    fn sub(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        for ((x, y), z) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *z = x - y;
        }
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        for ((x, y), z) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *z = x * y;
        }
    }

    fn div(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        for ((x, y), z) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *z = x / y;
        }
    }

    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(out.len(), m * n);

        // Build faer matrices from row-major slices.
        let a_mat = Mat::from_fn(m, k, |i, j| a[i * k + j]);
        let b_mat = Mat::from_fn(k, n, |i, j| b[i * n + j]);
        let c_mat = &a_mat * &b_mat;

        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c_mat[(i, j)];
            }
        }
    }

    fn sum(&self, a: &[f32], out: &mut [f32], shape: &[usize], axis: usize) {
        assert!(
            axis < shape.len(),
            "sum: axis {} out of range for rank {}",
            axis,
            shape.len()
        );

        let ndim = shape.len();

        // Row-major strides for the input shape.
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1];
        }

        // Output shape: all dims except `axis`.
        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(d, _)| d != axis)
            .map(|(_, &s)| s)
            .collect();

        let out_numel: usize = if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product()
        };
        assert_eq!(
            out.len(),
            out_numel,
            "sum: output buffer has length {} but expected {}",
            out.len(),
            out_numel
        );

        // Row-major strides for the output shape.
        let out_strides: Vec<usize> = {
            let n = out_shape.len();
            let mut st = vec![1usize; n];
            for i in (0..n.saturating_sub(1)).rev() {
                st[i] = st[i + 1] * out_shape[i + 1];
            }
            st
        };

        out.iter_mut().for_each(|x| *x = 0.0);

        let numel: usize = shape.iter().product();
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let src: usize = indices
                .iter()
                .zip(in_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();

            let out_flat: usize = indices
                .iter()
                .enumerate()
                .filter(|&(d, _)| d != axis)
                .map(|(d, &i)| {
                    let out_d = if d < axis { d } else { d - 1 };
                    i * out_strides[out_d]
                })
                .sum();

            out[out_flat] += a[src];

            // Increment indices in row-major order.
            let mut carry = true;
            for d in (0..ndim).rev() {
                if carry {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
    }

    fn exp(&self, a: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), out.len());
        for (x, z) in a.iter().zip(out.iter_mut()) {
            *z = x.exp();
        }
    }

    fn ln(&self, a: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), out.len());
        for (x, z) in a.iter().zip(out.iter_mut()) {
            *z = x.ln();
        }
    }

    fn sqrt(&self, a: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), out.len());
        for (x, z) in a.iter().zip(out.iter_mut()) {
            *z = x.sqrt();
        }
    }

    fn embedding_forward(
        &self,
        weight: &[f32],
        indices: &[usize],
        out: &mut [f32],
        num_tokens: usize,
        embed_dim: usize,
        vocab_size: usize,
    ) {
        debug_assert_eq!(weight.len(), vocab_size * embed_dim);
        debug_assert_eq!(indices.len(), num_tokens);
        debug_assert_eq!(out.len(), num_tokens * embed_dim);

        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < vocab_size,
                "embedding_forward: index {idx} out of range for vocab_size {vocab_size}"
            );
            let src = idx * embed_dim;
            let dst = i * embed_dim;
            out[dst..dst + embed_dim].copy_from_slice(&weight[src..src + embed_dim]);
        }
    }

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
    ) {
        debug_assert_eq!(q.len(), batch_size * num_heads * seq_q * d_k);
        debug_assert_eq!(k.len(), batch_size * num_heads * seq_k * d_k);
        debug_assert_eq!(v.len(), batch_size * num_heads * seq_k * d_v);
        debug_assert_eq!(out.len(), batch_size * num_heads * seq_q * d_v);
        debug_assert_eq!(attn_weights.len(), batch_size * num_heads * seq_q * seq_k);

        let scale = 1.0f32 / (d_k as f32).sqrt();
        // Scratch buffers reused across heads.
        let mut scores = vec![0.0f32; seq_q * seq_k];
        let mut k_t = vec![0.0f32; d_k * seq_k];

        for b in 0..batch_size {
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let q_off = bh * seq_q * d_k;
                let k_off = bh * seq_k * d_k;
                let v_off = bh * seq_k * d_v;
                let out_off = bh * seq_q * d_v;
                let aw_off = bh * seq_q * seq_k;

                // Transpose K: [seq_k, d_k] → [d_k, seq_k].
                for s in 0..seq_k {
                    for d in 0..d_k {
                        k_t[d * seq_k + s] = k[k_off + s * d_k + d];
                    }
                }

                // scores = Q @ K^T : [seq_q, d_k] @ [d_k, seq_k] → [seq_q, seq_k].
                self.matmul(
                    &q[q_off..q_off + seq_q * d_k],
                    &k_t,
                    &mut scores,
                    seq_q,
                    d_k,
                    seq_k,
                );

                // Scale.
                for x in scores.iter_mut() {
                    *x *= scale;
                }

                // Softmax over last axis (rows of [seq_q, seq_k]).
                self.softmax(
                    &scores,
                    &mut attn_weights[aw_off..aw_off + seq_q * seq_k],
                    &[seq_q, seq_k],
                    1,
                );

                // out = attn_weights @ V : [seq_q, seq_k] @ [seq_k, d_v] → [seq_q, d_v].
                self.matmul(
                    &attn_weights[aw_off..aw_off + seq_q * seq_k],
                    &v[v_off..v_off + seq_k * d_v],
                    &mut out[out_off..out_off + seq_q * d_v],
                    seq_q,
                    seq_k,
                    d_v,
                );
            }
        }
    }

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
    ) {
        debug_assert_eq!(x.len(), batch_size * norm_size);
        debug_assert_eq!(out.len(), batch_size * norm_size);
        debug_assert_eq!(x_norm.len(), batch_size * norm_size);
        debug_assert_eq!(rstd.len(), batch_size);
        debug_assert_eq!(weight.len(), norm_size);
        debug_assert_eq!(bias.len(), norm_size);

        let n = norm_size as f32;
        for b in 0..batch_size {
            let slice = &x[b * norm_size..(b + 1) * norm_size];

            // Mean.
            let mu: f32 = slice.iter().sum::<f32>() / n;

            // Variance.
            let var: f32 = slice.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / n;

            let r = 1.0 / (var + eps).sqrt();
            rstd[b] = r;

            let out_slice = &mut out[b * norm_size..(b + 1) * norm_size];
            let xn_slice = &mut x_norm[b * norm_size..(b + 1) * norm_size];
            for i in 0..norm_size {
                let xn = (slice[i] - mu) * r;
                xn_slice[i] = xn;
                out_slice[i] = xn * weight[i] + bias[i];
            }
        }
    }

    fn softmax(&self, a: &[f32], out: &mut [f32], shape: &[usize], axis: usize) {
        assert!(
            axis < shape.len(),
            "softmax: axis {} out of range for rank {}",
            axis,
            shape.len()
        );
        debug_assert_eq!(a.len(), out.len());

        let axis_size = shape[axis];
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = outer * axis_size * inner_size;

                // Max for numerical stability.
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..axis_size {
                    let v = a[base + k * inner_size + inner];
                    if v > max_val {
                        max_val = v;
                    }
                }

                // exp(x - max) and running sum.
                let mut sum = 0.0f32;
                for k in 0..axis_size {
                    let e = (a[base + k * inner_size + inner] - max_val).exp();
                    out[base + k * inner_size + inner] = e;
                    sum += e;
                }

                // Normalize.
                for k in 0..axis_size {
                    out[base + k * inner_size + inner] /= sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn add_elementwise() {
        let b = backend();
        let a = [1.0f32, 2.0, 3.0];
        let x = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        b.add(&a, &x, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_elementwise() {
        let b = backend();
        let mut out = [0.0f32; 3];
        b.sub(&[3.0, 2.0, 1.0], &[1.0, 1.0, 1.0], &mut out);
        assert_eq!(out, [2.0, 1.0, 0.0]);
    }

    #[test]
    fn mul_elementwise() {
        let b = backend();
        let mut out = [0.0f32; 3];
        b.mul(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &mut out);
        assert_eq!(out, [2.0, 6.0, 12.0]);
    }

    #[test]
    fn div_elementwise() {
        let b = backend();
        let mut out = [0.0f32; 3];
        b.div(&[6.0, 4.0, 2.0], &[2.0, 2.0, 2.0], &mut out);
        assert_eq!(out, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn matmul_2x2() {
        let b = backend();
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let x = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        b.matmul(&a, &x, &mut out, 2, 2, 2);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn exp_ln_roundtrip() {
        let b = backend();
        let a = [1.0f32, 2.0, 3.0];
        let mut exp_out = [0.0f32; 3];
        let mut ln_out = [0.0f32; 3];
        b.exp(&a, &mut exp_out);
        b.ln(&exp_out, &mut ln_out);
        for (orig, result) in a.iter().zip(ln_out.iter()) {
            assert!((orig - result).abs() < 1e-5, "{} vs {}", orig, result);
        }
    }

    #[test]
    fn sqrt_values() {
        let b = backend();
        let a = [1.0f32, 4.0, 9.0];
        let mut out = [0.0f32; 3];
        b.sqrt(&a, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_along_axis_0() {
        // Shape [2,3], axis 0 → out shape [3]
        // [[1,2,3],[4,5,6]] → [5,7,9]
        let b = backend();
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        b.sum(&a, &mut out, &[2, 3], 0);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_along_axis_1() {
        // Shape [2,3], axis 1 → out shape [2]
        // [[1,2,3],[4,5,6]] → [6,15]
        let b = backend();
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];
        b.sum(&a, &mut out, &[2, 3], 1);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn embedding_forward_basic() {
        // weight: 3 tokens × 2 dims. Look up [2, 0, 1].
        let b = backend();
        let weight = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0]; // rows: [10,11],[20,21],[30,31]
        let indices = [2usize, 0, 1];
        let mut out = [0.0f32; 6];
        b.embedding_forward(&weight, &indices, &mut out, 3, 2, 3);
        assert_eq!(out, [30.0, 31.0, 10.0, 11.0, 20.0, 21.0]);
    }

    #[test]
    fn embedding_forward_repeated_index() {
        // Same index twice: both output rows should be identical copies.
        let b = backend();
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows × 2 dims
        let indices = [1usize, 1];
        let mut out = [0.0f32; 4];
        b.embedding_forward(&weight, &indices, &mut out, 2, 2, 3);
        assert_eq!(out, [3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn sdpa_identity_values() {
        // batch=1, heads=1, seq_q=2, seq_k=2, d_k=2, d_v=2.
        // Q = K = I (identity 2×2), V = I. Attention weights should be ~[0.73, 0.27; 0.27, 0.73].
        // Output = attn_weights @ V = attn_weights (since V=I).
        let b = backend();
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        let mut aw = [0.0f32; 4];
        b.sdpa_forward(&q, &k, &v, &mut out, &mut aw, 1, 1, 2, 2, 2, 2);
        // Each attention row sums to 1.
        assert!(
            (aw[0] + aw[1] - 1.0).abs() < 1e-5,
            "row0 sum = {}",
            aw[0] + aw[1]
        );
        assert!(
            (aw[2] + aw[3] - 1.0).abs() < 1e-5,
            "row1 sum = {}",
            aw[2] + aw[3]
        );
        // Diagonal dominates (matching query attends to itself).
        assert!(aw[0] > aw[1], "aw[0]={} should > aw[1]={}", aw[0], aw[1]);
    }

    #[test]
    fn sdpa_output_shape_batched() {
        // batch=2, heads=3, seq_q=4, seq_k=5, d_k=8, d_v=6.
        let b = backend();
        let batch = 2;
        let heads = 3;
        let sq = 4;
        let sk = 5;
        let dk = 8;
        let dv = 6;
        let q = vec![0.1f32; batch * heads * sq * dk];
        let k = vec![0.1f32; batch * heads * sk * dk];
        let v = vec![0.1f32; batch * heads * sk * dv];
        let mut out = vec![0.0f32; batch * heads * sq * dv];
        let mut aw = vec![0.0f32; batch * heads * sq * sk];
        b.sdpa_forward(&q, &k, &v, &mut out, &mut aw, batch, heads, sq, sk, dk, dv);
        // Output contains no NaN/Inf.
        assert!(
            out.iter().all(|x| x.is_finite()),
            "output has non-finite values"
        );
        // All attention weight rows sum to 1.
        for row_start in (0..aw.len()).step_by(sk) {
            let s: f32 = aw[row_start..row_start + sk].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum={s}");
        }
    }

    #[test]
    fn layer_norm_unit_mean_var() {
        // Single batch, 4 elements: after layer norm, mean≈0 and std≈1 (before affine).
        let b = backend();
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let mut xn = [0.0f32; 4];
        let mut rstd = [0.0f32; 1];
        b.layer_norm(&x, &weight, &bias, &mut out, &mut xn, &mut rstd, 1, 4, 1e-5);
        // With identity affine, out == xn.
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
        assert!((var - 1.0).abs() < 1e-4, "var={var}");
    }

    #[test]
    fn layer_norm_affine() {
        // weight=2, bias=1 → out = 2*xn + 1.
        let b = backend();
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [2.0f32; 4];
        let bias = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        let mut xn = [0.0f32; 4];
        let mut rstd = [0.0f32; 1];
        b.layer_norm(&x, &weight, &bias, &mut out, &mut xn, &mut rstd, 1, 4, 1e-5);
        for i in 0..4 {
            let expected = 2.0 * xn[i] + 1.0;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "out[{i}]={} expected={expected}",
                out[i]
            );
        }
    }

    #[test]
    fn layer_norm_two_batches() {
        // Two independent rows, each should normalize independently.
        let b = backend();
        let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let weight = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 8];
        let mut xn = [0.0f32; 8];
        let mut rstd = [0.0f32; 2];
        b.layer_norm(&x, &weight, &bias, &mut out, &mut xn, &mut rstd, 2, 4, 1e-5);
        // Both rows should have mean≈0 after normalization.
        for row in 0..2 {
            let slice = &out[row * 4..(row + 1) * 4];
            let mean: f32 = slice.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "row {row} mean={mean}");
        }
        // Both normalized outputs should be identical (same relative spread).
        for i in 0..4 {
            assert!(
                (out[i] - out[4 + i]).abs() < 1e-4,
                "row0[{i}]={} row1[{i}]={}",
                out[i],
                out[4 + i]
            );
        }
    }

    #[test]
    fn softmax_last_axis() {
        // Shape [2,3], axis 1: each row sums to 1.
        let b = backend();
        let a = [1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 6];
        b.softmax(&a, &mut out, &[2, 3], 1);
        // Each row must sum to 1.
        let row0: f32 = out[..3].iter().sum();
        let row1: f32 = out[3..].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-6, "row0 sum={row0}");
        assert!((row1 - 1.0).abs() < 1e-6, "row1 sum={row1}");
        // Uniform row: all 1/3.
        for &v in &out[3..] {
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-6,
                "uniform row: expected 1/3, got {v}"
            );
        }
    }

    #[test]
    fn softmax_first_axis() {
        // Shape [2,3], axis 0: each column sums to 1.
        let b = backend();
        let a = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
        let mut out = [0.0f32; 6];
        b.softmax(&a, &mut out, &[2, 3], 0);
        // Same values along axis 0 → each column is [0.5, 0.5].
        for col in 0..3 {
            let sum = out[col] + out[3 + col];
            assert!((sum - 1.0).abs() < 1e-6, "col{col} sum={sum}");
            assert!(
                (out[col] - 0.5).abs() < 1e-6,
                "expected 0.5, got {}",
                out[col]
            );
        }
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large logits: without max subtraction this overflows.
        let b = backend();
        let a = [1000.0f32, 1001.0, 1002.0];
        let mut out = [0.0f32; 3];
        b.softmax(&a, &mut out, &[3], 0);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        // Largest logit dominates.
        assert!(out[2] > out[1] && out[1] > out[0]);
    }
}
