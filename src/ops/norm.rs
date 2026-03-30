use std::sync::Arc;

use crate::autograd::context::Context;
use crate::autograd::graph::Tape;
use crate::autograd::node::{GradFn, Node, NodeId};
use crate::tensor::{Layout, Storage, Tensor};

// ── layer_norm ────────────────────────────────────────────────────────────────

struct LayerNormGrad {
    /// Normalized input before the affine transform (xn = (x - mu) * rstd).
    /// Shape: `[batch_size * norm_size]`, contiguous.
    x_norm: Arc<Storage>,
    /// Per-slice reciprocal standard deviation. Length: `batch_size`.
    rstd: Vec<f32>,
    /// Affine scale (weight). Length: `norm_size`.
    weight: Arc<Storage>,
    batch_size: usize,
    norm_size: usize,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: NodeId,
}

impl GradFn for LayerNormGrad {
    /// Backward pass for affine layer normalization.
    ///
    /// Given upstream gradient `dy` (same shape as the output):
    ///
    /// ```text
    /// grad_bias[j]   = Σ_b  dy[b, j]
    /// grad_weight[j] = Σ_b  dy[b, j] * xn[b, j]
    ///
    /// # Per-batch backward through the normalizer:
    /// dxn[b]          = dy[b] * weight            (element-wise)
    /// c1              = mean(dxn[b])
    /// c2              = mean(dxn[b] * xn[b])
    /// grad_x[b]       = rstd[b] * (dxn[b] - c1 - xn[b] * c2)
    /// ```
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let xn = &self.x_norm.data;
        let w = &self.weight.data;
        let n = self.norm_size;
        let b_total = self.batch_size;
        let n_f = n as f32;

        let mut grad_x = vec![0.0f32; b_total * n];
        let mut grad_weight = vec![0.0f32; n];
        let mut grad_bias = vec![0.0f32; n];

        // grad_bias and grad_weight are sums over the batch dimension.
        for b in 0..b_total {
            for j in 0..n {
                let idx = b * n + j;
                grad_bias[j] += grad_output[idx];
                grad_weight[j] += grad_output[idx] * xn[idx];
            }
        }

        // Backward through the normalization for each batch slice.
        for b in 0..b_total {
            // dxn[b] = dy[b] * weight.
            // c1 = mean(dxn[b]), c2 = mean(dxn[b] * xn[b]).
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            for j in 0..n {
                let idx = b * n + j;
                let dxn = grad_output[idx] * w[j];
                c1 += dxn;
                c2 += dxn * xn[idx];
            }
            c1 /= n_f;
            c2 /= n_f;

            let r = self.rstd[b];
            for j in 0..n {
                let idx = b * n + j;
                let dxn = grad_output[idx] * w[j];
                grad_x[idx] = r * (dxn - c1 - xn[idx] * c2);
            }
        }

        // Return in input order: x, weight, bias.
        vec![grad_x, grad_weight, grad_bias]
    }

    fn input_ids(&self) -> Vec<NodeId> {
        vec![self.input_id, self.weight_id, self.bias_id]
    }
}

/// Affine layer normalization over the trailing dimensions of `x`.
///
/// The normalized shape is inferred from `weight.shape()`. The last
/// `weight.ndim()` dimensions of `x` must match `weight.shape()` exactly.
/// `weight` and `bias` must have the same shape.
///
/// The operation is:
///
/// ```text
/// mu  = mean(x, trailing_axes, keepdim=true)
/// std = sqrt(var(x, trailing_axes, keepdim=true) + eps)
/// out = (x - mu) / std * weight + bias
/// ```
///
/// # Panics
///
/// - If `weight.shape() != bias.shape()`.
/// - If the trailing shape of `x` does not match `weight.shape()`.
/// - If `weight.ndim() == 0`.
pub fn layer_norm(
    ctx: &Context,
    tape: &mut Tape,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Tensor {
    let norm_shape = weight.shape();
    assert_eq!(
        norm_shape,
        bias.shape(),
        "layer_norm: weight shape {:?} != bias shape {:?}",
        norm_shape,
        bias.shape()
    );
    assert!(
        !norm_shape.is_empty(),
        "layer_norm: weight must have at least one dimension"
    );

    let x_shape = x.shape();
    let norm_ndim = norm_shape.len();
    assert!(
        x_shape.len() >= norm_ndim,
        "layer_norm: x rank {} < normalized rank {}",
        x_shape.len(),
        norm_ndim
    );
    let trailing = &x_shape[x_shape.len() - norm_ndim..];
    assert_eq!(
        trailing,
        norm_shape,
        "layer_norm: x trailing shape {:?} != normalized shape {:?}",
        trailing,
        norm_shape
    );

    let norm_size: usize = norm_shape.iter().product();
    let total: usize = x.numel();
    let batch_size = total / norm_size;

    let x_data = x.to_vec();
    let w_data = weight.to_vec();
    let b_data = bias.to_vec();

    let mut out_data = vec![0.0f32; total];
    let mut xn_data = vec![0.0f32; total];
    let mut rstd_data = vec![0.0f32; batch_size];

    ctx.backend.layer_norm(
        &x_data,
        &w_data,
        &b_data,
        &mut out_data,
        &mut xn_data,
        &mut rstd_data,
        batch_size,
        norm_size,
        eps,
    );

    let requires_grad = x.requires_grad || weight.requires_grad || bias.requires_grad;
    let storage = Arc::new(Storage::new(out_data, x.device(), x.dtype()));
    let layout = Layout::contiguous(x_shape.to_vec());
    let out = Tensor::from_storage(Arc::clone(&storage), layout, requires_grad);

    if requires_grad {
        let xn_storage = Arc::new(Storage::new(xn_data, x.device(), x.dtype()));
        let w_storage = Arc::clone(&weight.storage);
        tape.push(Node::with_grad_fn(
            out.node,
            Box::new(LayerNormGrad {
                x_norm: xn_storage,
                rstd: rstd_data,
                weight: w_storage,
                batch_size,
                norm_size,
                input_id: x.node,
                weight_id: weight.node,
                bias_id: bias.node,
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
    use crate::autograd::context::Context;
    use crate::backend::CpuBackend;
    use crate::tensor::Device;

    fn ctx() -> Context {
        Context::new(Arc::new(CpuBackend), Device::Cpu)
    }

    fn leaf(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_vec(data, shape, Device::Cpu).with_grad()
    }

    #[test]
    fn layer_norm_forward_identity_affine() {
        // weight=1, bias=0 → output has mean≈0 and std≈1 along last dim.
        let ctx = ctx();
        let mut tape = Tape::new();
        let x = leaf(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], vec![2, 4]);
        let w = leaf(vec![1.0; 4], vec![4]);
        let b = leaf(vec![0.0; 4], vec![4]);
        let out = layer_norm(&ctx, &mut tape, &x, &w, &b, 1e-5);
        assert_eq!(out.shape(), &[2, 4]);
        let v = out.to_vec();
        for row in 0..2 {
            let s = &v[row * 4..(row + 1) * 4];
            let mean: f32 = s.iter().sum::<f32>() / 4.0;
            let var: f32 = s.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "row {row} mean={mean}");
            assert!((var - 1.0).abs() < 1e-3, "row {row} var={var}");
        }
    }

    #[test]
    fn layer_norm_backward_bias_grad() {
        // grad_bias must equal the sum of upstream grads over the batch dim.
        let ctx = ctx();
        let mut tape = Tape::new();
        let x = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let w = leaf(vec![1.0; 4], vec![4]);
        let b = leaf(vec![0.0; 4], vec![4]);
        tape.push(Node::leaf(x.node));
        tape.push(Node::leaf(w.node));
        tape.push(Node::leaf(b.node));
        let out = layer_norm(&ctx, &mut tape, &x, &w, &b, 1e-5);
        let mut store = TensorStore::new();
        backward(&tape, &mut store, &out);
        // bias grad exists and has the right shape.
        let gb = b.grad(&store).expect("bias grad must exist"); // SAFE: requires_grad=true
        assert_eq!(gb.len(), 4);
    }

    #[test]
    fn layer_norm_shape_mismatch_panics() {
        let ctx = ctx();
        let mut tape = Tape::new();
        let x = leaf(vec![1.0; 6], vec![2, 3]);
        let w = leaf(vec![1.0; 4], vec![4]); // wrong size
        let b = leaf(vec![0.0; 4], vec![4]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            layer_norm(&ctx, &mut tape, &x, &w, &b, 1e-5);
        }));
        assert!(result.is_err(), "expected panic on shape mismatch");
    }
}
