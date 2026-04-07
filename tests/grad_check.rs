use std::sync::Arc;

use rustgrad::autograd::{backward, Context, Node, Tape, TensorStore};
use rustgrad::backend::CpuBackend;
use rustgrad::ops::binary::{add, div, mul, sub};
use rustgrad::ops::matmul::matmul;
use rustgrad::ops::reduction::{mean, sum};
use rustgrad::ops::shape::{permute, reshape, squeeze, transpose, unsqueeze};
use rustgrad::ops::attention::scaled_dot_product_attention;
use rustgrad::ops::embedding::embedding;
use rustgrad::ops::norm::layer_norm;
use rustgrad::ops::unary::softmax;
use rustgrad::tensor::{Device, Tensor};

// ── Grad check helpers ────────────────────────────────────────────────────────

const TOL: f32 = 1e-3;

fn cpu_ctx() -> Context {
    Context::new(Arc::new(CpuBackend), Device::Cpu)
}

/// Compute the numerical gradient of `sum(f(x))` with respect to `x`
/// using central finite differences.
///
/// Precision notes:
/// - Perturbations are computed in f64 and rounded to f32 so the effective
///   step matches what the f32 op actually receives.
/// - Output sums are accumulated in f64 to avoid catastrophic cancellation
///   when individual output values are large.
/// - `effective_eps` is the actual f64 difference between xp[i] and xm[i]
///   after the f32 roundtrip, making the division exact.
fn numerical_grad<F>(x_data: &[f32], f: F) -> Vec<f32>
where
    F: Fn(Vec<f32>) -> Vec<f32>,
{
    // eps=1e-2: large enough that f32 rounding (ulp ≈ 5e-7 at magnitude 8)
    // is ~1e-4 relative to 2*eps — well within TOL=1e-3.
    // All our ops are linear in each argument, so truncation error = 0.
    // For div grad_b (nonlinear in b), denominator values are kept ≥ 1.5
    // so the third-derivative truncation term stays below 5e-5.
    const EPS: f64 = 1e-2;
    let n = x_data.len();
    let mut grad = vec![0.0f32; n];
    for i in 0..n {
        let mut xp = x_data.to_vec();
        let mut xm = x_data.to_vec();
        xp[i] = (x_data[i] as f64 + EPS) as f32;
        xm[i] = (x_data[i] as f64 - EPS) as f32;
        // Effective step: actual f64 difference after f32 roundtrip.
        let effective_eps: f64 = xp[i] as f64 - xm[i] as f64;
        // Accumulate outputs in f64 to avoid losing the tiny perturbation.
        let fp: f64 = f(xp).iter().map(|&v| v as f64).sum();
        let fm: f64 = f(xm).iter().map(|&v| v as f64).sum();
        grad[i] = if effective_eps.abs() < 1e-30 {
            0.0
        } else {
            ((fp - fm) / effective_eps) as f32
        };
    }
    grad
}

/// Check the gradient of a single-input op. Returns the max absolute difference.
/// Panics if any element exceeds `tol`.
fn check_grad_1<F>(
    x_data: Vec<f32>,
    x_shape: Vec<usize>,
    f_analytical: F,
    f_numerical: impl Fn(Vec<f32>) -> Vec<f32>,
    op_name: &str,
) -> f32
where
    F: Fn(&Context, &mut Tape, &Tensor) -> Tensor,
{
    let ctx = cpu_ctx();

    // Analytical.
    let mut tape = Tape::new();
    let x = Tensor::from_vec(x_data.clone(), x_shape.clone(), Device::Cpu).with_grad();
    tape.push(Node::leaf(x.node));
    let out = f_analytical(&ctx, &mut tape, &x);
    let mut store = TensorStore::new();
    backward(&tape, &mut store, &out);
    let analytic = x.grad_strict(&store).to_vec();

    // Numerical.
    let numeric = numerical_grad(&x_data, f_numerical);

    let max_diff = analytic
        .iter()
        .zip(numeric.iter())
        .map(|(a, n)| (a - n).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < TOL,
        "grad_check FAIL: {op_name} — max_diff={max_diff:.6} (tol={TOL})\n  analytic={analytic:?}\n  numeric={numeric:?}"
    );
    max_diff
}

/// Check the gradient for both inputs of a binary op.
/// Returns (max_diff_a, max_diff_b).
#[allow(clippy::too_many_arguments)]
fn check_grad_2<F>(
    a_data: Vec<f32>,
    a_shape: Vec<usize>,
    b_data: Vec<f32>,
    b_shape: Vec<usize>,
    f_analytical: F,
    f_num_a: impl Fn(Vec<f32>) -> Vec<f32>,
    f_num_b: impl Fn(Vec<f32>) -> Vec<f32>,
    op_name: &str,
) -> (f32, f32)
where
    F: Fn(&Context, &mut Tape, &Tensor, &Tensor) -> Tensor,
{
    let ctx = cpu_ctx();

    // Analytical — both grads in one backward pass.
    let mut tape = Tape::new();
    let a = Tensor::from_vec(a_data.clone(), a_shape.clone(), Device::Cpu).with_grad();
    let b = Tensor::from_vec(b_data.clone(), b_shape.clone(), Device::Cpu).with_grad();
    tape.push(Node::leaf(a.node));
    tape.push(Node::leaf(b.node));
    let out = f_analytical(&ctx, &mut tape, &a, &b);
    let mut store = TensorStore::new();
    backward(&tape, &mut store, &out);
    let analytic_a = a.grad_strict(&store).to_vec();
    let analytic_b = b.grad_strict(&store).to_vec();

    // Numerical — perturb a.
    let num_a = numerical_grad(&a_data, f_num_a);
    // Numerical — perturb b.
    let num_b = numerical_grad(&b_data, f_num_b);

    let max_a = analytic_a
        .iter()
        .zip(num_a.iter())
        .map(|(a, n)| (a - n).abs())
        .fold(0.0f32, f32::max);

    let max_b = analytic_b
        .iter()
        .zip(num_b.iter())
        .map(|(a, n)| (a - n).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_a < TOL,
        "grad_check FAIL: {op_name} grad_a — max_diff={max_a:.6} (tol={TOL})\n  analytic={analytic_a:?}\n  numeric={num_a:?}"
    );
    assert!(
        max_b < TOL,
        "grad_check FAIL: {op_name} grad_b — max_diff={max_b:.6} (tol={TOL})\n  analytic={analytic_b:?}\n  numeric={num_b:?}"
    );
    (max_a, max_b)
}

// ── Elementwise ops ───────────────────────────────────────────────────────────

#[test]
fn test_grad_check_add() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];
    let (da, db) = check_grad_2(
        a.clone(),
        vec![4],
        b.clone(),
        vec![4],
        add,
        {
            let b = b.clone();
            move |a_data| a_data.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        },
        {
            let a = a.clone();
            move |b_data| a.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect()
        },
        "add",
    );
    println!("PASS: add grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

#[test]
fn test_grad_check_sub() {
    let a = vec![3.0f32, 2.0, 5.0, 1.0];
    let b = vec![1.0f32, 0.5, 2.0, 0.1];
    let (da, db) = check_grad_2(
        a.clone(),
        vec![4],
        b.clone(),
        vec![4],
        sub,
        {
            let b = b.clone();
            move |a_data| a_data.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
        },
        {
            let a = a.clone();
            move |b_data| a.iter().zip(b_data.iter()).map(|(x, y)| x - y).collect()
        },
        "sub",
    );
    println!("PASS: sub grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

#[test]
fn test_grad_check_mul() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![2.0f32, 0.5, 3.0, 1.5];
    let (da, db) = check_grad_2(
        a.clone(),
        vec![4],
        b.clone(),
        vec![4],
        mul,
        {
            let b = b.clone();
            move |a_data| a_data.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
        },
        {
            let a = a.clone();
            move |b_data| a.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect()
        },
        "mul",
    );
    println!("PASS: mul grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

#[test]
fn test_grad_check_div() {
    // Keep denominator well away from zero.
    let a = vec![2.0f32, 4.0, 6.0, 8.0];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];
    let (da, db) = check_grad_2(
        a.clone(),
        vec![4],
        b.clone(),
        vec![4],
        div,
        {
            let b = b.clone();
            move |a_data| a_data.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
        },
        {
            let a = a.clone();
            move |b_data| a.iter().zip(b_data.iter()).map(|(x, y)| x / y).collect()
        },
        "div",
    );
    println!("PASS: div grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

#[test]
fn test_grad_check_add_broadcast() {
    // a: [2,3], b: [3] — b is broadcast over rows
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0f32, 2.0, 3.0];
    let (da, db) = check_grad_2(
        a_data.clone(),
        vec![2, 3],
        b_data.clone(),
        vec![3],
        add,
        {
            let b = b_data.clone();
            move |a_data| {
                // broadcast b over rows
                a_data
                    .chunks(3)
                    .flat_map(|row| row.iter().zip(b.iter()).map(|(x, y)| x + y))
                    .collect()
            }
        },
        {
            let a = a_data.clone();
            move |b_data| {
                a.chunks(3)
                    .flat_map(|row| row.iter().zip(b_data.iter()).map(|(x, y)| x + y))
                    .collect()
            }
        },
        "add_broadcast",
    );
    println!("PASS: add_broadcast grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

// ── Reduction ops ─────────────────────────────────────────────────────────────

#[test]
fn test_grad_check_sum() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_shape = vec![2, 3];
    for axis in 0..2 {
        let max_diff = check_grad_1(
            x_data.clone(),
            x_shape.clone(),
            move |ctx, tape, x| sum(ctx, tape, x, axis),
            {
                let shape = x_shape.clone();
                move |x_data| {
                    let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                    let ctx = cpu_ctx();
                    let mut tape = Tape::new();
                    sum(&ctx, &mut tape, &t, axis).to_vec()
                }
            },
            &format!("sum_axis{axis}"),
        );
        println!("PASS: sum axis={axis} grad_check — max_diff={max_diff:.6}");
    }
}

#[test]
fn test_grad_check_mean() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_shape = vec![2, 3];
    for axis in 0..2 {
        let max_diff = check_grad_1(
            x_data.clone(),
            x_shape.clone(),
            move |ctx, tape, x| mean(ctx, tape, x, axis),
            {
                let shape = x_shape.clone();
                move |x_data| {
                    let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                    let ctx = cpu_ctx();
                    let mut tape = Tape::new();
                    mean(&ctx, &mut tape, &t, axis).to_vec()
                }
            },
            &format!("mean_axis{axis}"),
        );
        println!("PASS: mean axis={axis} grad_check — max_diff={max_diff:.6}");
    }
}

// ── Shape ops ─────────────────────────────────────────────────────────────────

#[test]
fn test_grad_check_reshape() {
    let x_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let max_diff = check_grad_1(
        x_data,
        vec![2, 3],
        |ctx, tape, x| reshape(ctx, tape, x, vec![3, 2]),
        |x_data| {
            // reshape is a no-op on values — just reinterpret
            x_data
        },
        "reshape",
    );
    println!("PASS: reshape grad_check — max_diff={max_diff:.6}");
}

#[test]
fn test_grad_check_permute() {
    let x_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let x_shape = vec![2, 3, 4];
    let max_diff = check_grad_1(
        x_data,
        x_shape.clone(),
        |ctx, tape, x| permute(ctx, tape, x, vec![2, 0, 1]),
        {
            let shape = x_shape.clone();
            move |x_data| {
                let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                permute(&ctx, &mut tape, &t, vec![2, 0, 1]).to_vec()
            }
        },
        "permute",
    );
    println!("PASS: permute grad_check — max_diff={max_diff:.6}");
}

#[test]
fn test_grad_check_transpose() {
    let x_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let x_shape = vec![2, 3];
    let max_diff = check_grad_1(
        x_data,
        x_shape.clone(),
        |ctx, tape, x| transpose(ctx, tape, x, 0, 1),
        {
            let shape = x_shape.clone();
            move |x_data| {
                let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                transpose(&ctx, &mut tape, &t, 0, 1).to_vec()
            }
        },
        "transpose",
    );
    println!("PASS: transpose grad_check — max_diff={max_diff:.6}");
}

#[test]
fn test_grad_check_squeeze() {
    let x_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let x_shape = vec![1, 6];
    let max_diff = check_grad_1(
        x_data,
        x_shape.clone(),
        |ctx, tape, x| squeeze(ctx, tape, x, Some(0)),
        {
            let shape = x_shape.clone();
            move |x_data| {
                let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                squeeze(&ctx, &mut tape, &t, Some(0)).to_vec()
            }
        },
        "squeeze",
    );
    println!("PASS: squeeze grad_check — max_diff={max_diff:.6}");
}

#[test]
fn test_grad_check_unsqueeze() {
    let x_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let x_shape = vec![6];
    let max_diff = check_grad_1(
        x_data,
        x_shape.clone(),
        |ctx, tape, x| unsqueeze(ctx, tape, x, 0),
        {
            let shape = x_shape.clone();
            move |x_data| {
                let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                unsqueeze(&ctx, &mut tape, &t, 0).to_vec()
            }
        },
        "unsqueeze",
    );
    println!("PASS: unsqueeze grad_check — max_diff={max_diff:.6}");
}

// ── Softmax ───────────────────────────────────────────────────────────────────

#[test]
fn test_grad_check_softmax() {
    // Non-uniform logits — avoid all-equal inputs where finite-diff cancellation
    // can inflate error. Use moderately spread values so s_i are all > 1e-3.
    let x_data = vec![1.0f32, 2.0, 0.5, 3.0, 1.5, 2.5];
    let x_shape = vec![2, 3];

    for axis in 0..2usize {
        let max_diff = check_grad_1(
            x_data.clone(),
            x_shape.clone(),
            move |ctx, tape, x| softmax(ctx, tape, x, axis),
            {
                let shape = x_shape.clone();
                move |x_data| {
                    let t = Tensor::from_vec(x_data, shape.clone(), Device::Cpu);
                    let ctx = cpu_ctx();
                    let mut tape = Tape::new();
                    softmax(&ctx, &mut tape, &t, axis).to_vec()
                }
            },
            &format!("softmax_axis{axis}"),
        );
        println!("PASS: softmax axis={axis} grad_check — max_diff={max_diff:.6}");
    }
}

// ── Embedding ────────────────────────────────────────────────────────────────

#[test]
fn test_grad_check_embedding() {
    // weight: [5, 4] — 5 tokens, embed_dim=4.
    // indices: [3, 0, 2, 0, 1] — index 0 appears twice (tests scatter-add).
    let w_data: Vec<f32> = (0..20).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let w_shape = vec![5usize, 4];
    let indices = vec![3usize, 0, 2, 0, 1];
    let index_shape = vec![5usize];

    let max_diff = check_grad_1(
        w_data.clone(),
        w_shape.clone(),
        {
            let idx = indices.clone();
            let ishape = index_shape.clone();
            move |ctx, tape, w| embedding(ctx, tape, w, &idx, ishape.clone())
        },
        {
            let idx = indices.clone();
            let ws = w_shape.clone();
            let ishape = index_shape.clone();
            move |wd| {
                let wt = Tensor::from_vec(wd, ws.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                embedding(&ctx, &mut tape, &wt, &idx, ishape.clone()).to_vec()
            }
        },
        "embedding",
    );
    println!("PASS: embedding grad_check — max_diff={max_diff:.6}");
}

// ── Scaled dot-product attention ─────────────────────────────────────────────

/// Grad check for scaled_dot_product_attention: Q, K, and V are each verified
/// with central finite differences. A single backward pass provides all three
/// analytical gradients.
#[test]
fn test_grad_check_sdpa() {
    // Small shapes to keep finite-diff tractable.
    // batch=1, heads=1, seq_q=2, seq_k=3, d_k=4, d_v=4.
    let batch = 1usize;
    let heads = 1usize;
    let sq = 2usize;
    let sk = 3usize;
    let dk = 4usize;
    let dv = 4usize;

    let q_shape = vec![batch, heads, sq, dk];
    let k_shape = vec![batch, heads, sk, dk];
    let v_shape = vec![batch, heads, sk, dv];

    // Use moderate values — avoid very large logits that cause near-zero
    // softmax gradients and inflate finite-diff error.
    let q_data: Vec<f32> = (0..batch * heads * sq * dk)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..batch * heads * sk * dk)
        .map(|i| (i as f32 + 1.0) * 0.1 - 0.15)
        .collect();
    let v_data: Vec<f32> = (0..batch * heads * sk * dv)
        .map(|i| (i as f32 + 1.0) * 0.2 - 0.5)
        .collect();

    // ── Analytical: one backward pass ───────────────────────────────────────
    let ctx = cpu_ctx();
    let mut tape = Tape::new();
    let q = Tensor::from_vec(q_data.clone(), q_shape.clone(), Device::Cpu).with_grad();
    let k = Tensor::from_vec(k_data.clone(), k_shape.clone(), Device::Cpu).with_grad();
    let v = Tensor::from_vec(v_data.clone(), v_shape.clone(), Device::Cpu).with_grad();
    tape.push(Node::leaf(q.node));
    tape.push(Node::leaf(k.node));
    tape.push(Node::leaf(v.node));
    let out = scaled_dot_product_attention(&ctx, &mut tape, &q, &k, &v);
    let mut store = TensorStore::new();
    backward(&tape, &mut store, &out);
    let analytic_q = q.grad_strict(&store).to_vec();
    let analytic_k = k.grad_strict(&store).to_vec();
    let analytic_v = v.grad_strict(&store).to_vec();

    // ── Numerical ────────────────────────────────────────────────────────────
    let num_q = numerical_grad(&q_data, {
        let k = k_data.clone(); let v = v_data.clone();
        let qs = q_shape.clone(); let ks = k_shape.clone(); let vs = v_shape.clone();
        move |qd| {
            let qt = Tensor::from_vec(qd, qs.clone(), Device::Cpu);
            let kt = Tensor::from_vec(k.clone(), ks.clone(), Device::Cpu);
            let vt = Tensor::from_vec(v.clone(), vs.clone(), Device::Cpu);
            let ctx = cpu_ctx(); let mut tape = Tape::new();
            scaled_dot_product_attention(&ctx, &mut tape, &qt, &kt, &vt).to_vec()
        }
    });

    let num_k = numerical_grad(&k_data, {
        let q = q_data.clone(); let v = v_data.clone();
        let qs = q_shape.clone(); let ks = k_shape.clone(); let vs = v_shape.clone();
        move |kd| {
            let qt = Tensor::from_vec(q.clone(), qs.clone(), Device::Cpu);
            let kt = Tensor::from_vec(kd, ks.clone(), Device::Cpu);
            let vt = Tensor::from_vec(v.clone(), vs.clone(), Device::Cpu);
            let ctx = cpu_ctx(); let mut tape = Tape::new();
            scaled_dot_product_attention(&ctx, &mut tape, &qt, &kt, &vt).to_vec()
        }
    });

    let num_v = numerical_grad(&v_data, {
        let q = q_data.clone(); let k = k_data.clone();
        let qs = q_shape.clone(); let ks = k_shape.clone(); let vs = v_shape.clone();
        move |vd| {
            let qt = Tensor::from_vec(q.clone(), qs.clone(), Device::Cpu);
            let kt = Tensor::from_vec(k.clone(), ks.clone(), Device::Cpu);
            let vt = Tensor::from_vec(vd, vs.clone(), Device::Cpu);
            let ctx = cpu_ctx(); let mut tape = Tape::new();
            scaled_dot_product_attention(&ctx, &mut tape, &qt, &kt, &vt).to_vec()
        }
    });

    // ── Compare ──────────────────────────────────────────────────────────────
    let max_q = analytic_q.iter().zip(num_q.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
    let max_k = analytic_k.iter().zip(num_k.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
    let max_v = analytic_v.iter().zip(num_v.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);

    assert!(max_q < TOL, "sdpa grad_q FAIL — max_diff={max_q:.6}\n  analytic={analytic_q:?}\n  numeric={num_q:?}");
    assert!(max_k < TOL, "sdpa grad_k FAIL — max_diff={max_k:.6}\n  analytic={analytic_k:?}\n  numeric={num_k:?}");
    assert!(max_v < TOL, "sdpa grad_v FAIL — max_diff={max_v:.6}\n  analytic={analytic_v:?}\n  numeric={num_v:?}");

    println!("PASS: sdpa grad_check — max_diff_q={max_q:.6}, max_diff_k={max_k:.6}, max_diff_v={max_v:.6}");
}

// ── Layer norm ────────────────────────────────────────────────────────────────

/// Grad check for layer_norm: x, weight, and bias are each verified with
/// central finite differences. A single backward pass provides all three
/// analytical gradients.
#[test]
fn test_grad_check_layer_norm() {
    // Shape [3, 4]: 3 batch slices, each normalized over 4 features.
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5, 3.0, 1.0, -2.0, 0.0];
    let w_data: Vec<f32> = vec![0.8, 1.2, 0.5, 1.0];
    let b_data: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
    let x_shape = vec![3usize, 4];
    let wb_shape = vec![4usize];
    let eps = 1e-5f32;

    // ── Analytical: one backward pass, collect all three grads ──────────────
    let ctx = cpu_ctx();
    let mut tape = Tape::new();
    let x = Tensor::from_vec(x_data.clone(), x_shape.clone(), Device::Cpu).with_grad();
    let w = Tensor::from_vec(w_data.clone(), wb_shape.clone(), Device::Cpu).with_grad();
    let b = Tensor::from_vec(b_data.clone(), wb_shape.clone(), Device::Cpu).with_grad();
    tape.push(Node::leaf(x.node));
    tape.push(Node::leaf(w.node));
    tape.push(Node::leaf(b.node));
    let out = layer_norm(&ctx, &mut tape, &x, &w, &b, eps);
    let mut store = TensorStore::new();
    backward(&tape, &mut store, &out);
    let analytic_x = x.grad_strict(&store).to_vec();
    let analytic_w = w.grad_strict(&store).to_vec();
    let analytic_b = b.grad_strict(&store).to_vec();

    // ── Numerical: perturb each input independently ──────────────────────────
    let num_x = numerical_grad(&x_data, {
        let w = w_data.clone();
        let b = b_data.clone();
        let xs = x_shape.clone();
        let ws = wb_shape.clone();
        move |xd| {
            let xt = Tensor::from_vec(xd, xs.clone(), Device::Cpu);
            let wt = Tensor::from_vec(w.clone(), ws.clone(), Device::Cpu);
            let bt = Tensor::from_vec(b.clone(), ws.clone(), Device::Cpu);
            let ctx = cpu_ctx();
            let mut tape = Tape::new();
            layer_norm(&ctx, &mut tape, &xt, &wt, &bt, eps).to_vec()
        }
    });

    let num_w = numerical_grad(&w_data, {
        let x = x_data.clone();
        let b = b_data.clone();
        let xs = x_shape.clone();
        let ws = wb_shape.clone();
        move |wd| {
            let xt = Tensor::from_vec(x.clone(), xs.clone(), Device::Cpu);
            let wt = Tensor::from_vec(wd, ws.clone(), Device::Cpu);
            let bt = Tensor::from_vec(b.clone(), ws.clone(), Device::Cpu);
            let ctx = cpu_ctx();
            let mut tape = Tape::new();
            layer_norm(&ctx, &mut tape, &xt, &wt, &bt, eps).to_vec()
        }
    });

    let num_b = numerical_grad(&b_data, {
        let x = x_data.clone();
        let w = w_data.clone();
        let xs = x_shape.clone();
        let ws = wb_shape.clone();
        move |bd| {
            let xt = Tensor::from_vec(x.clone(), xs.clone(), Device::Cpu);
            let wt = Tensor::from_vec(w.clone(), ws.clone(), Device::Cpu);
            let bt = Tensor::from_vec(bd, ws.clone(), Device::Cpu);
            let ctx = cpu_ctx();
            let mut tape = Tape::new();
            layer_norm(&ctx, &mut tape, &xt, &wt, &bt, eps).to_vec()
        }
    });

    // ── Compare ──────────────────────────────────────────────────────────────
    let max_x = analytic_x.iter().zip(num_x.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
    let max_w = analytic_w.iter().zip(num_w.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
    let max_b = analytic_b.iter().zip(num_b.iter())
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);

    assert!(max_x < TOL, "layer_norm grad_x FAIL — max_diff={max_x:.6}\n  analytic={analytic_x:?}\n  numeric={num_x:?}");
    assert!(max_w < TOL, "layer_norm grad_weight FAIL — max_diff={max_w:.6}\n  analytic={analytic_w:?}\n  numeric={num_w:?}");
    assert!(max_b < TOL, "layer_norm grad_bias FAIL — max_diff={max_b:.6}\n  analytic={analytic_b:?}\n  numeric={num_b:?}");

    println!("PASS: layer_norm grad_check — max_diff_x={max_x:.6}, max_diff_w={max_w:.6}, max_diff_b={max_b:.6}");
}

// ── Matmul ────────────────────────────────────────────────────────────────────

#[test]
fn test_grad_check_matmul() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 2];

    let (da, db) = check_grad_2(
        a_data.clone(),
        a_shape.clone(),
        b_data.clone(),
        b_shape.clone(),
        matmul,
        {
            let b = b_data.clone();
            let b_shape = b_shape.clone();
            let a_shape = a_shape.clone();
            move |a_data| {
                let at = Tensor::from_vec(a_data, a_shape.clone(), Device::Cpu);
                let bt = Tensor::from_vec(b.clone(), b_shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                matmul(&ctx, &mut tape, &at, &bt).to_vec()
            }
        },
        {
            let a = a_data.clone();
            let a_shape = a_shape.clone();
            let b_shape = b_shape.clone();
            move |b_data| {
                let at = Tensor::from_vec(a.clone(), a_shape.clone(), Device::Cpu);
                let bt = Tensor::from_vec(b_data, b_shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                matmul(&ctx, &mut tape, &at, &bt).to_vec()
            }
        },
        "matmul",
    );
    println!("PASS: matmul grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

#[test]
fn test_grad_check_matmul_batched() {
    // [2,3,4] @ [2,4,2] → [2,3,2]
    let a_data: Vec<f32> = (1..=24).map(|x| x as f32 * 0.1).collect();
    let b_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.2).collect();
    let a_shape = vec![2, 3, 4];
    let b_shape = vec![2, 4, 2];

    let (da, db) = check_grad_2(
        a_data.clone(),
        a_shape.clone(),
        b_data.clone(),
        b_shape.clone(),
        matmul,
        {
            let b = b_data.clone();
            let b_shape = b_shape.clone();
            let a_shape = a_shape.clone();
            move |a_data| {
                let at = Tensor::from_vec(a_data, a_shape.clone(), Device::Cpu);
                let bt = Tensor::from_vec(b.clone(), b_shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                matmul(&ctx, &mut tape, &at, &bt).to_vec()
            }
        },
        {
            let a = a_data.clone();
            let a_shape = a_shape.clone();
            let b_shape = b_shape.clone();
            move |b_data| {
                let at = Tensor::from_vec(a.clone(), a_shape.clone(), Device::Cpu);
                let bt = Tensor::from_vec(b_data, b_shape.clone(), Device::Cpu);
                let ctx = cpu_ctx();
                let mut tape = Tape::new();
                matmul(&ctx, &mut tape, &at, &bt).to_vec()
            }
        },
        "matmul_batched",
    );
    println!("PASS: matmul_batched grad_check — max_diff_a={da:.6}, max_diff_b={db:.6}");
}

// ── Diamond accumulation ──────────────────────────────────────────────────────

/// Graph: x → mul(x, x) → y
///        x → add(x, x) → z
///        add(y, z) → out
///
/// f(x) = x^2 + 2x  →  df/dx = 2x + 2
///
/// Verifies that gradients from two separate paths through the same leaf
/// are correctly accumulated.
#[test]
fn test_grad_accumulation_diamond() {
    let ctx = cpu_ctx();
    let mut tape = Tape::new();

    let x = Tensor::from_vec(vec![2.0f32, 3.0], vec![2], Device::Cpu).with_grad();
    tape.push(Node::leaf(x.node));

    let y = mul(&ctx, &mut tape, &x, &x); // x^2
    let z = add(&ctx, &mut tape, &x, &x); // 2x
    let out = add(&ctx, &mut tape, &y, &z); // x^2 + 2x

    let mut store = TensorStore::new();
    backward(&tape, &mut store, &out);

    // df/dx = 2x + 2: for x=[2,3] → [6, 8]
    let grad = x.grad_strict(&store);
    assert!(
        (grad[0] - 6.0).abs() < 1e-5,
        "expected grad[0]=6.0, got {}",
        grad[0]
    );
    assert!(
        (grad[1] - 8.0).abs() < 1e-5,
        "expected grad[1]=8.0, got {}",
        grad[1]
    );
    println!("PASS: diamond accumulation — grad={grad:?}");
}
