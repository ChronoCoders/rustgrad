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

    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
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
        let out_shape: Vec<usize> =
            shape.iter().enumerate().filter(|&(d, _)| d != axis).map(|(_, &s)| s).collect();

        let out_numel: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
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
            let src: usize = indices.iter().zip(in_strides.iter()).map(|(&i, &s)| i * s).sum();

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
}
