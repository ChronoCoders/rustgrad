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
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    );

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
}
