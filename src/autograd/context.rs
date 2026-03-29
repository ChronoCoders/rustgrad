use std::sync::Arc;

use crate::backend::traits::Backend;
use crate::tensor::Device;

/// The single entry point for all compute operations.
///
/// Passed into every op function. Owns the backend and the target device.
/// Using `Arc<dyn Backend>` keeps `Tensor` generic-free — dispatch is
/// at runtime, not compile time.
pub struct Context {
    /// The compute backend (CPU, CUDA, etc.).
    pub backend: Arc<dyn Backend>,
    /// Device associated with this context.
    pub device: Device,
}

impl Context {
    /// Construct a context with the given backend and device.
    pub fn new(backend: Arc<dyn Backend>, device: Device) -> Self {
        Self { backend, device }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    #[test]
    fn context_cpu_backend() {
        let ctx = Context::new(Arc::new(CpuBackend), Device::Cpu);
        assert_eq!(ctx.device, Device::Cpu);
    }
}
