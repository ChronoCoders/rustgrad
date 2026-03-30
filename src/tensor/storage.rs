use std::sync::Arc;

/// Compute device for a tensor. Every tensor carries a device tag.
/// Ops assert matching devices — no silent cross-device copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device index. Unused until Phase 6.
    Cuda(usize),
}

/// Element data type. `F32` is the only active dtype in Phase 1.
/// The field exists on `Storage` to enable future multi-dtype dispatch
/// without changing the tensor API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

/// Owned, reference-counted tensor storage.
///
/// Always accessed through `Arc<Storage>`. Never cloned — all zero-copy
/// views (transpose, slice, reshape) share the same `Arc<Storage>` and
/// vary only in their `Layout`.
#[derive(Debug)]
pub struct Storage {
    /// Flat buffer of raw element data.
    pub data: Arc<Vec<f32>>,
    /// Device the data lives on.
    pub device: Device,
    /// Element type.
    pub dtype: DType,
}

impl Storage {
    /// Create new storage from a `Vec<f32>`.
    pub fn new(data: Vec<f32>, device: Device, dtype: DType) -> Self {
        Self {
            data: Arc::new(data),
            device,
            dtype,
        }
    }

    /// Wrap an existing `Arc<Vec<f32>>` without copying.
    pub fn from_arc(data: Arc<Vec<f32>>, device: Device, dtype: DType) -> Self {
        Self {
            data,
            device,
            dtype,
        }
    }

    /// Number of elements in the flat buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_new_wraps_arc() {
        let s = Storage::new(vec![1.0, 2.0, 3.0], Device::Cpu, DType::F32);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.device, Device::Cpu);
        assert_eq!(s.dtype, DType::F32);
    }

    #[test]
    fn storage_from_arc_no_copy() {
        let arc = Arc::new(vec![4.0f32, 5.0, 6.0]);
        let ptr = Arc::as_ptr(&arc);
        let s = Storage::from_arc(arc, Device::Cpu, DType::F32);
        // Same allocation — no copy occurred.
        assert_eq!(Arc::as_ptr(&s.data), ptr);
    }

    #[test]
    fn storage_empty() {
        let s = Storage::new(vec![], Device::Cpu, DType::F32);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }
}
