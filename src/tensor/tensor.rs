use std::sync::{Arc, RwLock};

use crate::autograd::node::{next_node_id, NodeId};
use crate::autograd::store::TensorStore;
use crate::tensor::layout::Layout;
use crate::tensor::storage::{DType, Device, Storage};

/// The central data structure. Represents a multi-dimensional array with
/// optional gradient tracking.
///
/// All zero-copy views (transpose, reshape, slice) produce a new `Tensor`
/// sharing the same `Arc<Storage>` with a different `Layout`.
/// No generic parameter — ever.
pub struct Tensor {
    /// Reference-counted storage. Never cloned.
    pub storage: Arc<Storage>,
    /// Describes how logical indices map to the flat buffer.
    pub layout: Layout,
    /// Whether this tensor participates in autograd.
    pub requires_grad: bool,
    /// Unique identity in the computation graph. Always present.
    pub node: NodeId,
    /// Accumulated gradient for this tensor. Only populated for leaf
    /// tensors after `backward()`. Uses `Arc<RwLock>` so multiple graph
    /// paths can accumulate into a shared grad buffer.
    pub grad: Option<Arc<RwLock<Vec<f32>>>>,
}

impl Tensor {
    // ── Constructors ────────────────────────────────────────────────────

    /// Create a tensor from a flat `Vec<f32>` and a shape.
    ///
    /// The data is laid out in row-major (C) order. `data.len()` must
    /// equal the product of `shape`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.iter().product::<usize>()`.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "from_vec: data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            numel
        );
        let storage = Arc::new(Storage::new(data, device, DType::F32));
        let layout = Layout::contiguous(shape);
        Self {
            storage,
            layout,
            requires_grad: false,
            node: next_node_id(),
            grad: None,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![0.0f32; numel], shape, device)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![1.0f32; numel], shape, device)
    }

    /// Construct directly from existing storage and layout.
    ///
    /// Used internally by ops that produce views or new tensors.
    pub fn from_storage(storage: Arc<Storage>, layout: Layout, requires_grad: bool) -> Self {
        Self {
            storage,
            layout,
            requires_grad,
            node: next_node_id(),
            grad: None,
        }
    }

    /// Enable gradient tracking on this tensor (builder style).
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    // ── Shape accessors ─────────────────────────────────────────────────

    /// Logical shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// Strides in elements per dimension.
    pub fn strides(&self) -> &[usize] {
        &self.layout.strides
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    // ── Device / dtype ───────────────────────────────────────────────────

    /// Device the storage lives on.
    pub fn device(&self) -> Device {
        self.storage.device
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.storage.dtype
    }

    // ── Data access ──────────────────────────────────────────────────────

    /// Returns `true` if the tensor is stored contiguously in memory.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Direct slice into the underlying buffer.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous. Use `to_vec()` to
    /// materialize a non-contiguous tensor first.
    pub fn data(&self) -> &[f32] {
        assert!(
            self.is_contiguous(),
            "data(): tensor is not contiguous — call to_vec() to materialize"
        );
        &self.storage.data[self.layout.offset..]
    }

    /// Materialize the logical tensor values into a new `Vec<f32>`.
    ///
    /// Iterates in logical row-major order, respecting strides and offset.
    /// Always allocates; for contiguous tensors prefer `data()`.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.numel());
        self.iter_logical(|v| result.push(v));
        result
    }

    /// Iterate over all logical elements in row-major order, invoking `f`
    /// for each value. Handles arbitrary strides and offsets.
    fn iter_logical<F: FnMut(f32)>(&self, mut f: F) {
        let data = &self.storage.data;
        let numel = self.numel();
        if numel == 0 {
            return;
        }
        // Walk the index space in row-major order.
        let ndim = self.ndim();
        if ndim == 0 {
            f(data[self.layout.offset]);
            return;
        }
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let flat = self.layout.flat_index(&indices);
            f(data[flat]);
            // Increment indices right-to-left (row-major).
            let mut carry = true;
            for d in (0..ndim).rev() {
                if carry {
                    indices[d] += 1;
                    if indices[d] < self.layout.shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
    }

    // ── Grad API ─────────────────────────────────────────────────────────

    /// Look up this tensor's gradient in `store`.
    ///
    /// Returns `None` if `requires_grad == false` or no gradient has been
    /// accumulated for this node yet.
    pub fn grad<'s>(&self, store: &'s TensorStore) -> Option<&'s [f32]> {
        if !self.requires_grad {
            return None;
        }
        store.get(self.node)
    }

    /// Look up this tensor's gradient in `store`, panicking if unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `requires_grad == false` or no gradient is present in the
    /// store for this tensor.
    pub fn grad_strict<'s>(&self, store: &'s TensorStore) -> &'s [f32] {
        assert!(
            self.requires_grad,
            "grad_strict: tensor (node={}) does not require grad",
            self.node
        );
        store
            .get(self.node)
            .unwrap_or_else(|| panic!("grad_strict: no gradient found for node {}", self.node))
    }

    /// Returns `true` if a gradient is present in `store` for this tensor.
    pub fn has_grad(&self, store: &TensorStore) -> bool {
        self.requires_grad && store.get(self.node).is_some()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.layout.shape)
            .field("strides", &self.layout.strides)
            .field("offset", &self.layout.offset)
            .field("device", &self.storage.device)
            .field("dtype", &self.storage.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("node", &self.node)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_shape_and_strides() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::Cpu);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert!(t.is_contiguous());
    }

    #[test]
    fn from_vec_data_accessible() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "data length 3 does not match shape")]
    fn from_vec_shape_mismatch_panics() {
        Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 3], Device::Cpu);
    }

    #[test]
    fn zeros_and_ones() {
        let z = Tensor::zeros(vec![2, 3], Device::Cpu);
        assert!(z.data().iter().all(|&x| x == 0.0));

        let o = Tensor::ones(vec![2, 3], Device::Cpu);
        assert!(o.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn to_vec_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data.clone(), vec![2, 3], Device::Cpu);
        assert_eq!(t.to_vec(), data);
    }

    #[test]
    fn to_vec_transposed() {
        // [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]]
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::Cpu);
        let layout_t = t.layout.transpose(0, 1);
        let view = Tensor::from_storage(Arc::clone(&t.storage), layout_t, false);
        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn node_ids_unique() {
        let a = Tensor::zeros(vec![2], Device::Cpu);
        let b = Tensor::zeros(vec![2], Device::Cpu);
        assert_ne!(a.node, b.node);
    }

    #[test]
    fn grad_api_no_requires_grad() {
        let t = Tensor::zeros(vec![3], Device::Cpu);
        let store = TensorStore::new();
        assert!(t.grad(&store).is_none());
        assert!(!t.has_grad(&store));
    }

    #[test]
    fn grad_api_with_store() {
        let t = Tensor::zeros(vec![3], Device::Cpu).with_grad();
        let mut store = TensorStore::new();
        store.accumulate(t.node, vec![1.0, 2.0, 3.0]);
        let g = t.grad(&store);
        assert_eq!(g, Some([1.0, 2.0, 3.0].as_slice()));
        assert!(t.has_grad(&store));
    }

    #[test]
    #[should_panic(expected = "grad_strict: tensor")]
    fn grad_strict_no_requires_grad_panics() {
        let t = Tensor::zeros(vec![3], Device::Cpu);
        let store = TensorStore::new();
        t.grad_strict(&store);
    }

    #[test]
    #[should_panic(expected = "grad_strict: no gradient found")]
    fn grad_strict_missing_grad_panics() {
        let t = Tensor::zeros(vec![3], Device::Cpu).with_grad();
        let store = TensorStore::new();
        t.grad_strict(&store);
    }

    #[test]
    fn device_and_dtype() {
        let t = Tensor::zeros(vec![2], Device::Cpu);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.dtype(), DType::F32);
    }
}
