use std::collections::HashMap;

use crate::autograd::node::NodeId;

/// Gradient accumulator for a backward pass.
///
/// Stores one gradient vector per `NodeId`. The backward engine writes
/// here; the caller reads leaf gradients out via `tensor.grad(store)`.
///
/// Never holds tensor data — only gradients.
#[derive(Default)]
pub struct TensorStore {
    /// Accumulated gradients keyed by node identity.
    pub grads: HashMap<NodeId, Vec<f32>>,
}

impl TensorStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self { grads: HashMap::new() }
    }

    /// Insert or accumulate a gradient for `node_id`.
    ///
    /// If a gradient already exists, adds element-wise in place.
    /// If not, inserts `grad` directly (no clone — moves the Vec).
    pub fn accumulate(&mut self, node_id: NodeId, grad: Vec<f32>) {
        match self.grads.get_mut(&node_id) {
            Some(existing) => {
                assert_eq!(
                    existing.len(),
                    grad.len(),
                    "gradient shape mismatch for node {}: existing len {}, incoming len {}",
                    node_id,
                    existing.len(),
                    grad.len()
                );
                for (acc, g) in existing.iter_mut().zip(grad.iter()) {
                    *acc += g;
                }
            }
            None => {
                self.grads.insert(node_id, grad);
            }
        }
    }

    /// Return the gradient for `node_id`, if present.
    pub fn get(&self, node_id: NodeId) -> Option<&[f32]> {
        self.grads.get(&node_id).map(Vec::as_slice)
    }

    /// Zero all accumulated gradients. Call between optimizer steps.
    pub fn zero_grads(&mut self) {
        for v in self.grads.values_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Clear all gradient entries entirely (frees memory).
    pub fn clear(&mut self) {
        self.grads.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_new_entry() {
        let mut store = TensorStore::new();
        store.accumulate(1, vec![1.0, 2.0, 3.0]);
        assert_eq!(store.get(1), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn accumulate_existing_adds_elementwise() {
        let mut store = TensorStore::new();
        store.accumulate(1, vec![1.0, 2.0, 3.0]);
        store.accumulate(1, vec![0.5, 0.5, 0.5]);
        assert_eq!(store.get(1), Some([1.5, 2.5, 3.5].as_slice()));
    }

    #[test]
    fn zero_grads_zeroes_not_removes() {
        let mut store = TensorStore::new();
        store.accumulate(1, vec![1.0, 2.0]);
        store.zero_grads();
        assert_eq!(store.get(1), Some([0.0, 0.0].as_slice()));
    }

    #[test]
    fn get_missing_returns_none() {
        let store = TensorStore::new();
        assert!(store.get(42).is_none());
    }
}
