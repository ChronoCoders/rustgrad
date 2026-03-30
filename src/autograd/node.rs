use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::tensor::{Layout, Storage};

/// Unique identifier for every node in the computation graph.
/// Leaf tensors and op outputs all carry a `NodeId`.
pub type NodeId = u64;

static NODE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate the next unique `NodeId`. Thread-safe, monotonically increasing,
/// never returns 0.
pub fn next_node_id() -> NodeId {
    NODE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A backward function attached to an op output node.
///
/// Receives the upstream gradient and returns one gradient slice per input
/// to the op, in the same order as the op's inputs.
///
/// Implementations must be `Send + Sync` — gradients may be computed
/// across threads in future phases.
pub trait GradFn: Send + Sync {
    /// Compute input gradients from the upstream gradient.
    ///
    /// Returns `Vec<Vec<f32>>` with one entry per op input.
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>>;

    /// Node IDs of the op inputs. Used by the backward engine to route
    /// gradients to the correct accumulators.
    fn input_ids(&self) -> Vec<NodeId>;
}

/// A node in the computation graph.
///
/// Leaf tensors have `grad_fn == None`. Op output tensors carry a boxed
/// `GradFn` and the list of input `NodeId`s it corresponds to.
pub struct Node {
    /// The node's own identity.
    pub id: NodeId,
    /// Present only for non-leaf nodes (op outputs).
    pub grad_fn: Option<Box<dyn GradFn>>,
}

impl Node {
    /// Create a leaf node — no grad function.
    pub fn leaf(id: NodeId) -> Self {
        Self { id, grad_fn: None }
    }

    /// Create an op output node with a gradient function.
    pub fn with_grad_fn(id: NodeId, grad_fn: Box<dyn GradFn>) -> Self {
        Self {
            id,
            grad_fn: Some(grad_fn),
        }
    }

    /// Returns `true` if this is a leaf node (no grad function).
    pub fn is_leaf(&self) -> bool {
        self.grad_fn.is_none()
    }
}

/// A `GradFn` that holds one input's storage and layout — used by
/// single-input ops (unary). Multi-input ops define their own structs.
///
/// This is a convenience base; actual grad fns are op-specific.
pub struct StorageCapture {
    pub storage: Arc<Storage>,
    pub layout: Layout,
    pub node_id: NodeId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ids_are_unique_and_nonzero() {
        let a = next_node_id();
        let b = next_node_id();
        let c = next_node_id();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert_ne!(c, 0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn leaf_node_has_no_grad_fn() {
        let id = next_node_id();
        let node = Node::leaf(id);
        assert!(node.is_leaf());
        assert!(node.grad_fn.is_none());
    }
}
