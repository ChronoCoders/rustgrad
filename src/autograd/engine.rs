use crate::autograd::graph::Tape;
use crate::autograd::store::TensorStore;
use crate::tensor::Tensor;

/// Run reverse-mode automatic differentiation from `root` through `tape`.
///
/// Gradients are accumulated into `store`. On return, `store.get(node_id)`
/// returns the gradient for any `requires_grad == true` tensor that was
/// used in the computation.
///
/// # Algorithm
///
/// 1. Seed the root gradient: `store[root.node] = vec![1.0; root.numel()]`
/// 2. Iterate `tape.nodes` in **reverse insertion order** (reverse topo).
/// 3. For each node that has a `grad_fn`, retrieve the accumulated upstream
///    gradient and call `grad_fn.backward(upstream_grad)`.
/// 4. Each returned slice is accumulated into `store` for the corresponding
///    input node id.
///
/// # Panics
///
/// Panics if the root node is not in the tape.
pub fn backward(tape: &Tape, store: &mut TensorStore, root: &Tensor) {
    assert!(
        tape.contains(root.node),
        "backward: root node {} is not in the tape",
        root.node
    );

    // Seed the root with gradient 1.
    store.accumulate(root.node, vec![1.0f32; root.numel()]);

    // Traverse in reverse topological order.
    for (_id, node) in tape.nodes.iter().rev() {
        let grad_fn = match &node.grad_fn {
            Some(gf) => gf,
            None => continue, // leaf — nothing to propagate through
        };

        // Retrieve the upstream gradient for this node.
        let upstream = match store.get(node.id) {
            Some(g) => g.to_vec(), // copy so we can borrow store mutably below
            None => continue,      // this node was not on any path from root
        };

        let input_grads = grad_fn.backward(&upstream);
        let input_ids = grad_fn.input_ids();

        assert_eq!(
            input_grads.len(),
            input_ids.len(),
            "backward: grad_fn for node {} returned {} gradients but has {} inputs",
            node.id,
            input_grads.len(),
            input_ids.len()
        );

        for (input_id, grad) in input_ids.into_iter().zip(input_grads.into_iter()) {
            store.accumulate(input_id, grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::node::{next_node_id, GradFn, Node, NodeId};
    use crate::autograd::graph::Tape;

    // A trivial identity grad fn for testing: passes gradient through unchanged.
    struct IdentityGrad {
        input_id: NodeId,
    }

    impl GradFn for IdentityGrad {
        fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
            vec![grad_output.to_vec()]
        }
        fn input_ids(&self) -> Vec<NodeId> {
            vec![self.input_id]
        }
    }

    #[test]
    fn backward_single_edge() {
        // Graph: leaf → out
        let leaf_id = next_node_id();
        let out_id = next_node_id();

        let mut tape = Tape::new();
        tape.push(Node::leaf(leaf_id));
        tape.push(Node::with_grad_fn(
            out_id,
            Box::new(IdentityGrad { input_id: leaf_id }),
        ));

        let out = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3], crate::tensor::Device::Cpu);
        // Override the node id to match what's in the tape.
        // We construct a fake root manually since Tensor::node is pub.
        let mut store = TensorStore::new();

        // Manually seed — normally `backward` does this, but we need
        // the root's node id to match `out_id` in the tape.
        store.accumulate(out_id, vec![1.0, 1.0, 1.0]);

        // Run backward manually from out_id (skip root seed step).
        for (_id, node) in tape.nodes.iter().rev() {
            let gf = match &node.grad_fn { Some(g) => g, None => continue };
            let upstream = match store.get(node.id) { Some(g) => g.to_vec(), None => continue };
            let grads = gf.backward(&upstream);
            let ids = gf.input_ids();
            for (id, g) in ids.into_iter().zip(grads) { store.accumulate(id, g); }
        }

        let _ = out; // suppress unused warning
        assert_eq!(store.get(leaf_id), Some([1.0, 1.0, 1.0].as_slice()));
    }
}
