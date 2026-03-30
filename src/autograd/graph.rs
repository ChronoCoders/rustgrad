use indexmap::IndexMap;

use crate::autograd::node::{Node, NodeId};

/// The computation graph. Records every op output in insertion order.
///
/// Insertion order equals topological order — no separate topo sort is
/// needed. `backward()` iterates in reverse insertion order to propagate
/// gradients from outputs to inputs.
#[derive(Default)]
pub struct Tape {
    /// Nodes keyed by `NodeId`, stored in insertion (topological) order.
    pub nodes: IndexMap<NodeId, Node>,
}

impl Tape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
        }
    }

    /// Record a node. The caller is responsible for inserting nodes in
    /// topological order (i.e., inputs before outputs).
    pub fn push(&mut self, node: Node) {
        self.nodes.insert(node.id, node);
    }

    /// Returns `true` if `node_id` is recorded in the tape.
    pub fn contains(&self, node_id: NodeId) -> bool {
        self.nodes.contains_key(&node_id)
    }

    /// Number of nodes recorded.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tape has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Print a human-readable summary of the tape to stdout.
///
/// Each line shows the insertion index, node id, and its input ids:
/// ```text
/// Node 0 [id=3] inputs=[]
/// Node 1 [id=5] inputs=[3]
/// Node 2 [id=7] inputs=[3, 5]
/// ```
pub fn print_graph(tape: &Tape) {
    for (i, (_id, node)) in tape.nodes.iter().enumerate() {
        let input_ids: Vec<String> = node
            .grad_fn
            .as_ref()
            .map(|gf| gf.input_ids().iter().map(|id| id.to_string()).collect())
            .unwrap_or_default();
        println!(
            "Node {} [id={}] inputs=[{}]",
            i,
            node.id,
            input_ids.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::node::{next_node_id, Node};

    #[test]
    fn tape_push_and_contains() {
        let mut tape = Tape::new();
        let id = next_node_id();
        tape.push(Node::leaf(id));
        assert!(tape.contains(id));
        assert_eq!(tape.len(), 1);
    }

    #[test]
    fn tape_insertion_order_preserved() {
        let mut tape = Tape::new();
        let ids: Vec<NodeId> = (0..5).map(|_| next_node_id()).collect();
        for &id in &ids {
            tape.push(Node::leaf(id));
        }
        let recorded: Vec<NodeId> = tape.nodes.keys().copied().collect();
        assert_eq!(recorded, ids);
    }

    #[test]
    fn tape_empty_by_default() {
        let tape = Tape::new();
        assert!(tape.is_empty());
    }
}
