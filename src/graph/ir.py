"""Minimal Computation Graph Intermediate Representation (IR).

This module defines the fundamental Node and Graph abstractions representing
an execution DAG without relying on black-box compilers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class Node:
    """Represents an operator invocation in the computation graph."""
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, np.ndarray] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Node(name={self.name!r}, op_type={self.op_type!r}, "
            f"inputs={self.inputs}, outputs={self.outputs}, "
            f"attributes={self.attributes})"
        )


class Graph:
    """Represents a topologically sorted computation graph (DAG)."""

    def __init__(
        self,
        name: str = "computation_graph",
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.nodes: List[Node] = []
        self.inputs: List[str] = inputs or []
        self.outputs: List[str] = outputs or []
        self.initializers: Dict[str, np.ndarray] = {}
        self.tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        self.tensor_dtypes: Dict[str, str] = {}

    def add_node(self, node: Node) -> Node:
        """Appends a node to the graph."""
        self.nodes.append(node)
        return node

    def add_initializer(self, name: str, data: np.ndarray) -> None:
        """Registers a constant parameter tensor (weight/bias)."""
        self.initializers[name] = np.asarray(data)
        self.tensor_shapes[name] = tuple(self.initializers[name].shape)
        self.tensor_dtypes[name] = str(self.initializers[name].dtype)

    def set_tensor_meta(self, name: str, shape: Tuple[int, ...], dtype: str = "float32") -> None:
        """Sets shape and dtype metadata for an intermediate or input tensor."""
        self.tensor_shapes[name] = tuple(shape)
        self.tensor_dtypes[name] = dtype

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """Looks up a node by its unique name."""
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def get_producers(self) -> Dict[str, Node]:
        """Returns a mapping from tensor output name to the Node that produces it."""
        producers = {}
        for node in self.nodes:
            for out in node.outputs:
                producers[out] = node
        return producers

    def get_consumers(self) -> Dict[str, List[Node]]:
        """Returns a mapping from tensor name to a list of Nodes that consume it."""
        consumers: Dict[str, List[Node]] = {}
        for node in self.nodes:
            for inp in node.inputs:
                consumers.setdefault(inp, []).append(node)
        return consumers

    def is_valid_dag(self) -> bool:
        """Verifies if the graph is acyclic using Kahn's algorithm."""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def topological_sort(self) -> List[Node]:
        """Sorts nodes topologically, ensuring every producer precedes its consumers.

        Raises:
            ValueError: If a cycle or missing dependency is detected.
        """
        in_degree: Dict[str, int] = {node.name: 0 for node in self.nodes}
        producers = self.get_producers()
        node_map = {node.name: node for node in self.nodes}
        
        # Build adjacency for nodes
        dependents: Dict[str, List[str]] = {node.name: [] for node in self.nodes}
        for node in self.nodes:
            for inp in node.inputs:
                if inp in producers:
                    prod_node = producers[inp]
                    if prod_node.name != node.name:
                        dependents[prod_node.name].append(node.name)
                        in_degree[node.name] = in_degree.get(node.name, 0) + 1

        # Queue nodes with in_degree 0
        queue = [n.name for n in self.nodes if in_degree.get(n.name, 0) == 0]
        sorted_nodes: List[Node] = []

        while queue:
            curr_name = queue.pop(0)
            sorted_nodes.append(node_map[curr_name])
            for dep_name in dependents.get(curr_name, []):
                in_degree[dep_name] -= 1
                if in_degree[dep_name] == 0:
                    queue.append(dep_name)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError(
                f"Graph contains a cycle or unresolved dependencies: "
                f"{len(sorted_nodes)} of {len(self.nodes)} sorted."
            )

        self.nodes = sorted_nodes
        return self.nodes

    def clone(self) -> Graph:
        """Deep copies the graph structure and initializers."""
        new_g = Graph(
            name=self.name,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
        )
        for name, data in self.initializers.items():
            new_g.add_initializer(name, data.copy())
        new_g.tensor_shapes = dict(self.tensor_shapes)
        new_g.tensor_dtypes = dict(self.tensor_dtypes)
        for node in self.nodes:
            new_node = Node(
                name=node.name,
                op_type=node.op_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                attributes=dict(node.attributes),
                params={k: v.copy() for k, v in node.params.items()},
            )
            new_g.add_node(new_node)
        return new_g
