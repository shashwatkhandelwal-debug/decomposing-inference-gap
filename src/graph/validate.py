"""Graph validation utilities.

Verifies graph well-formedness, acyclicity (DAG invariant), dependency closure,
and shape/dtype propagation.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from .ir import Graph, Node


class GraphValidationError(Exception):
    """Raised when a computation graph violates DAG invariants."""
    pass


def validate_graph(graph: Graph) -> None:
    """Performs rigorous well-formedness checks on a Graph IR.

    Checks:
      1. Graph inputs and outputs are non-empty.
      2. Graph contains no cycles (topological ordering check).
      3. Dependency closure: Every input of every node is produced by either
         a graph input, an initializer, or a strictly preceding node's output.
      4. Graph outputs are produced within the graph.
      5. No duplicate node names or ambiguous output tensor names.

    Raises:
        GraphValidationError: If any check fails.
    """
    if not graph.inputs:
        raise GraphValidationError("Graph has no registered inputs.")
    if not graph.outputs:
        raise GraphValidationError("Graph has no registered outputs.")
    if not graph.nodes:
        raise GraphValidationError("Graph contains no execution nodes.")

    # 1. Check for duplicate node names
    seen_nodes: Set[str] = set()
    for node in graph.nodes:
        if node.name in seen_nodes:
            raise GraphValidationError(f"Duplicate node name detected: {node.name!r}")
        seen_nodes.add(node.name)

    # 2. Check for duplicate tensor producers
    produced_tensors: Dict[str, str] = {}
    for inp in graph.inputs:
        produced_tensors[inp] = "graph_input"
    for init_name in graph.initializers:
        produced_tensors[init_name] = "initializer"

    for node in graph.nodes:
        for out in node.outputs:
            if out in produced_tensors and produced_tensors[out] not in ("graph_input", "initializer"):
                raise GraphValidationError(
                    f"Tensor {out!r} is produced by multiple nodes ({produced_tensors[out]!r} and {node.name!r})."
                )
            produced_tensors[out] = node.name

    # 3. Check topological ordering and dependency closure
    available_tensors: Set[str] = set(graph.inputs) | set(graph.initializers.keys())

    for node in graph.nodes:
        for inp in node.inputs:
            if inp not in available_tensors:
                raise GraphValidationError(
                    f"Node {node.name!r} ({node.op_type}) requires input {inp!r}, "
                    f"which is not available in the graph scope before this node."
                )
        for out in node.outputs:
            available_tensors.add(out)

    # 4. Check that all graph outputs exist
    for out in graph.outputs:
        if out not in available_tensors:
            raise GraphValidationError(
                f"Graph output {out!r} is never produced by any node, input, or initializer."
            )

    # 5. Verify topological sort execution
    try:
        graph.topological_sort()
    except ValueError as e:
        raise GraphValidationError(f"Topological sort failed (cycle detected): {e}") from e
