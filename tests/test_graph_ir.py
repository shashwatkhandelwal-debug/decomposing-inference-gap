"""Unit and correctness tests for Graph IR, manual builder, and validation."""

import pytest
import numpy as np
import torch

from src.graph.ir import Graph, Node
from src.graph.builder import build_cnn_graph, build_transformer_graph, build_from_pytorch
from src.graph.validate import validate_graph, GraphValidationError
from src.models.tiny_cnn import TinyCNN
from src.models.tiny_transformer import TinyTransformer


def test_node_creation():
    """Verify Node dataclass fields and representation."""
    node = Node(
        name="test_matmul",
        op_type="MatMul",
        inputs=["A", "B"],
        outputs=["C"],
        attributes={"transpose_b": True},
    )
    assert node.name == "test_matmul"
    assert node.op_type == "MatMul"
    assert node.inputs == ["A", "B"]
    assert node.outputs == ["C"]
    assert node.attributes.get("transpose_b") is True


def test_graph_topological_sort_valid():
    """Ensure a linear DAG sorts correctly."""
    g = Graph(inputs=["in"], outputs=["out"])
    g.add_node(Node("n2", "ReLU", ["mid"], ["out"]))
    g.add_node(Node("n1", "MatMul", ["in", "w"], ["mid"]))
    g.add_initializer("w", np.ones((4, 4)))

    sorted_nodes = g.topological_sort()
    assert [n.name for n in sorted_nodes] == ["n1", "n2"]
    assert g.is_valid_dag() is True


def test_graph_cycle_detection():
    """Ensure circular dependencies are rejected by topological sort."""
    g = Graph(inputs=["in"], outputs=["out"])
    g.add_node(Node("n1", "Add", ["in", "feedback"], ["t1"]))
    g.add_node(Node("n2", "ReLU", ["t1"], ["feedback"]))
    g.add_node(Node("n3", "MatMul", ["feedback", "w"], ["out"]))
    g.add_initializer("w", np.ones((4, 4)))

    with pytest.raises(ValueError, match="contains a cycle"):
        g.topological_sort()
    assert g.is_valid_dag() is False


def test_graph_validation_missing_input():
    """Verify that validate_graph catches missing input tensors."""
    g = Graph(inputs=["in"], outputs=["out"])
    g.add_node(Node("n1", "MatMul", ["in", "ghost_tensor"], ["out"]))

    with pytest.raises(GraphValidationError, match="requires input 'ghost_tensor'"):
        validate_graph(g)


def test_graph_validation_duplicate_node_name():
    """Verify duplicate node names are rejected."""
    g = Graph(inputs=["in"], outputs=["out"])
    g.add_node(Node("node_a", "ReLU", ["in"], ["mid"]))
    g.add_node(Node("node_a", "ReLU", ["mid"], ["out"]))

    with pytest.raises(GraphValidationError, match="Duplicate node name"):
        validate_graph(g)


def test_manual_cnn_builder():
    """Verify manual builder produces a valid DAG for TinyCNN without torch.fx."""
    model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=16)
    sample_input = torch.randn(2, 3, 16, 16)
    graph = build_from_pytorch(model, sample_input)

    assert graph.name == "tiny_cnn"
    assert "input_0" in graph.inputs
    assert "logits" in graph.outputs
    assert len(graph.nodes) > 0
    # Must pass full validation
    validate_graph(graph)


def test_manual_transformer_builder():
    """Verify manual builder produces a valid DAG for TinyTransformer without torch.fx."""
    model = TinyTransformer(num_layers=2, d_model=32, n_heads=2, d_ff=64)
    sample_input = torch.randn(2, 8, 32)
    graph = build_from_pytorch(model, sample_input)

    assert graph.name == "tiny_transformer"
    assert "input_0" in graph.inputs
    assert "logits" in graph.outputs
    assert len(graph.nodes) > 0
    validate_graph(graph)


def test_graph_clone():
    """Verify deep cloning produces an independent graph structure."""
    g = Graph(inputs=["x"], outputs=["y"])
    g.add_initializer("w", np.eye(4, dtype=np.float32))
    g.add_node(Node("n1", "MatMul", ["x", "w"], ["y"]))

    cloned = g.clone()
    assert len(cloned.nodes) == 1
    assert cloned.nodes[0].name == "n1"
    # Mutating cloned initializer shouldn't affect original
    cloned.initializers["w"][0, 0] = 999.0
    assert g.initializers["w"][0, 0] == 1.0
