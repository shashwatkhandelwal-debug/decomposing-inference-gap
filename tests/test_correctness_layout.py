"""Correctness and numerical validation for memory layout optimizations (Phase 3)."""

import pytest
import numpy as np
import torch

from src.kernels.tiled_matmul import tiled_matmul
from src.graph.builder import build_from_pytorch
from src.executors.naive_executor import NaiveExecutor
from src.executors.layout_executor import LayoutExecutor
from src.models.tiny_cnn import TinyCNN
from src.models.tiny_transformer import TinyTransformer


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("seed", [42, 1337])
def test_tiled_matmul_correctness(block_size, seed):
    """Verify tiled matrix multiplication matches NumPy BLAS matmul numerically."""
    np.random.seed(seed)
    M, K, N = 128, 128, 128
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    ref = np.matmul(A, B)
    tiled_res = tiled_matmul(A, B, block_size=block_size)

    max_diff = float(np.max(np.abs(ref - tiled_res)))
    assert np.allclose(ref, tiled_res, atol=1e-5, rtol=1e-4), (
        f"Tiled matmul (block={block_size}) mismatch: max diff = {max_diff}"
    )


def test_layout_executor_tiny_cnn():
    """Verify LayoutExecutor produces identical output to NaiveExecutor on TinyCNN."""
    model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=16)
    sample_input = torch.randn(2, 3, 16, 16)
    graph = build_from_pytorch(model, sample_input)

    naive_exec = NaiveExecutor(graph)
    layout_exec = LayoutExecutor(graph, enable_buffer_pool=True)

    inputs = {"input_0": sample_input.numpy()}
    naive_out = naive_exec.run(inputs)["logits"]
    layout_out = layout_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(naive_out - layout_out)))
    assert np.allclose(naive_out, layout_out, atol=1e-5, rtol=1e-4), (
        f"TinyCNN LayoutExecutor mismatch: max diff = {max_diff}"
    )


def test_layout_executor_tiny_transformer():
    """Verify LayoutExecutor produces identical output to NaiveExecutor on TinyTransformer."""
    model = TinyTransformer(num_layers=2, d_model=32, n_heads=2, d_ff=64)
    sample_input = torch.randn(2, 8, 32)
    graph = build_from_pytorch(model, sample_input)

    naive_exec = NaiveExecutor(graph)
    layout_exec = LayoutExecutor(graph, enable_buffer_pool=True)

    inputs = {"input_0": sample_input.numpy()}
    naive_out = naive_exec.run(inputs)["logits"]
    layout_out = layout_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(naive_out - layout_out)))
    assert np.allclose(naive_out, layout_out, atol=1e-5, rtol=1e-4), (
        f"TinyTransformer LayoutExecutor mismatch: max diff = {max_diff}"
    )
