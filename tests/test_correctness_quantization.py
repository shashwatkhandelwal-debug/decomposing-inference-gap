"""Numerical validation and error bounds for affine int8 quantization (Phase 4)."""

import pytest
import numpy as np
import torch

from src.quantization.calibration import Calibrator, QuantizationParams
from src.quantization.quantize import quantize_tensor, quantize_weight, quantized_matmul
from src.quantization.dequantize import dequantize_tensor
from src.graph.builder import build_from_pytorch
from src.executors.naive_executor import NaiveExecutor
from src.executors.quantized_executor import QuantizedExecutor
from src.models.tiny_cnn import TinyCNN
from src.models.tiny_transformer import TinyTransformer


@pytest.mark.parametrize("seed", [42, 1337])
def test_affine_quantize_dequantize_roundtrip(seed):
    """Verify quantization roundtrip error is strictly bounded by half-scale step size."""
    np.random.seed(seed)
    x = np.random.uniform(-5.0, 5.0, size=(100, 100)).astype(np.float32)

    calib = Calibrator(method="minmax")
    calib.observe("x", x)
    params = calib.compute_params("x", symmetric=True)

    q = quantize_tensor(x, params)
    assert q.dtype == np.int8
    assert np.all(q >= -128) and np.all(q <= 127)

    x_rec = dequantize_tensor(q, params)
    max_error = float(np.max(np.abs(x - x_rec)))

    # Theoretical maximum reconstruction error for uniform rounding is scale / 2
    assert max_error <= (params.scale / 2.0) + 1e-5, (
        f"Quantization roundtrip error {max_error} exceeds theoretical step bound {params.scale / 2.0}"
    )


def test_quantized_matmul_error_bound():
    """Verify quantized integer matmul stays within acceptable quantization error envelope."""
    np.random.seed(42)
    M, K, N = 32, 64, 32
    A = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)

    ref_out = np.matmul(A, B)

    q_a, params_a = quantize_weight(A, symmetric=True)
    q_b, params_b = quantize_weight(B, symmetric=True)

    q_out, params_out = quantized_matmul(q_a, q_b, params_a, params_b)
    out_rec = dequantize_tensor(q_out, params_out)

    # Relative L2 error should be small (< 5%) for well-conditioned matrices
    rel_l2_err = np.linalg.norm(ref_out - out_rec) / np.linalg.norm(ref_out)
    assert rel_l2_err < 0.05, f"Quantized matmul relative error too high: {rel_l2_err:.4f}"


def test_quantized_executor_tiny_cnn():
    """Verify QuantizedExecutor runs on TinyCNN and produces bounded output."""
    model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=16)
    sample_input = torch.randn(2, 3, 16, 16)
    graph = build_from_pytorch(model, sample_input)

    naive = NaiveExecutor(graph)
    q_exec = QuantizedExecutor(graph)

    inputs = {"input_0": sample_input.numpy()}
    fp32_out = naive.run(inputs)["logits"]
    q_out = q_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(fp32_out - q_out)))
    assert max_diff < 0.5, f"Quantized TinyCNN output deviated significantly: {max_diff}"


def test_quantized_executor_tiny_transformer():
    """Verify QuantizedExecutor runs on TinyTransformer and produces bounded output."""
    model = TinyTransformer(num_layers=2, d_model=32, n_heads=2, d_ff=64)
    sample_input = torch.randn(2, 8, 32)
    graph = build_from_pytorch(model, sample_input)

    naive = NaiveExecutor(graph)
    q_exec = QuantizedExecutor(graph)

    inputs = {"input_0": sample_input.numpy()}
    fp32_out = naive.run(inputs)["logits"]
    q_out = q_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(fp32_out - q_out)))
    assert max_diff < 1.0, f"Quantized TinyTransformer output deviated significantly: {max_diff}"
