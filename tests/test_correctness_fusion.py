"""Numerical correctness and equivalence tests for operator fusion (Phase 2).

Asserts fused kernels match naive execution within atol=1e-5, rtol=1e-4 across random seeds.
"""

import pytest
import numpy as np
import torch

from src.kernels.matmul_bias_relu_fused import matmul_bias_relu_fused
from src.kernels.conv_bn_relu_fused import fold_batchnorm_into_conv, conv_bn_relu_fused
from src.kernels.layernorm_fused import layernorm_fused
from src.graph.ops import op_matmul, op_add, op_relu, op_gelu, op_conv2d, op_batchnorm2d, op_layernorm
from src.graph.builder import build_from_pytorch
from src.executors.naive_executor import NaiveExecutor
from src.executors.fused_executor import FusedExecutor
from src.models.tiny_cnn import TinyCNN
from src.models.tiny_transformer import TinyTransformer


@pytest.mark.parametrize("seed", [42, 1337, 2026])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_matmul_bias_act_fusion_correctness(seed, activation):
    """Verify Fused MatMul+Bias+Act matches separate MatMul -> Add -> Act."""
    np.random.seed(seed)
    B, M, N = 4, 32, 64
    x = np.random.randn(B, M).astype(np.float32)
    w_t = np.random.randn(M, N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    # Naive step-by-step
    mm_out = op_matmul([x, w_t], {})
    add_out = op_add([mm_out, b], {})
    naive_out = op_relu([add_out], {}) if activation == "relu" else op_gelu([add_out], {})

    # Fused
    fused_out = matmul_bias_relu_fused(x, w_t, b, activation=activation)

    assert np.allclose(naive_out, fused_out, atol=1e-5, rtol=1e-4), (
        f"Fused MatMul+Bias+{activation} mismatch: max diff = {np.max(np.abs(naive_out - fused_out))}"
    )


@pytest.mark.parametrize("seed", [42, 1337, 2026])
def test_conv_bn_relu_fusion_correctness(seed):
    """Verify Conv2d + BatchNorm2d + ReLU parameter folding matches sequential evaluation."""
    np.random.seed(seed)
    N, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    conv_w = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    conv_b = np.random.randn(C_out).astype(np.float32)

    bn_gamma = np.random.uniform(0.5, 1.5, size=(C_out,)).astype(np.float32)
    bn_beta = np.random.randn(C_out).astype(np.float32)
    bn_mean = np.random.randn(C_out).astype(np.float32)
    bn_var = np.random.uniform(0.1, 2.0, size=(C_out,)).astype(np.float32)

    # Naive execution
    conv_out = op_conv2d([x, conv_w, conv_b], {"stride": (1, 1), "padding": (1, 1)})
    bn_out = op_batchnorm2d([conv_out, bn_gamma, bn_beta, bn_mean, bn_var], {"eps": 1e-5})
    naive_out = op_relu([bn_out], {})

    # Fused parameter folding & kernel
    w_fused, b_fused = fold_batchnorm_into_conv(conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps=1e-5)
    fused_out = conv_bn_relu_fused(x, w_fused, b_fused, stride=(1, 1), padding=(1, 1), apply_relu=True)

    assert np.allclose(naive_out, fused_out, atol=1e-5, rtol=1e-4), (
        f"Fused Conv+BN+ReLU mismatch: max diff = {np.max(np.abs(naive_out - fused_out))}"
    )


@pytest.mark.parametrize("seed", [42, 1337, 2026])
def test_layernorm_fusion_correctness(seed):
    """Verify single-pass fused LayerNorm matches standard LayerNorm."""
    np.random.seed(seed)
    B, S, D = 2, 8, 32
    x = np.random.randn(B, S, D).astype(np.float32)
    gamma = np.random.uniform(0.5, 1.5, size=(D,)).astype(np.float32)
    beta = np.random.randn(D).astype(np.float32)

    naive_out = op_layernorm([x, gamma, beta], {"eps": 1e-5, "axis": -1})
    fused_out = layernorm_fused(x, gamma, beta, eps=1e-5, axis=-1)

    assert np.allclose(naive_out, fused_out, atol=1e-5, rtol=1e-4), (
        f"Fused LayerNorm mismatch: max diff = {np.max(np.abs(naive_out - fused_out))}"
    )


def test_fused_executor_tiny_cnn():
    """Verify FusedExecutor produces numerically equivalent output to NaiveExecutor on TinyCNN."""
    model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=16)
    sample_input = torch.randn(2, 3, 16, 16)
    graph = build_from_pytorch(model, sample_input)

    naive_exec = NaiveExecutor(graph)
    fused_exec = FusedExecutor(graph)

    inputs = {"input_0": sample_input.numpy()}
    naive_out = naive_exec.run(inputs)["logits"]
    fused_out = fused_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(naive_out - fused_out)))
    assert np.allclose(naive_out, fused_out, atol=1e-5, rtol=1e-4), (
        f"TinyCNN FusedExecutor mismatch: max diff = {max_diff}"
    )


def test_fused_executor_tiny_transformer():
    """Verify FusedExecutor produces numerically equivalent output to NaiveExecutor on TinyTransformer."""
    model = TinyTransformer(num_layers=2, d_model=32, n_heads=2, d_ff=64)
    sample_input = torch.randn(2, 8, 32)
    graph = build_from_pytorch(model, sample_input)

    naive_exec = NaiveExecutor(graph)
    fused_exec = FusedExecutor(graph)

    inputs = {"input_0": sample_input.numpy()}
    naive_out = naive_exec.run(inputs)["logits"]
    fused_out = fused_exec.run(inputs)["logits"]

    max_diff = float(np.max(np.abs(naive_out - fused_out)))
    assert np.allclose(naive_out, fused_out, atol=1e-5, rtol=1e-4), (
        f"TinyTransformer FusedExecutor mismatch: max diff = {max_diff}"
    )
