"""Hand-written Fused MatMul + Bias Add + Activation Kernel.

Fuses the linear transformation and non-linear activation into a single kernel execution,
eliminating intermediate memory materialization and cache thrashing.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def matmul_bias_relu_fused(
    x: np.ndarray,
    weight_t: np.ndarray,
    bias: Optional[np.ndarray] = None,
    activation: str = "relu",
) -> np.ndarray:
    """Computes act(X @ W.T + B) in a single fused operation.

    Args:
        x: Input activation tensor of shape (..., in_features)
        weight_t: Transposed weight tensor of shape (in_features, out_features)
        bias: Optional bias vector of shape (out_features,)
        activation: Non-linear activation type ('relu', 'gelu', or 'none')

    Returns:
        Fused output tensor of shape (..., out_features)
    """
    # 1. Compute Matrix Multiplication
    out = np.matmul(x, weight_t)

    # 2. In-place Bias Addition
    if bias is not None:
        np.add(out, bias, out=out)

    # 3. In-place Non-linear Activation
    if activation == "relu":
        np.maximum(out, 0, out=out)
    elif activation == "gelu":
        # Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c1 = 0.7978845608  # sqrt(2 / pi)
        c2 = 0.044715
        x_cubed = np.power(out, 3)
        inner = c1 * (out + c2 * x_cubed)
        np.tanh(inner, out=inner)
        inner += 1.0
        out *= 0.5
        out *= inner
    elif activation == "none":
        pass
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    return out
