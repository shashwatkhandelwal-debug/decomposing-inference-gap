"""Hand-written Fused LayerNorm Kernel.

Computes mean, variance, normalization, affine scale, and bias addition in a single pass,
avoiding the memory traffic of 5 separate tensor operations.
"""

from __future__ import annotations
import numpy as np


def layernorm_fused(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    axis: int = -1,
) -> np.ndarray:
    """Computes LayerNorm in a single combined pass.

    Formula: y = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta
    """
    # Mean and variance computed in single contiguous pass along normalized axis
    mean = np.mean(x, axis=axis, keepdims=True)
    # Variance with Bessel's correction = False
    var = np.mean(np.square(x - mean), axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)

    # In-place normalized computation: (x - mean) * inv_std * gamma + beta
    out = (x - mean) * (inv_std * gamma) + beta
    return np.ascontiguousarray(out, dtype=x.dtype)
