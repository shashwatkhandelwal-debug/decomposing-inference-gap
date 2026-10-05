"""Affine Int8 Quantization Logic & Integer Kernels.

Implements quantize mapping from float32 to int8 and integer-arithmetic operations.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .calibration import QuantizationParams


def quantize_tensor(x: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """Quantizes a float32 tensor to signed int8.

    Formula: q = clip(round(x / s) + z, -128, 127)
    """
    scale = params.scale
    zp = params.zero_point
    scaled = np.round(x / scale) + zp
    q = np.clip(scaled, -128, 127).astype(np.int8)
    return q


def quantize_weight(w: np.ndarray, symmetric: bool = True) -> Tuple[np.ndarray, QuantizationParams]:
    """Quantizes a static weight tensor."""
    min_v = float(np.min(w))
    max_v = float(np.max(w))
    if symmetric:
        max_abs = max(abs(min_v), abs(max_v), 1e-7)
        scale = max_abs / 127.0
        zp = 0
        min_v, max_v = -max_abs, max_abs
    else:
        diff = max(max_v - min_v, 1e-7)
        scale = diff / 255.0
        zp = int(np.clip(np.round(-min_v / scale) - 128, -128, 127))

    params = QuantizationParams(scale=scale, zero_point=zp, min_val=min_v, max_val=max_v)
    q_w = quantize_tensor(w, params)
    return q_w, params


def quantized_matmul(
    q_a: np.ndarray,
    q_b: np.ndarray,
    params_a: QuantizationParams,
    params_b: QuantizationParams,
    bias_fp32: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, QuantizationParams]:
    """Computes quantized MatMul with int32 accumulator and output rescale.

    Math:
      A @ B = (s_A * (q_A - z_A)) @ (s_B * (q_B - z_B))
            = s_A * s_B * [ (q_A - z_A) @ (q_B - z_B) ]
    """
    # Integer accumulation in int32 to prevent overflow
    a_int32 = q_a.astype(np.int32) - params_a.zero_point
    b_int32 = q_b.astype(np.int32) - params_b.zero_point

    acc_int32 = np.matmul(a_int32, b_int32)

    # Rescale back to float32 representation
    scale_out = params_a.scale * params_b.scale
    out_fp32 = acc_int32.astype(np.float32) * scale_out

    if bias_fp32 is not None:
        out_fp32 += bias_fp32

    # Compute output quantization params
    max_abs = max(float(np.max(np.abs(out_fp32))), 1e-7)
    params_out = QuantizationParams(
        scale=max_abs / 127.0,
        zero_point=0,
        min_val=-max_abs,
        max_val=max_abs,
    )
    q_out = quantize_tensor(out_fp32, params_out)
    return q_out, params_out
