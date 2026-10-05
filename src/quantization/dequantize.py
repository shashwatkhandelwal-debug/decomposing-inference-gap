"""Dequantization Logic (Phase 4).

Converts int8 quantized tensors back to float32 representation for sensitive ops.
"""

from __future__ import annotations
import numpy as np
from .calibration import QuantizationParams


def dequantize_tensor(q: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """Dequantizes signed int8 tensor to float32.

    Formula: x_approx = (q - z) * s
    """
    scale = params.scale
    zp = params.zero_point
    return (q.astype(np.float32) - zp) * scale
