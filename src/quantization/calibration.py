"""Calibration routines for post-training static quantization (Phase 4).

Records activation dynamic ranges across calibration batches to derive scale and zero-point parameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class QuantizationParams:
    scale: float
    zero_point: int
    min_val: float
    max_val: float
    dtype: str = "int8"


class Calibrator:
    """Collects activation statistics to calculate optimal quantization scales."""

    def __init__(self, method: str = "minmax", percentile: float = 99.99):
        self.method = method
        self.percentile = percentile
        self.min_vals: Dict[str, float] = {}
        self.max_vals: Dict[str, float] = {}
        self.histograms: Dict[str, List[np.ndarray]] = {}

    def observe(self, tensor_name: str, tensor: np.ndarray) -> None:
        """Records values for a named tensor during a calibration forward pass."""
        arr = np.asarray(tensor, dtype=np.float32)
        if self.method == "minmax":
            cur_min = float(np.min(arr))
            cur_max = float(np.max(arr))
            self.min_vals[tensor_name] = min(self.min_vals.get(tensor_name, cur_min), cur_min)
            self.max_vals[tensor_name] = max(self.max_vals.get(tensor_name, cur_max), cur_max)
        elif self.method == "percentile":
            low_p = (100.0 - self.percentile) / 2.0
            high_p = 100.0 - low_p
            cur_min = float(np.percentile(arr, low_p))
            cur_max = float(np.percentile(arr, high_p))
            self.min_vals[tensor_name] = min(self.min_vals.get(tensor_name, cur_min), cur_min)
            self.max_vals[tensor_name] = max(self.max_vals.get(tensor_name, cur_max), cur_max)

    def compute_params(self, tensor_name: str, symmetric: bool = True) -> QuantizationParams:
        """Computes affine scale and zero-point for int8 range [-128, 127]."""
        min_v = self.min_vals.get(tensor_name, -1.0)
        max_v = self.max_vals.get(tensor_name, 1.0)

        # Prevent divide-by-zero on flat tensors
        if abs(max_v - min_v) < 1e-7:
            max_v = min_v + 1e-4

        if symmetric:
            max_abs = max(abs(min_v), abs(max_v))
            scale = max_abs / 127.0
            zero_point = 0
            min_v = -max_abs
            max_v = max_abs
        else:
            scale = (max_v - min_v) / 255.0
            zero_point = int(np.round(-min_v / scale)) - 128
            zero_point = int(np.clip(zero_point, -128, 127))

        return QuantizationParams(
            scale=float(scale),
            zero_point=int(zero_point),
            min_val=float(min_v),
            max_val=float(max_v),
            dtype="int8",
        )
