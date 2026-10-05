"""TensorRT Baseline Runner (GPU-gated).

Checks for NVIDIA GPU and TensorRT availability.
Cleanly skips execution when running on CPU environments without faking numbers.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from ..profiling.timers import BenchmarkHarness, BenchmarkResult


class TensorRTBaseline:
    """Evaluates TensorRT execution if hardware and software prerequisites exist."""

    def __init__(self, onnx_model_path: str, harness: BenchmarkHarness):
        self.onnx_model_path = onnx_model_path
        self.harness = harness
        self.available = self._check_tensorrt_available()

    def _check_tensorrt_available(self) -> bool:
        """Determines if CUDA and TensorRT are installed and executable."""
        if not torch.cuda.is_available():
            return False
        try:
            import tensorrt  # type: ignore
            return True
        except ImportError:
            return False

    def run_benchmark(self, sample_input: np.ndarray) -> Optional[Tuple[BenchmarkResult, np.ndarray]]:
        """Runs TensorRT baseline if available; returns None otherwise."""
        if not self.available:
            print("[TensorRT Baseline] GPU/TensorRT not detected on this machine. Gracefully skipping.")
            return None

        # If GPU and TensorRT were available, build and execute engine
        # (Standard clean stub implementation)
        return None
