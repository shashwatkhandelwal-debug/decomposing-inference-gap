"""PyTorch Eager Mode Baseline Runner.

Executes native PyTorch models directly under the standardized BenchmarkHarness.
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..profiling.timers import BenchmarkHarness, BenchmarkResult


class PyTorchEagerBaseline:
    """Evaluates native PyTorch forward pass latency."""

    def __init__(self, model: nn.Module, harness: BenchmarkHarness):
        self.model = model.eval()
        self.harness = harness

    def run_benchmark(self, sample_input: torch.Tensor) -> Tuple[BenchmarkResult, np.ndarray]:
        """Runs the PyTorch eager baseline and returns the benchmark result and output."""
        sample_input = sample_input.detach()

        @torch.no_grad()
        def forward_fn():
            return self.model(sample_input)

        # Obtain output
        with torch.no_grad():
            output = forward_fn().cpu().numpy()

        res = self.harness.benchmark(forward_fn)
        return res, output
