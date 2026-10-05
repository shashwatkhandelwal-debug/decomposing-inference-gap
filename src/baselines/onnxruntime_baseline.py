"""ONNX Runtime Baseline Runner.

Executes exported ONNX models using official ONNX Runtime with graph optimizations enabled.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np
import onnxruntime as ort

from ..profiling.timers import BenchmarkHarness, BenchmarkResult


class ONNXRuntimeBaseline:
    """Evaluates ONNX Runtime performance under the standardized BenchmarkHarness."""

    def __init__(self, onnx_model_path: str, harness: BenchmarkHarness, num_threads: int = 1):
        self.onnx_model_path = onnx_model_path
        self.harness = harness
        self.num_threads = num_threads

        # Configure session options with full optimizations
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            self.onnx_model_path,
            opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def run_benchmark(self, sample_input: np.ndarray) -> Tuple[BenchmarkResult, np.ndarray]:
        """Runs the ONNX Runtime session and measures latency."""
        input_dict = {self.input_name: np.ascontiguousarray(sample_input)}

        def forward_fn():
            return self.session.run(None, input_dict)[0]

        # Initial validation run
        output = forward_fn()

        res = self.harness.benchmark(forward_fn)
        return res, output
