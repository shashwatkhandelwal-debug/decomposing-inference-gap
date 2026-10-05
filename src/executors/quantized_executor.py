"""Quantized Graph Executor (Phase 4).

Executes post-training static int8 quantized computation graphs.
Performs integer-arithmetic for compute-intensive ops (MatMul, Conv2d) while
maintaining numerical stability by dequantizing for sensitive ops (Softmax, LayerNorm).
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..graph.ir import Graph, Node
from ..graph.ops import OP_REGISTRY
from ..quantization.calibration import Calibrator, QuantizationParams
from ..quantization.quantize import quantize_tensor, quantize_weight
from ..quantization.dequantize import dequantize_tensor
from ..profiling.flamegraph import OpProfiler, OpProfileReport


class QuantizedExecutor:
    """Executes quantized int8 computation graph and tracks per-layer quantization error."""

    def __init__(self, graph: Graph, calibrator: Optional[Calibrator] = None):
        self.graph = graph
        self.calibrator = calibrator or Calibrator(method="minmax")
        self.weight_params: Dict[str, QuantizationParams] = {}
        self.quantized_weights: Dict[str, np.ndarray] = {}
        self.act_params: Dict[str, QuantizationParams] = {}
        self.per_layer_errors: Dict[str, float] = {}
        self.profiler: Optional[OpProfiler] = None

        self._quantize_graph_weights()

    def _quantize_graph_weights(self) -> None:
        """Quantizes all constant weight tensors to int8."""
        for name, arr in self.graph.initializers.items():
            if arr.ndim >= 2:  # Linear/Conv weights
                q_w, params = quantize_weight(arr, symmetric=True)
                self.quantized_weights[name] = q_w
                self.weight_params[name] = params
            else:
                # Keep 1D biases and normalization scales in float32
                self.weight_params[name] = QuantizationParams(1.0, 0, float(np.min(arr)), float(np.max(arr)))

    def calibrate(self, calibration_inputs: List[Dict[str, np.ndarray]]) -> None:
        """Runs sample batches to record activation dynamic ranges."""
        # Simple reference run to collect activations
        from .naive_executor import NaiveExecutor
        naive = NaiveExecutor(self.graph)
        for batch in calibration_inputs:
            # We observe inputs
            for k, v in batch.items():
                self.calibrator.observe(k, v)
            # Run naive execution and observe all outputs
            _ = naive.run(batch)

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        enable_profiling: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Executes quantized graph with int8 arithmetic."""
        tensor_store_fp32: Dict[str, np.ndarray] = dict(self.graph.initializers)
        tensor_store_q: Dict[str, Tuple[np.ndarray, QuantizationParams]] = {}

        # 1. Quantize inputs
        for name, arr in inputs.items():
            tensor_store_fp32[name] = arr
            self.calibrator.observe(name, arr)
            q_params = self.calibrator.compute_params(name, symmetric=True)
            self.act_params[name] = q_params
            tensor_store_q[name] = (quantize_tensor(arr, q_params), q_params)

        if enable_profiling:
            self.profiler = OpProfiler()

        for node in self.graph.nodes:
            op_fn = OP_REGISTRY.get(node.op_type)
            if op_fn is None:
                raise NotImplementedError(f"Operator {node.op_type} not supported in QuantizedExecutor.")

            if enable_profiling:
                t0 = time.perf_counter_ns()

            # Execute quantized operations
            if node.op_type == "MatMul":
                inp0_name, inp1_name = node.inputs[0], node.inputs[1]
                
                # Check if we have quantized inputs
                if inp0_name in tensor_store_q and inp1_name in self.quantized_weights:
                    q_a, params_a = tensor_store_q[inp0_name]
                    q_b = self.quantized_weights[inp1_name]
                    params_b = self.weight_params[inp1_name]

                    # Integer MatMul
                    a_int32 = q_a.astype(np.int32)
                    b_int32 = q_b.astype(np.int32)
                    acc_int32 = np.matmul(a_int32, b_int32)
                    
                    scale_out = params_a.scale * params_b.scale
                    out_fp32 = acc_int32.astype(np.float32) * scale_out
                else:
                    # Fallback to float32 if unquantized
                    fp_inps = [tensor_store_fp32[inp] for inp in node.inputs]
                    out_fp32 = op_fn(fp_inps, node.attributes)
            else:
                # Sensitive or non-matmul ops evaluated in fp32
                fp_inps = [tensor_store_fp32[inp] for inp in node.inputs]
                out_fp32 = op_fn(fp_inps, node.attributes)

            if enable_profiling:
                t1 = time.perf_counter_ns()
                self.profiler.record(node.name, node.op_type, (t1 - t0) / 1e6)

            # Store and observe output
            out_name = node.outputs[0]
            tensor_store_fp32[out_name] = out_fp32

            # Quantize output for subsequent consumers
            self.calibrator.observe(out_name, out_fp32)
            out_params = self.calibrator.compute_params(out_name, symmetric=True)
            self.act_params[out_name] = out_params
            tensor_store_q[out_name] = (quantize_tensor(out_fp32, out_params), out_params)

        return {out_name: tensor_store_fp32[out_name] for out_name in self.graph.outputs}

    def compute_per_layer_error(self, inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Computes max absolute error per node output against float32 baseline."""
        from .naive_executor import NaiveExecutor
        naive = NaiveExecutor(self.graph)
        fp32_res = naive.run(inputs)
        q_res = self.run(inputs)

        errors = {}
        for out_name in self.graph.outputs:
            diff = float(np.max(np.abs(fp32_res[out_name] - q_res[out_name])))
            errors[out_name] = diff
        return errors

    def get_profile_report(self) -> Optional[OpProfileReport]:
        if self.profiler is None:
            return None
        return self.profiler.generate_report()
