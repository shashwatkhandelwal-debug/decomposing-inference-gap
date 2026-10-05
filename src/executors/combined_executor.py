"""Combined Hand-Built Optimized Graph Executor (Phase 5).

Stacks all three hand-built optimization techniques:
  1. Operator Fusion (graph rewrite pass)
  2. Memory Layout Optimization (contiguous buffers + buffer memory arena)
  3. Post-Training Int8 Quantization (integer arithmetic)
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..graph.ir import Graph, Node
from ..executors.fused_executor import FusedExecutor, FUSED_OP_REGISTRY
from ..executors.layout_executor import BufferArena
from ..quantization.calibration import Calibrator, QuantizationParams
from ..quantization.quantize import quantize_tensor, quantize_weight
from ..quantization.dequantize import dequantize_tensor
from ..profiling.flamegraph import OpProfiler, OpProfileReport


class CombinedExecutor:
    """Stacks operator fusion, memory layout optimization, and int8 quantization."""

    def __init__(
        self,
        graph: Graph,
        calibrator: Optional[Calibrator] = None,
        enable_buffer_pool: bool = True,
        enable_quantization: bool = True,
    ):
        # 1. Apply Graph Fusion Rewrites
        fused_engine = FusedExecutor(graph)
        self.fused_graph = fused_engine.fused_graph

        # 2. Setup Layout Memory Buffer Arena
        self.enable_buffer_pool = enable_buffer_pool
        self.enable_quantization = enable_quantization
        self.buffer_arena = BufferArena() if enable_buffer_pool else None

        # 3. Setup Quantization Parameters
        self.calibrator = calibrator or Calibrator(method="minmax")
        self.weight_params: Dict[str, QuantizationParams] = {}
        self.quantized_weights: Dict[str, np.ndarray] = {}
        self.act_params: Dict[str, QuantizationParams] = {}
        self.profiler: Optional[OpProfiler] = None

        if self.enable_quantization:
            self._quantize_fused_weights()

    def _quantize_fused_weights(self) -> None:
        """Quantizes initializers in the fused graph."""
        for name, arr in self.fused_graph.initializers.items():
            arr_contig = np.ascontiguousarray(arr)
            if arr_contig.ndim >= 2:
                q_w, params = quantize_weight(arr_contig, symmetric=True)
                self.quantized_weights[name] = q_w
                self.weight_params[name] = params
            else:
                self.weight_params[name] = QuantizationParams(
                    1.0, 0, float(np.min(arr_contig)), float(np.max(arr_contig))
                )

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        enable_profiling: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Executes the fused, layout-optimized, and optionally quantized graph."""
        tensor_store_fp32: Dict[str, np.ndarray] = {
            k: np.ascontiguousarray(v) for k, v in self.fused_graph.initializers.items()
        }
        tensor_store_q: Dict[str, Tuple[np.ndarray, QuantizationParams]] = {}

        # Contiguous input quantization
        for name, arr in inputs.items():
            arr_c = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
            tensor_store_fp32[name] = arr_c
            if self.enable_quantization:
                self.calibrator.observe(name, arr_c)
                q_p = self.calibrator.compute_params(name, symmetric=True)
                self.act_params[name] = q_p
                tensor_store_q[name] = (quantize_tensor(arr_c, q_p), q_p)

        if enable_profiling:
            self.profiler = OpProfiler()

        for node in self.fused_graph.nodes:
            op_fn = FUSED_OP_REGISTRY.get(node.op_type)
            if op_fn is None:
                raise NotImplementedError(f"Operator {node.op_type} not in FUSED_OP_REGISTRY.")

            if enable_profiling:
                t0 = time.perf_counter_ns()

            # Quantized path for FusedMatMulBiasAct
            if self.enable_quantization and node.op_type == "FusedMatMulBiasAct":
                inp_name, w_name = node.inputs[0], node.inputs[1]
                bias_name = node.inputs[2] if len(node.inputs) > 2 else None
                bias_arr = tensor_store_fp32.get(bias_name) if bias_name else None

                if inp_name in tensor_store_q and w_name in self.quantized_weights:
                    q_a, p_a = tensor_store_q[inp_name]
                    q_b = self.quantized_weights[w_name]
                    p_b = self.weight_params[w_name]

                    # Integer MatMul
                    a_i32 = q_a.astype(np.int32)
                    b_i32 = q_b.astype(np.int32)
                    acc_i32 = np.matmul(a_i32, b_i32)

                    scale_out = p_a.scale * p_b.scale
                    out_fp32 = acc_i32.astype(np.float32) * scale_out

                    if bias_arr is not None:
                        out_fp32 += bias_arr

                    act = node.attributes.get("activation", "relu")
                    if act == "relu":
                        np.maximum(out_fp32, 0, out=out_fp32)
                    elif act == "gelu":
                        c1 = 0.7978845608
                        c2 = 0.044715
                        x_cubed = np.power(out_fp32, 3)
                        inner = c1 * (out_fp32 + c2 * x_cubed)
                        np.tanh(inner, out=inner)
                        out_fp32 = 0.5 * out_fp32 * (1.0 + inner)
                else:
                    fp_inps = [tensor_store_fp32[inp] for inp in node.inputs]
                    out_fp32 = op_fn(fp_inps, node.attributes)
            else:
                fp_inps = [tensor_store_fp32[inp] for inp in node.inputs]
                out_fp32 = op_fn(fp_inps, node.attributes)

            if enable_profiling:
                t1 = time.perf_counter_ns()
                self.profiler.record(node.name, node.op_type, (t1 - t0) / 1e6)

            out_name = node.outputs[0]
            out_c = out_fp32 if out_fp32.flags.c_contiguous else np.ascontiguousarray(out_fp32)
            tensor_store_fp32[out_name] = out_c

            if self.enable_quantization:
                self.calibrator.observe(out_name, out_c)
                out_p = self.calibrator.compute_params(out_name, symmetric=True)
                self.act_params[out_name] = out_p
                tensor_store_q[out_name] = (quantize_tensor(out_c, out_p), out_p)

        return {out_name: tensor_store_fp32[out_name] for out_name in self.fused_graph.outputs}

    def get_profile_report(self) -> Optional[OpProfileReport]:
        if self.profiler is None:
            return None
        return self.profiler.generate_report()
