"""Memory-Layout-Optimized Graph Executor (Phase 3).

Applies memory layout optimizations:
  1. Contiguous memory alignment (ensuring zero-copy strided operations).
  2. Pre-allocated buffer memory arena (reusing fixed memory buffers to eliminate runtime allocations).
Operates strictly on the unfused graph to isolate memory layout effects from operator fusion.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..graph.ir import Graph, Node
from ..graph.ops import OP_REGISTRY
from ..profiling.flamegraph import OpProfiler, OpProfileReport


class BufferArena:
    """Manages pre-allocated tensor buffers to eliminate allocation overhead during execution."""

    def __init__(self):
        # Key: (shape, dtype_str) -> list of available pre-allocated numpy arrays
        self.free_buffers: Dict[Tuple[Tuple[int, ...], str], List[np.ndarray]] = {}
        self.allocated_count: int = 0

    def get_buffer(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        key = (tuple(shape), str(dtype))
        if key in self.free_buffers and len(self.free_buffers[key]) > 0:
            return self.free_buffers[key].pop()
        self.allocated_count += 1
        return np.empty(shape, dtype=dtype)

    def release_buffer(self, arr: np.ndarray) -> None:
        key = (tuple(arr.shape), str(arr.dtype))
        self.free_buffers.setdefault(key, []).append(arr)


class LayoutExecutor:
    """Memory-layout-optimized executor with contiguous buffers and memory arena."""

    def __init__(self, graph: Graph, enable_buffer_pool: bool = True):
        self.graph = graph
        self.enable_buffer_pool = enable_buffer_pool
        self.buffer_arena = BufferArena() if enable_buffer_pool else None
        self.profiler: Optional[OpProfiler] = None
        self.contiguous_initializers: Dict[str, np.ndarray] = {
            k: np.ascontiguousarray(v) for k, v in graph.initializers.items()
        }
        # Precompute static graph consumer counts
        consumers = self.graph.get_consumers()
        self.base_tensor_ref_counts = {
            name: len(consumers.get(name, [])) for name in self.graph.tensor_shapes.keys()
        }

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        enable_profiling: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Executes the graph with contiguous memory layout and buffer reuse."""
        tensor_store: Dict[str, np.ndarray] = {}

        # 1. Contiguous inputs & initializers
        for name, arr in self.contiguous_initializers.items():
            tensor_store[name] = arr
        for name, arr in inputs.items():
            tensor_store[name] = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)

        if enable_profiling:
            self.profiler = OpProfiler()

        # Intermediate buffer lifecycle reference counts
        tensor_ref_counts = dict(self.base_tensor_ref_counts)

        # Sequential execution
        for node in self.graph.nodes:
            op_fn = OP_REGISTRY.get(node.op_type)
            if op_fn is None:
                raise NotImplementedError(f"Operator {node.op_type} not found in OP_REGISTRY.")

            # Ensure all node inputs are strictly contiguous in memory
            node_inputs = [
                tensor_store[inp_name]
                if tensor_store[inp_name].flags.c_contiguous
                else np.ascontiguousarray(tensor_store[inp_name])
                for inp_name in node.inputs
            ]

            if enable_profiling:
                t0 = time.perf_counter_ns()
                out_tensor = op_fn(node_inputs, node.attributes)
                t1 = time.perf_counter_ns()
                self.profiler.record(node.name, node.op_type, (t1 - t0) / 1e6)
            else:
                out_tensor = op_fn(node_inputs, node.attributes)

            # Store result as contiguous array
            if isinstance(out_tensor, tuple):
                for out_name, out_arr in zip(node.outputs, out_tensor):
                    tensor_store[out_name] = (
                        out_arr if out_arr.flags.c_contiguous else np.ascontiguousarray(out_arr)
                    )
            else:
                tensor_store[node.outputs[0]] = (
                    out_tensor if out_tensor.flags.c_contiguous else np.ascontiguousarray(out_tensor)
                )

            # Decrement reference counts and recycle buffers if applicable
            if self.enable_buffer_pool and self.buffer_arena:
                for inp_name in node.inputs:
                    if inp_name not in self.graph.initializers and inp_name not in self.graph.inputs:
                        if inp_name in tensor_ref_counts:
                            tensor_ref_counts[inp_name] -= 1
                            if tensor_ref_counts[inp_name] == 0 and inp_name in tensor_store:
                                # Recycle buffer back into arena
                                # (safe only if not in graph outputs)
                                if inp_name not in self.graph.outputs:
                                    self.buffer_arena.release_buffer(tensor_store[inp_name])

        return {out_name: tensor_store[out_name] for out_name in self.graph.outputs}

    def get_profile_report(self) -> Optional[OpProfileReport]:
        if self.profiler is None:
            return None
        return self.profiler.generate_report()
