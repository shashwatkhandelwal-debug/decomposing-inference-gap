"""Naive Graph Executor (Phase 1).

Executes the computation graph op-by-op in topologically sorted order,
allocating fresh intermediate tensors for every node without reuse or fusion.
Instruments per-operator timings to identify execution bottlenecks.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..graph.ir import Graph, Node
from ..graph.ops import OP_REGISTRY
from ..profiling.flamegraph import OpProfiler, OpProfileReport


class NaiveExecutor:
    """Op-by-op eager-style reference graph executor."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.profiler: Optional[OpProfiler] = None

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        enable_profiling: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Executes the graph in topologically sorted order.

        Args:
            inputs: Dictionary mapping input tensor names to numpy arrays.
            enable_profiling: If True, records per-node latency.

        Returns:
            Dictionary mapping graph output names to output numpy arrays.
        """
        tensor_store: Dict[str, np.ndarray] = {}

        # 1. Populate initializers (weights/biases)
        for name, arr in self.graph.initializers.items():
            tensor_store[name] = arr

        # 2. Populate inputs
        for name, arr in inputs.items():
            tensor_store[name] = arr

        if enable_profiling:
            self.profiler = OpProfiler()

        # 3. Sequential op-by-op execution
        for node in self.graph.nodes:
            op_fn = OP_REGISTRY.get(node.op_type)
            if op_fn is None:
                raise NotImplementedError(
                    f"Operator '{node.op_type}' for node '{node.name}' is not in OP_REGISTRY."
                )

            # Retrieve inputs (allocating fresh list)
            node_inputs = [tensor_store[inp_name] for inp_name in node.inputs]

            if enable_profiling:
                t0 = time.perf_counter_ns()
                out_tensor = op_fn(node_inputs, node.attributes)
                t1 = time.perf_counter_ns()
                self.profiler.record(node.name, node.op_type, (t1 - t0) / 1e6)
            else:
                out_tensor = op_fn(node_inputs, node.attributes)

            # Store outputs (allocating fresh references in tensor_store)
            if isinstance(out_tensor, tuple):
                for out_name, out_arr in zip(node.outputs, out_tensor):
                    tensor_store[out_name] = np.array(out_arr, copy=True)
            else:
                out_name = node.outputs[0]
                tensor_store[out_name] = np.array(out_tensor, copy=True)

        # 4. Extract graph outputs
        return {out_name: tensor_store[out_name] for out_name in self.graph.outputs}

    def get_profile_report(self) -> Optional[OpProfileReport]:
        """Returns the operator profile report if profiling was enabled."""
        if self.profiler is None:
            return None
        return self.profiler.generate_report()
