"""Fused Graph Executor (Phase 2).

Implements custom pattern-matching graph-rewrite passes to fuse:
  1. MatMul + Add (bias) + ReLU / GELU -> FusedMatMulBiasAct
  2. Conv2d + BatchNorm2d + ReLU -> FusedConvBNReLU (with algebraic BN folding)
  3. LayerNorm -> FusedLayerNorm
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..graph.ir import Graph, Node
from ..graph.ops import OP_REGISTRY
from ..kernels.matmul_bias_relu_fused import matmul_bias_relu_fused
from ..kernels.conv_bn_relu_fused import fold_batchnorm_into_conv, conv_bn_relu_fused
from ..kernels.layernorm_fused import layernorm_fused
from ..profiling.flamegraph import OpProfiler, OpProfileReport


# Extend registry with fused operators
FUSED_OP_REGISTRY = dict(OP_REGISTRY)

def op_fused_matmul_bias_act(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    x = inputs[0]
    weight_t = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    act = attributes.get("activation", "relu")
    return matmul_bias_relu_fused(x, weight_t, bias, activation=act)

def op_fused_conv_bn_relu(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    x = inputs[0]
    w_fused = inputs[1]
    b_fused = inputs[2]
    stride = attributes.get("stride", (1, 1))
    padding = attributes.get("padding", (0, 0))
    apply_relu = attributes.get("apply_relu", True)
    return conv_bn_relu_fused(x, w_fused, b_fused, stride=stride, padding=padding, apply_relu=apply_relu)

def op_fused_layernorm(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    x = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    eps = attributes.get("eps", 1e-5)
    axis = attributes.get("axis", -1)
    return layernorm_fused(x, gamma, beta, eps=eps, axis=axis)

FUSED_OP_REGISTRY["FusedMatMulBiasAct"] = op_fused_matmul_bias_act
FUSED_OP_REGISTRY["FusedConvBNReLU"] = op_fused_conv_bn_relu
FUSED_OP_REGISTRY["FusedLayerNorm"] = op_fused_layernorm


class FusedExecutor:
    """Executes a graph rewritten with custom operator fusion passes."""

    def __init__(
        self,
        graph: Graph,
        fuse_matmul: bool = True,
        fuse_conv_bn: bool = True,
        fuse_layernorm: bool = True,
    ):
        self.original_graph = graph
        self.fuse_matmul = fuse_matmul
        self.fuse_conv_bn = fuse_conv_bn
        self.fuse_layernorm = fuse_layernorm
        self.fused_graph = self._optimize_graph(graph.clone())
        self.profiler: Optional[OpProfiler] = None

    def _optimize_graph(self, graph: Graph) -> Graph:
        """Applies graph rewriting passes sequentially."""
        if self.fuse_conv_bn:
            graph = self._fuse_conv_bn_relu_pass(graph)
        if self.fuse_matmul:
            graph = self._fuse_matmul_bias_act_pass(graph)
        if self.fuse_layernorm:
            graph = self._fuse_layernorm_pass(graph)
        graph.topological_sort()
        return graph

    def _fuse_conv_bn_relu_pass(self, graph: Graph) -> Graph:
        """Fuses Conv2d + BatchNorm2d + (optional) ReLU by folding parameters."""
        consumers = graph.get_consumers()
        producers = graph.get_producers()
        nodes_to_remove: Set[str] = set()
        new_nodes: List[Node] = []

        for node in graph.nodes:
            if node.name in nodes_to_remove:
                continue

            # Check if node is Conv2d
            if node.op_type == "Conv2d":
                conv_out = node.outputs[0]
                conv_consumers = consumers.get(conv_out, [])

                # Match Conv2d -> BatchNorm2d
                if len(conv_consumers) == 1 and conv_consumers[0].op_type == "BatchNorm2d":
                    bn_node = conv_consumers[0]
                    bn_out = bn_node.outputs[0]
                    bn_consumers = consumers.get(bn_out, [])

                    # Check for subsequent ReLU
                    has_relu = len(bn_consumers) == 1 and bn_consumers[0].op_type == "ReLU"
                    relu_node = bn_consumers[0] if has_relu else None

                    # Extract weights and fold
                    conv_w = graph.initializers[node.inputs[1]]
                    conv_b = graph.initializers.get(node.inputs[2]) if len(node.inputs) > 2 else None

                    bn_gamma = graph.initializers[bn_node.inputs[1]]
                    bn_beta = graph.initializers[bn_node.inputs[2]]
                    bn_mean = graph.initializers[bn_node.inputs[3]]
                    bn_var = graph.initializers[bn_node.inputs[4]]
                    bn_eps = bn_node.attributes.get("eps", 1e-5)

                    w_fused, b_fused = fold_batchnorm_into_conv(
                        conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps
                    )

                    fused_w_name = f"{node.name}.fused_w"
                    fused_b_name = f"{node.name}.fused_b"
                    graph.add_initializer(fused_w_name, w_fused)
                    graph.add_initializer(fused_b_name, b_fused)

                    final_out = relu_node.outputs[0] if has_relu else bn_out
                    fused_node = Node(
                        name=f"{node.name}_fused_bn_relu" if has_relu else f"{node.name}_fused_bn",
                        op_type="FusedConvBNReLU",
                        inputs=[node.inputs[0], fused_w_name, fused_b_name],
                        outputs=[final_out],
                        attributes={
                            "stride": node.attributes.get("stride", (1, 1)),
                            "padding": node.attributes.get("padding", (0, 0)),
                            "apply_relu": has_relu,
                        },
                    )

                    new_nodes.append(fused_node)
                    nodes_to_remove.add(node.name)
                    nodes_to_remove.add(bn_node.name)
                    if has_relu and relu_node:
                        nodes_to_remove.add(relu_node.name)
                    continue

            new_nodes.append(node)

        graph.nodes = [n for n in new_nodes if n.name not in nodes_to_remove]
        return graph

    def _fuse_matmul_bias_act_pass(self, graph: Graph) -> Graph:
        """Fuses MatMul + Add(bias) + (optional) ReLU/GELU."""
        consumers = graph.get_consumers()
        nodes_to_remove: Set[str] = set()
        new_nodes: List[Node] = []

        for node in graph.nodes:
            if node.name in nodes_to_remove:
                continue

            if node.op_type == "MatMul":
                mm_out = node.outputs[0]
                mm_consumers = consumers.get(mm_out, [])

                # Match MatMul -> Add (bias)
                if len(mm_consumers) == 1 and mm_consumers[0].op_type == "Add":
                    add_node = mm_consumers[0]
                    # Check if second input to Add is an initializer (bias)
                    bias_name = None
                    if add_node.inputs[0] == mm_out and add_node.inputs[1] in graph.initializers:
                        bias_name = add_node.inputs[1]
                    elif add_node.inputs[1] == mm_out and add_node.inputs[0] in graph.initializers:
                        bias_name = add_node.inputs[0]

                    if bias_name is not None:
                        add_out = add_node.outputs[0]
                        add_consumers = consumers.get(add_out, [])

                        # Match optional activation (ReLU or GELU)
                        has_act = len(add_consumers) == 1 and add_consumers[0].op_type in ("ReLU", "GELU")
                        act_node = add_consumers[0] if has_act else None
                        act_type = act_node.op_type.lower() if has_act and act_node else "none"

                        final_out = act_node.outputs[0] if (has_act and act_node) else add_out
                        fused_inputs = [node.inputs[0], node.inputs[1], bias_name]

                        fused_node = Node(
                            name=f"{node.name}_fused_bias_{act_type}",
                            op_type="FusedMatMulBiasAct",
                            inputs=fused_inputs,
                            outputs=[final_out],
                            attributes={"activation": act_type},
                        )

                        new_nodes.append(fused_node)
                        nodes_to_remove.add(node.name)
                        nodes_to_remove.add(add_node.name)
                        if has_act and act_node:
                            nodes_to_remove.add(act_node.name)
                        continue

            new_nodes.append(node)

        graph.nodes = [n for n in new_nodes if n.name not in nodes_to_remove]
        return graph

    def _fuse_layernorm_pass(self, graph: Graph) -> Graph:
        """Upgrades LayerNorm nodes to use single-pass fused kernel."""
        for node in graph.nodes:
            if node.op_type == "LayerNorm":
                node.op_type = "FusedLayerNorm"
        return graph

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        enable_profiling: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Executes the fused computation graph."""
        tensor_store: Dict[str, np.ndarray] = {}

        for name, arr in self.fused_graph.initializers.items():
            tensor_store[name] = arr
        for name, arr in inputs.items():
            tensor_store[name] = arr

        if enable_profiling:
            self.profiler = OpProfiler()

        for node in self.fused_graph.nodes:
            op_fn = FUSED_OP_REGISTRY.get(node.op_type)
            if op_fn is None:
                raise NotImplementedError(f"Operator '{node.op_type}' not in FUSED_OP_REGISTRY.")

            node_inputs = [tensor_store[inp_name] for inp_name in node.inputs]

            if enable_profiling:
                t0 = time.perf_counter_ns()
                out_tensor = op_fn(node_inputs, node.attributes)
                t1 = time.perf_counter_ns()
                self.profiler.record(node.name, node.op_type, (t1 - t0) / 1e6)
            else:
                out_tensor = op_fn(node_inputs, node.attributes)

            if isinstance(out_tensor, tuple):
                for out_name, out_arr in zip(node.outputs, out_tensor):
                    tensor_store[out_name] = out_arr
            else:
                tensor_store[node.outputs[0]] = out_tensor

        return {out_name: tensor_store[out_name] for out_name in self.fused_graph.outputs}

    def get_profile_report(self) -> Optional[OpProfileReport]:
        if self.profiler is None:
            return None
        return self.profiler.generate_report()
