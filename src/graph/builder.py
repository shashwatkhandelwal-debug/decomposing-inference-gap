"""Manual Graph Builder for PyTorch models.

Constructs computation graph IR by inspecting model architecture, hooks, and parameter tensors
without using torch.fx, torch.jit, or black-box tracing frameworks.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .ir import Graph, Node
from ..models.tiny_transformer import TinyTransformer
from ..models.tiny_cnn import TinyCNN


def build_cnn_graph(model: TinyCNN, input_shape: Tuple[int, ...]) -> Graph:
    """Manually constructs a Graph IR from a TinyCNN model."""
    model.eval()
    g = Graph(name="tiny_cnn", inputs=["input_0"], outputs=["logits"])
    g.set_tensor_meta("input_0", input_shape, "float32")

    curr_tensor = "input_0"
    conv_blocks = [
        (model.conv1, model.bn1, "layer1"),
        (model.conv2, model.bn2, "layer2"),
        (model.conv3, model.bn3, "layer3"),
        (model.conv4, model.bn4, "layer4"),
    ]

    for conv, bn, name in conv_blocks:
        # Conv2d weights
        w_name = f"{name}.conv.weight"
        g.add_initializer(w_name, conv.weight.detach().cpu().numpy())
        conv_inputs = [curr_tensor, w_name]
        if conv.bias is not None:
            b_name = f"{name}.conv.bias"
            g.add_initializer(b_name, conv.bias.detach().cpu().numpy())
            conv_inputs.append(b_name)

        conv_out = f"{name}.conv_out"
        g.add_node(
            Node(
                name=f"{name}_conv",
                op_type="Conv2d",
                inputs=conv_inputs,
                outputs=[conv_out],
                attributes={
                    "stride": conv.stride,
                    "padding": conv.padding,
                },
            )
        )

        # BatchNorm2d parameters
        gamma_name = f"{name}.bn.gamma"
        beta_name = f"{name}.bn.beta"
        mean_name = f"{name}.bn.mean"
        var_name = f"{name}.bn.var"
        g.add_initializer(gamma_name, bn.weight.detach().cpu().numpy())
        g.add_initializer(beta_name, bn.bias.detach().cpu().numpy())
        g.add_initializer(mean_name, bn.running_mean.detach().cpu().numpy())
        g.add_initializer(var_name, bn.running_var.detach().cpu().numpy())

        bn_out = f"{name}.bn_out"
        g.add_node(
            Node(
                name=f"{name}_bn",
                op_type="BatchNorm2d",
                inputs=[conv_out, gamma_name, beta_name, mean_name, var_name],
                outputs=[bn_out],
                attributes={"eps": bn.eps},
            )
        )

        # ReLU
        relu_out = f"{name}.relu_out"
        g.add_node(
            Node(
                name=f"{name}_relu",
                op_type="ReLU",
                inputs=[bn_out],
                outputs=[relu_out],
            )
        )
        curr_tensor = relu_out

    # Global Average Pooling modeled as AdaptiveAvgPool2d / spatial mean
    # For inference on (N, C, H, W) where adaptive avg pool is (1, 1),
    # we can express this via a Reshape/Flatten or standard pooling node.
    pool_out = "pool_out"
    g.add_node(
        Node(
            name="adaptive_avg_pool",
            op_type="ReduceMean",
            inputs=[curr_tensor],
            outputs=[pool_out],
            attributes={"axis": (2, 3), "keepdims": True},
        )
    )

    # Flatten
    flat_out = "flat_out"
    g.add_node(
        Node(
            name="flatten",
            op_type="Flatten",
            inputs=[pool_out],
            outputs=[flat_out],
            attributes={"start_dim": 1, "end_dim": -1},
        )
    )

    # Linear Classifier: out = flat @ weight.T + bias
    fc_w = model.fc.weight.detach().cpu().numpy().T
    fc_b = model.fc.bias.detach().cpu().numpy()
    g.add_initializer("fc.weight_t", fc_w)
    g.add_initializer("fc.bias", fc_b)

    fc_matmul_out = "fc.matmul_out"
    g.add_node(
        Node(
            name="fc_matmul",
            op_type="MatMul",
            inputs=[flat_out, "fc.weight_t"],
            outputs=[fc_matmul_out],
        )
    )

    g.add_node(
        Node(
            name="fc_add",
            op_type="Add",
            inputs=[fc_matmul_out, "fc.bias"],
            outputs=["logits"],
        )
    )

    g.topological_sort()
    return g


def build_transformer_graph(model: TinyTransformer, input_shape: Tuple[int, ...]) -> Graph:
    """Manually constructs a Graph IR from a TinyTransformer model."""
    model.eval()
    g = Graph(name="tiny_transformer", inputs=["input_0"], outputs=["logits"])
    g.set_tensor_meta("input_0", input_shape, "float32")

    B, S, D = input_shape
    curr_x = "input_0"

    for l_idx, block in enumerate(model.blocks):
        prefix = f"block_{l_idx}"
        
        # 1. LayerNorm 1
        ln1_w = block.ln1.weight.detach().cpu().numpy()
        ln1_b = block.ln1.bias.detach().cpu().numpy()
        g.add_initializer(f"{prefix}.ln1.gamma", ln1_w)
        g.add_initializer(f"{prefix}.ln1.beta", ln1_b)
        
        ln1_out = f"{prefix}.ln1_out"
        g.add_node(
            Node(
                name=f"{prefix}_ln1",
                op_type="LayerNorm",
                inputs=[curr_x, f"{prefix}.ln1.gamma", f"{prefix}.ln1.beta"],
                outputs=[ln1_out],
                attributes={"eps": block.ln1.eps, "axis": -1},
            )
        )

        # 2. Q, K, V Projections
        for proj_name, proj_mod in [("q", block.attn.q_proj), ("k", block.attn.k_proj), ("v", block.attn.v_proj)]:
            w_t = proj_mod.weight.detach().cpu().numpy().T
            b = proj_mod.bias.detach().cpu().numpy()
            g.add_initializer(f"{prefix}.{proj_name}.weight_t", w_t)
            g.add_initializer(f"{prefix}.{proj_name}.bias", b)

            mm_out = f"{prefix}.{proj_name}.mm_out"
            g.add_node(
                Node(
                    name=f"{prefix}_{proj_name}_mm",
                    op_type="MatMul",
                    inputs=[ln1_out, f"{prefix}.{proj_name}.weight_t"],
                    outputs=[mm_out],
                )
            )

            proj_out = f"{prefix}.{proj_name}_out"
            g.add_node(
                Node(
                    name=f"{prefix}_{proj_name}_add",
                    op_type="Add",
                    inputs=[mm_out, f"{prefix}.{proj_name}.bias"],
                    outputs=[proj_out],
                )
            )

        # 3. Multi-Head Attention Structure
        # Reshape Q, K, V -> (B, S, n_heads, d_k) -> Transpose to (B, n_heads, S, d_k)
        n_heads = block.attn.n_heads
        d_k = block.attn.d_k

        for proj_name in ["q", "k", "v"]:
            reshaped = f"{prefix}.{proj_name}_reshaped"
            g.add_node(
                Node(
                    name=f"{prefix}_{proj_name}_reshape",
                    op_type="Reshape",
                    inputs=[f"{prefix}.{proj_name}_out"],
                    outputs=[reshaped],
                    attributes={"shape": (B, S, n_heads, d_k)},
                )
            )
            transposed = f"{prefix}.{proj_name}_heads"
            g.add_node(
                Node(
                    name=f"{prefix}_{proj_name}_transpose",
                    op_type="Transpose",
                    inputs=[reshaped],
                    outputs=[transposed],
                    attributes={"axes": (0, 2, 1, 3)},  # (B, H, S, d_k)
                )
            )

        # Transpose K for dot product: (B, H, d_k, S)
        k_t = f"{prefix}.k_t"
        g.add_node(
            Node(
                name=f"{prefix}_k_transpose_last2",
                op_type="Transpose",
                inputs=[f"{prefix}.k_heads"],
                outputs=[k_t],
                attributes={"axes": (0, 1, 3, 2)},
            )
        )

        # Q @ K.T -> (B, H, S, S)
        scores_raw = f"{prefix}.scores_raw"
        g.add_node(
            Node(
                name=f"{prefix}_attn_scores_mm",
                op_type="MatMul",
                inputs=[f"{prefix}.q_heads", k_t],
                outputs=[scores_raw],
            )
        )

        # Softmax over last dimension with scaling (1 / sqrt(d_k))
        attn_weights = f"{prefix}.attn_weights"
        g.add_node(
            Node(
                name=f"{prefix}_softmax",
                op_type="Softmax",
                inputs=[scores_raw],
                outputs=[attn_weights],
                attributes={"axis": -1, "scale": 1.0 / math.sqrt(d_k)},
            )
        )

        # attn_weights @ V -> (B, H, S, d_k)
        context_heads = f"{prefix}.context_heads"
        g.add_node(
            Node(
                name=f"{prefix}_context_mm",
                op_type="MatMul",
                inputs=[attn_weights, f"{prefix}.v_heads"],
                outputs=[context_heads],
            )
        )

        # Transpose context back: (B, S, H, d_k) -> Reshape to (B, S, D)
        context_transposed = f"{prefix}.context_transposed"
        g.add_node(
            Node(
                name=f"{prefix}_context_transpose",
                op_type="Transpose",
                inputs=[context_heads],
                outputs=[context_transposed],
                attributes={"axes": (0, 2, 1, 3)},
            )
        )

        context_merged = f"{prefix}.context_merged"
        g.add_node(
            Node(
                name=f"{prefix}_context_reshape",
                op_type="Reshape",
                inputs=[context_transposed],
                outputs=[context_merged],
                attributes={"shape": (B, S, D)},
            )
        )

        # Out projection
        out_w_t = block.attn.out_proj.weight.detach().cpu().numpy().T
        out_b = block.attn.out_proj.bias.detach().cpu().numpy()
        g.add_initializer(f"{prefix}.out_proj.weight_t", out_w_t)
        g.add_initializer(f"{prefix}.out_proj.bias", out_b)

        out_mm = f"{prefix}.out_proj.mm_out"
        g.add_node(
            Node(
                name=f"{prefix}_out_proj_mm",
                op_type="MatMul",
                inputs=[context_merged, f"{prefix}.out_proj.weight_t"],
                outputs=[out_mm],
            )
        )

        attn_out = f"{prefix}.attn_out"
        g.add_node(
            Node(
                name=f"{prefix}_out_proj_add",
                op_type="Add",
                inputs=[out_mm, f"{prefix}.out_proj.bias"],
                outputs=[attn_out],
            )
        )

        # Residual connection 1: curr_x + attn_out
        res1_out = f"{prefix}.res1_out"
        g.add_node(
            Node(
                name=f"{prefix}_residual1",
                op_type="Add",
                inputs=[curr_x, attn_out],
                outputs=[res1_out],
            )
        )

        # 4. LayerNorm 2
        ln2_w = block.ln2.weight.detach().cpu().numpy()
        ln2_b = block.ln2.bias.detach().cpu().numpy()
        g.add_initializer(f"{prefix}.ln2.gamma", ln2_w)
        g.add_initializer(f"{prefix}.ln2.beta", ln2_b)

        ln2_out = f"{prefix}.ln2_out"
        g.add_node(
            Node(
                name=f"{prefix}_ln2",
                op_type="LayerNorm",
                inputs=[res1_out, f"{prefix}.ln2.gamma", f"{prefix}.ln2.beta"],
                outputs=[ln2_out],
                attributes={"eps": block.ln2.eps, "axis": -1},
            )
        )

        # 5. FeedForward Network
        ffn1_w_t = block.ffn.linear1.weight.detach().cpu().numpy().T
        ffn1_b = block.ffn.linear1.bias.detach().cpu().numpy()
        g.add_initializer(f"{prefix}.ffn1.weight_t", ffn1_w_t)
        g.add_initializer(f"{prefix}.ffn1.bias", ffn1_b)

        ffn1_mm = f"{prefix}.ffn1.mm_out"
        g.add_node(
            Node(
                name=f"{prefix}_ffn1_mm",
                op_type="MatMul",
                inputs=[ln2_out, f"{prefix}.ffn1.weight_t"],
                outputs=[ffn1_mm],
            )
        )

        ffn1_add = f"{prefix}.ffn1.add_out"
        g.add_node(
            Node(
                name=f"{prefix}_ffn1_add",
                op_type="Add",
                inputs=[ffn1_mm, f"{prefix}.ffn1.bias"],
                outputs=[ffn1_add],
            )
        )

        ffn_act = f"{prefix}.ffn.act_out"
        act_op = "GELU" if block.ffn.activation == "gelu" else "ReLU"
        g.add_node(
            Node(
                name=f"{prefix}_ffn_act",
                op_type=act_op,
                inputs=[ffn1_add],
                outputs=[ffn_act],
            )
        )

        ffn2_w_t = block.ffn.linear2.weight.detach().cpu().numpy().T
        ffn2_b = block.ffn.linear2.bias.detach().cpu().numpy()
        g.add_initializer(f"{prefix}.ffn2.weight_t", ffn2_w_t)
        g.add_initializer(f"{prefix}.ffn2.bias", ffn2_b)

        ffn2_mm = f"{prefix}.ffn2.mm_out"
        g.add_node(
            Node(
                name=f"{prefix}_ffn2_mm",
                op_type="MatMul",
                inputs=[ffn_act, f"{prefix}.ffn2.weight_t"],
                outputs=[ffn2_mm],
            )
        )

        ffn2_add = f"{prefix}.ffn2.add_out"
        g.add_node(
            Node(
                name=f"{prefix}_ffn2_add",
                op_type="Add",
                inputs=[ffn2_mm, f"{prefix}.ffn2.bias"],
                outputs=[ffn2_add],
            )
        )

        # Residual connection 2: res1_out + ffn2_add
        res2_out = f"{prefix}.res2_out"
        g.add_node(
            Node(
                name=f"{prefix}_residual2",
                op_type="Add",
                inputs=[res1_out, f"{prefix}.ffn2.add_out"],
                outputs=[res2_out],
            )
        )

        curr_x = res2_out

    # Final LayerNorm
    final_ln_w = model.final_ln.weight.detach().cpu().numpy()
    final_ln_b = model.final_ln.bias.detach().cpu().numpy()
    g.add_initializer("final_ln.gamma", final_ln_w)
    g.add_initializer("final_ln.beta", final_ln_b)

    final_ln_out = "final_ln_out"
    g.add_node(
        Node(
            name="final_ln",
            op_type="LayerNorm",
            inputs=[curr_x, "final_ln.gamma", "final_ln.beta"],
            outputs=[final_ln_out],
            attributes={"eps": model.final_ln.eps, "axis": -1},
        )
    )

    # Pooling over sequence dimension (Mean reduction)
    pooled_out = "pooled_out"
    g.add_node(
        Node(
            name="seq_mean_pool",
            op_type="ReduceMean",
            inputs=[final_ln_out],
            outputs=[pooled_out],
            attributes={"axis": 1, "keepdims": False},
        )
    )

    # Classification Head
    head_w_t = model.head.weight.detach().cpu().numpy().T
    head_b = model.head.bias.detach().cpu().numpy()
    g.add_initializer("head.weight_t", head_w_t)
    g.add_initializer("head.bias", head_b)

    head_mm = "head.mm_out"
    g.add_node(
        Node(
            name="head_matmul",
            op_type="MatMul",
            inputs=[pooled_out, "head.weight_t"],
            outputs=[head_mm],
        )
    )

    g.add_node(
        Node(
            name="head_add",
            op_type="Add",
            inputs=[head_mm, "head.bias"],
            outputs=["logits"],
        )
    )

    g.topological_sort()
    return g


def build_from_pytorch(model: nn.Module, sample_input: torch.Tensor) -> Graph:
    """Entrypoint to build a Graph IR from a supported PyTorch model."""
    if isinstance(model, TinyCNN):
        return build_cnn_graph(model, tuple(sample_input.shape))
    elif isinstance(model, TinyTransformer):
        return build_transformer_graph(model, tuple(sample_input.shape))
    else:
        raise NotImplementedError(
            f"Manual builder not implemented for model class {model.__class__.__name__}. "
            "Graph construction is implemented from first principles for research clarity."
        )
