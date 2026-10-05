"""Phase 2 Experiment: Operator Fusion Ablation.

Measures the speedup of each fusion pattern in isolation and combined,
and tracks fusion speedup scaling vs transformer layer depth.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import numpy as np
import torch

from ..graph.builder import build_from_pytorch
from ..executors.naive_executor import NaiveExecutor
from ..executors.fused_executor import FusedExecutor
from ..models.tiny_cnn import TinyCNN
from ..models.tiny_transformer import TinyTransformer
from ..profiling.timers import BenchmarkHarness


def run_phase2_experiment(
    output_json_path: str = "results/raw_logs/phase2_fusion.json",
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Runs Phase 2 fusion ablation experiments."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    harness = BenchmarkHarness(warmup_runs=25, num_trials=100, num_threads=num_threads)

    results: Dict[str, Any] = {
        "phase": 2,
        "description": "Operator Fusion Ablation and Depth Scaling",
        "patterns_isolated": {},
        "models_combined": {},
        "depth_scaling_sweep": [],
    }

    # -------------------------------------------------------------
    # 1. Isolated Fusion Patterns Benchmark
    # -------------------------------------------------------------
    # TinyCNN: Conv+BN+ReLU fusion in isolation
    cnn_model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=32)
    cnn_input_tensor = torch.randn(1, 3, 32, 32)
    cnn_input_np = {"input_0": cnn_input_tensor.numpy()}
    cnn_graph = build_from_pytorch(cnn_model, cnn_input_tensor)

    cnn_naive = NaiveExecutor(cnn_graph)
    cnn_fused_bn = FusedExecutor(cnn_graph, fuse_matmul=False, fuse_conv_bn=True, fuse_layernorm=False)
    cnn_fused_all = FusedExecutor(cnn_graph, fuse_matmul=True, fuse_conv_bn=True, fuse_layernorm=True)

    res_cnn_naive = harness.benchmark(lambda: cnn_naive.run(cnn_input_np))
    res_cnn_fused_bn = harness.benchmark(lambda: cnn_fused_bn.run(cnn_input_np))
    res_cnn_fused_all = harness.benchmark(lambda: cnn_fused_all.run(cnn_input_np))

    results["patterns_isolated"]["conv_bn_relu"] = {
        "naive_p50_ms": res_cnn_naive.median_ms,
        "fused_p50_ms": res_cnn_fused_bn.median_ms,
        "speedup": res_cnn_naive.median_ms / max(res_cnn_fused_bn.median_ms, 1e-9),
    }

    results["models_combined"]["tiny_cnn"] = {
        "naive_p50_ms": res_cnn_naive.median_ms,
        "fused_p50_ms": res_cnn_fused_all.median_ms,
        "speedup": res_cnn_naive.median_ms / max(res_cnn_fused_all.median_ms, 1e-9),
        "naive_stats": res_cnn_naive.to_dict(),
        "fused_stats": res_cnn_fused_all.to_dict(),
    }

    # TinyTransformer: MatMul+Bias+Act & LayerNorm in isolation
    trans_model = TinyTransformer(num_layers=2, d_model=64, n_heads=4, d_ff=128)
    trans_input_tensor = torch.randn(1, 16, 64)
    trans_input_np = {"input_0": trans_input_tensor.numpy()}
    trans_graph = build_from_pytorch(trans_model, trans_input_tensor)

    trans_naive = NaiveExecutor(trans_graph)
    trans_fused_mm = FusedExecutor(trans_graph, fuse_matmul=True, fuse_conv_bn=False, fuse_layernorm=False)
    trans_fused_ln = FusedExecutor(trans_graph, fuse_matmul=False, fuse_conv_bn=False, fuse_layernorm=True)
    trans_fused_all = FusedExecutor(trans_graph, fuse_matmul=True, fuse_conv_bn=True, fuse_layernorm=True)

    res_trans_naive = harness.benchmark(lambda: trans_naive.run(trans_input_np))
    res_trans_fused_mm = harness.benchmark(lambda: trans_fused_mm.run(trans_input_np))
    res_trans_fused_ln = harness.benchmark(lambda: trans_fused_ln.run(trans_input_np))
    res_trans_fused_all = harness.benchmark(lambda: trans_fused_all.run(trans_input_np))

    results["patterns_isolated"]["matmul_bias_act"] = {
        "naive_p50_ms": res_trans_naive.median_ms,
        "fused_p50_ms": res_trans_fused_mm.median_ms,
        "speedup": res_trans_naive.median_ms / max(res_trans_fused_mm.median_ms, 1e-9),
    }

    results["patterns_isolated"]["layernorm_fused"] = {
        "naive_p50_ms": res_trans_naive.median_ms,
        "fused_p50_ms": res_trans_fused_ln.median_ms,
        "speedup": res_trans_naive.median_ms / max(res_trans_fused_ln.median_ms, 1e-9),
    }

    results["models_combined"]["tiny_transformer"] = {
        "naive_p50_ms": res_trans_naive.median_ms,
        "fused_p50_ms": res_trans_fused_all.median_ms,
        "speedup": res_trans_naive.median_ms / max(res_trans_fused_all.median_ms, 1e-9),
        "naive_stats": res_trans_naive.to_dict(),
        "fused_stats": res_trans_fused_all.to_dict(),
    }

    # -------------------------------------------------------------
    # 2. Depth Scaling Sweep (Transformer depth: 1, 2, 4, 8)
    # -------------------------------------------------------------
    for depth in [1, 2, 4, 8]:
        d_model_inst = TinyTransformer(num_layers=depth, d_model=64, n_heads=4, d_ff=128)
        d_input_tensor = torch.randn(1, 16, 64)
        d_input_np = {"input_0": d_input_tensor.numpy()}
        d_graph = build_from_pytorch(d_model_inst, d_input_tensor)

        d_naive = NaiveExecutor(d_graph)
        d_fused = FusedExecutor(d_graph)

        res_n = harness.benchmark(lambda: d_naive.run(d_input_np), custom_trials=50)
        res_f = harness.benchmark(lambda: d_fused.run(d_input_np), custom_trials=50)

        speedup = res_n.median_ms / max(res_f.median_ms, 1e-9)
        pct_gain = (1.0 - res_f.median_ms / res_n.median_ms) * 100.0

        results["depth_scaling_sweep"].append({
            "num_layers": depth,
            "naive_p50_ms": res_n.median_ms,
            "fused_p50_ms": res_f.median_ms,
            "speedup": speedup,
            "percent_gain": pct_gain,
        })

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 2] Fusion ablation complete -> Saved to {output_json_path}")
    return results


if __name__ == "__main__":
    run_phase2_experiment()
