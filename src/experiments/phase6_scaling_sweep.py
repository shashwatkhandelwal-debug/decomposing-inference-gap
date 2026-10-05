"""Phase 6 Experiment: Multi-Dimensional Scaling Sweep.

Sweeps transformer depth (1, 2, 4, 8), hidden dimension (64, 128, 256),
and batch size (1, 8, 32, 128) to trace shifts between memory-bound and compute-bound regimes.
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
from ..executors.layout_executor import LayoutExecutor
from ..executors.quantized_executor import QuantizedExecutor
from ..executors.combined_executor import CombinedExecutor
from ..models.tiny_transformer import TinyTransformer
from ..models.export_onnx import export_to_onnx
from ..baselines.onnxruntime_baseline import ONNXRuntimeBaseline
from ..profiling.timers import BenchmarkHarness


def run_phase6_experiment(
    output_json_path: str = "results/raw_logs/phase6_scaling.json",
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Runs scaling sweeps across depth, hidden dim, and batch size."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    harness = BenchmarkHarness(warmup_runs=15, num_trials=40, num_threads=num_threads)

    results: Dict[str, Any] = {
        "phase": 6,
        "description": "Scaling Sweeps: Depth, Hidden Dim, and Batch Size",
        "depth_sweep": [],
        "dim_sweep": [],
        "batch_size_sweep": [],
    }

    # -------------------------------------------------------------
    # 1. Depth Sweep (Layers: 1, 2, 4, 8; Fixed D=64, B=1, S=16)
    # -------------------------------------------------------------
    print("[Phase 6] Running Depth Sweep...")
    for depth in [1, 2, 4, 8]:
        model = TinyTransformer(num_layers=depth, d_model=64, n_heads=4, d_ff=128)
        inp_tensor = torch.randn(1, 16, 64)
        inp_np = {"input_0": inp_tensor.numpy()}
        graph = build_from_pytorch(model, inp_tensor)

        onnx_p = f"results/models/trans_depth_{depth}.onnx"
        export_to_onnx(model, inp_tensor, onnx_p)
        ort_base = ONNXRuntimeBaseline(onnx_p, harness, num_threads=num_threads)

        naive_ex = NaiveExecutor(graph)
        fused_ex = FusedExecutor(graph)
        layout_ex = LayoutExecutor(graph)
        quant_ex = QuantizedExecutor(graph)
        comb_ex = CombinedExecutor(graph)

        n_res = harness.benchmark(lambda: naive_ex.run(inp_np))
        f_res = harness.benchmark(lambda: fused_ex.run(inp_np))
        l_res = harness.benchmark(lambda: layout_ex.run(inp_np))
        q_res = harness.benchmark(lambda: quant_ex.run(inp_np))
        c_res = harness.benchmark(lambda: comb_ex.run(inp_np))
        ort_res, _ = ort_base.run_benchmark(inp_tensor.numpy())

        tot_gap = max(n_res.median_ms - ort_res.median_ms, 1e-9)

        results["depth_sweep"].append({
            "num_layers": depth,
            "naive_ms": n_res.median_ms,
            "fused_ms": f_res.median_ms,
            "layout_ms": l_res.median_ms,
            "quantized_ms": q_res.median_ms,
            "combined_ms": c_res.median_ms,
            "onnxruntime_ms": ort_res.median_ms,
            "pct_gap_fused": ((n_res.median_ms - f_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_layout": ((n_res.median_ms - l_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_quant": ((n_res.median_ms - q_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_combined": ((n_res.median_ms - c_res.median_ms) / tot_gap) * 100.0,
        })

    # -------------------------------------------------------------
    # 2. Hidden Dim Sweep (Dims: 64, 128, 256; Fixed L=2, B=1, S=16)
    # -------------------------------------------------------------
    print("[Phase 6] Running Hidden Dim Sweep...")
    for dim in [64, 128, 256]:
        model = TinyTransformer(num_layers=2, d_model=dim, n_heads=4, d_ff=dim * 2)
        inp_tensor = torch.randn(1, 16, dim)
        inp_np = {"input_0": inp_tensor.numpy()}
        graph = build_from_pytorch(model, inp_tensor)

        onnx_p = f"results/models/trans_dim_{dim}.onnx"
        export_to_onnx(model, inp_tensor, onnx_p)
        ort_base = ONNXRuntimeBaseline(onnx_p, harness, num_threads=num_threads)

        naive_ex = NaiveExecutor(graph)
        fused_ex = FusedExecutor(graph)
        layout_ex = LayoutExecutor(graph)
        quant_ex = QuantizedExecutor(graph)
        comb_ex = CombinedExecutor(graph)

        n_res = harness.benchmark(lambda: naive_ex.run(inp_np))
        f_res = harness.benchmark(lambda: fused_ex.run(inp_np))
        l_res = harness.benchmark(lambda: layout_ex.run(inp_np))
        q_res = harness.benchmark(lambda: quant_ex.run(inp_np))
        c_res = harness.benchmark(lambda: comb_ex.run(inp_np))
        ort_res, _ = ort_base.run_benchmark(inp_tensor.numpy())

        tot_gap = max(n_res.median_ms - ort_res.median_ms, 1e-9)

        results["dim_sweep"].append({
            "d_model": dim,
            "naive_ms": n_res.median_ms,
            "fused_ms": f_res.median_ms,
            "layout_ms": l_res.median_ms,
            "quantized_ms": q_res.median_ms,
            "combined_ms": c_res.median_ms,
            "onnxruntime_ms": ort_res.median_ms,
            "pct_gap_fused": ((n_res.median_ms - f_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_layout": ((n_res.median_ms - l_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_quant": ((n_res.median_ms - q_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_combined": ((n_res.median_ms - c_res.median_ms) / tot_gap) * 100.0,
        })

    # -------------------------------------------------------------
    # 3. Batch Size Sweep (Batch: 1, 8, 32; Fixed L=2, D=64, S=16)
    # -------------------------------------------------------------
    print("[Phase 6] Running Batch Size Sweep...")
    model = TinyTransformer(num_layers=2, d_model=64, n_heads=4, d_ff=128)
    for bs in [1, 8, 32]:
        inp_tensor = torch.randn(bs, 16, 64)
        inp_np = {"input_0": inp_tensor.numpy()}
        graph = build_from_pytorch(model, inp_tensor)

        onnx_p = f"results/models/trans_bs_{bs}.onnx"
        export_to_onnx(model, inp_tensor, onnx_p)
        ort_base = ONNXRuntimeBaseline(onnx_p, harness, num_threads=num_threads)

        naive_ex = NaiveExecutor(graph)
        fused_ex = FusedExecutor(graph)
        layout_ex = LayoutExecutor(graph)
        quant_ex = QuantizedExecutor(graph)
        comb_ex = CombinedExecutor(graph)

        n_res = harness.benchmark(lambda: naive_ex.run(inp_np))
        f_res = harness.benchmark(lambda: fused_ex.run(inp_np))
        l_res = harness.benchmark(lambda: layout_ex.run(inp_np))
        q_res = harness.benchmark(lambda: quant_ex.run(inp_np))
        c_res = harness.benchmark(lambda: comb_ex.run(inp_np))
        ort_res, _ = ort_base.run_benchmark(inp_tensor.numpy())

        tot_gap = max(n_res.median_ms - ort_res.median_ms, 1e-9)

        results["batch_size_sweep"].append({
            "batch_size": bs,
            "naive_ms": n_res.median_ms,
            "fused_ms": f_res.median_ms,
            "layout_ms": l_res.median_ms,
            "quantized_ms": q_res.median_ms,
            "combined_ms": c_res.median_ms,
            "onnxruntime_ms": ort_res.median_ms,
            "pct_gap_fused": ((n_res.median_ms - f_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_layout": ((n_res.median_ms - l_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_quant": ((n_res.median_ms - q_res.median_ms) / tot_gap) * 100.0,
            "pct_gap_combined": ((n_res.median_ms - c_res.median_ms) / tot_gap) * 100.0,
        })

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 6] Scaling sweep complete -> Saved to {output_json_path}")
    return results


if __name__ == "__main__":
    run_phase6_experiment()
