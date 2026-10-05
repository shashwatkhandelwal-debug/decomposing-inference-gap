"""Phase 5 Experiment: Combined Optimizations vs Real Production Baselines.

Evaluates Naive, +Fusion, +Layout, +Quantization, and Combined Hand-Built Executors
against real ONNX Runtime and PyTorch eager baselines, generating the core Attribution Table.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from ..graph.builder import build_from_pytorch
from ..executors.naive_executor import NaiveExecutor
from ..executors.fused_executor import FusedExecutor
from ..executors.layout_executor import LayoutExecutor
from ..executors.quantized_executor import QuantizedExecutor
from ..executors.combined_executor import CombinedExecutor
from ..baselines.pytorch_eager import PyTorchEagerBaseline
from ..baselines.onnxruntime_baseline import ONNXRuntimeBaseline
from ..baselines.tensorrt_baseline import TensorRTBaseline
from ..models.tiny_cnn import TinyCNN
from ..models.tiny_transformer import TinyTransformer
from ..models.export_onnx import export_to_onnx
from ..profiling.timers import BenchmarkHarness


def calculate_attribution(
    naive_lat: float,
    opt_lat: float,
    ort_lat: float,
) -> Tuple[float, float, str]:
    """Calculates speedup vs naive and percentage of ONNX Runtime gap closed with descriptive label.

    Formula:
      speedup = naive_latency / opt_latency
      gap_closed_pct = (naive_latency - opt_latency) / (naive_latency - ort_latency) * 100

    Regression handling:
      If opt_latency > naive_latency (the optimization caused a slowdown),
      gap_closed_pct is negative. We provide both the raw percentage and a clear label.
    """
    speedup = naive_lat / max(opt_lat, 1e-9)
    total_gap = naive_lat - ort_lat
    if total_gap <= 1e-9:
        pct_gap = 100.0
        label = "100.0%"
    else:
        closed = naive_lat - opt_lat
        pct_gap = (closed / total_gap) * 100.0
        if opt_lat > naive_lat:
            label = f"REGRESSION ({pct_gap:.1f}%)"
        else:
            label = f"{pct_gap:.1f}%"
    return float(speedup), float(pct_gap), label


def run_phase5_experiment(
    output_json_path: str = "results/raw_logs/phase5_combined.json",
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Runs Phase 5 combined benchmark and populates the attribution table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    harness = BenchmarkHarness(warmup_runs=25, num_trials=100, num_threads=num_threads)

    results: Dict[str, Any] = {
        "phase": 5,
        "description": "Decomposing the Inference Speedup Gap: Attribution Table",
        "models": {},
    }

    # -------------------------------------------------------------
    # 1. Tiny Transformer
    # -------------------------------------------------------------
    print("[Phase 5] Benchmarking Tiny Transformer...")
    trans_model = TinyTransformer(num_layers=2, d_model=64, n_heads=4, d_ff=128)
    trans_input_tensor = torch.randn(1, 16, 64)
    trans_input_np = {"input_0": trans_input_tensor.numpy()}
    trans_graph = build_from_pytorch(trans_model, trans_input_tensor)

    # Export ONNX
    onnx_trans_path = "results/models/tiny_transformer.onnx"
    export_to_onnx(trans_model, trans_input_tensor, onnx_trans_path)

    # Instantiate executors
    trans_naive = NaiveExecutor(trans_graph)
    trans_fused = FusedExecutor(trans_graph)
    trans_layout = LayoutExecutor(trans_graph)
    trans_fused_layout = CombinedExecutor(trans_graph, enable_quantization=False)
    trans_quant = QuantizedExecutor(trans_graph)
    trans_combined = CombinedExecutor(trans_graph, enable_quantization=True)

    # Baselines
    trans_pt = PyTorchEagerBaseline(trans_model, harness)
    trans_ort = ONNXRuntimeBaseline(onnx_trans_path, harness, num_threads=num_threads)
    trans_trt = TensorRTBaseline(onnx_trans_path, harness)

    # Run benchmarks
    res_n = harness.benchmark(lambda: trans_naive.run(trans_input_np))
    res_f = harness.benchmark(lambda: trans_fused.run(trans_input_np))
    res_l = harness.benchmark(lambda: trans_layout.run(trans_input_np))
    res_fl = harness.benchmark(lambda: trans_fused_layout.run(trans_input_np))
    res_q = harness.benchmark(lambda: trans_quant.run(trans_input_np))
    res_c = harness.benchmark(lambda: trans_combined.run(trans_input_np))
    res_pt, _ = trans_pt.run_benchmark(trans_input_tensor)
    res_ort, _ = trans_ort.run_benchmark(trans_input_tensor.numpy())
    res_trt = trans_trt.run_benchmark(trans_input_tensor.numpy())

    n_lat = res_n.median_ms
    ort_lat = res_ort.median_ms

    sp_f, gap_f, lbl_f = calculate_attribution(n_lat, res_f.median_ms, ort_lat)
    sp_l, gap_l, lbl_l = calculate_attribution(n_lat, res_l.median_ms, ort_lat)
    sp_fl, gap_fl, lbl_fl = calculate_attribution(n_lat, res_fl.median_ms, ort_lat)
    sp_q, gap_q, lbl_q = calculate_attribution(n_lat, res_q.median_ms, ort_lat)
    sp_c, gap_c, lbl_c = calculate_attribution(n_lat, res_c.median_ms, ort_lat)
    sp_pt, gap_pt, lbl_pt = calculate_attribution(n_lat, res_pt.median_ms, ort_lat)

    trans_table = [
        {"config": "Naive (Phase 1)", "p50_ms": n_lat, "speedup": 1.00, "pct_ort_gap": 0.0, "gap_label": "0.0% (reference)"},
        {"config": "+ Fusion only (Phase 2)", "p50_ms": res_f.median_ms, "speedup": sp_f, "pct_ort_gap": gap_f, "gap_label": lbl_f},
        {"config": "+ Layout only (Phase 3)", "p50_ms": res_l.median_ms, "speedup": sp_l, "pct_ort_gap": gap_l, "gap_label": lbl_l},
        {"config": "+ Fusion + Layout (Quantization excluded)", "p50_ms": res_fl.median_ms, "speedup": sp_fl, "pct_ort_gap": gap_fl, "gap_label": lbl_fl},
        {"config": "+ Quantization only (Phase 4)", "p50_ms": res_q.median_ms, "speedup": sp_q, "pct_ort_gap": gap_q, "gap_label": lbl_q},
        {"config": "+ Combined incl. Quantization (not recommended — see Phase 4 negative result)", "p50_ms": res_c.median_ms, "speedup": sp_c, "pct_ort_gap": gap_c, "gap_label": lbl_c},
        {"config": "PyTorch Eager (Sanity Baseline)", "p50_ms": res_pt.median_ms, "speedup": sp_pt, "pct_ort_gap": gap_pt, "gap_label": lbl_pt},
        {"config": "Real ONNX Runtime (Target Baseline)", "p50_ms": ort_lat, "speedup": calculate_attribution(n_lat, ort_lat, ort_lat)[0], "pct_ort_gap": 100.0, "gap_label": "100.0% (reference)"},
        {"config": "Real TensorRT", "p50_ms": res_trt[0].median_ms if res_trt else None, "speedup": None, "pct_ort_gap": None, "gap_label": "Skipped (GPU-gated)"},
    ]

    results["models"]["tiny_transformer"] = {
        "attribution_table": trans_table,
        "raw_results": {
            "naive": res_n.to_dict(),
            "fused": res_f.to_dict(),
            "layout": res_l.to_dict(),
            "fused_layout": res_fl.to_dict(),
            "quantized": res_q.to_dict(),
            "combined": res_c.to_dict(),
            "pytorch_eager": res_pt.to_dict(),
            "onnxruntime": res_ort.to_dict(),
        },
    }

    # -------------------------------------------------------------
    # 2. Tiny CNN
    # -------------------------------------------------------------
    print("[Phase 5] Benchmarking Tiny CNN...")
    cnn_model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=32)
    cnn_input_tensor = torch.randn(1, 3, 32, 32)
    cnn_input_np = {"input_0": cnn_input_tensor.numpy()}
    cnn_graph = build_from_pytorch(cnn_model, cnn_input_tensor)

    onnx_cnn_path = "results/models/tiny_cnn.onnx"
    export_to_onnx(cnn_model, cnn_input_tensor, onnx_cnn_path)

    cnn_naive = NaiveExecutor(cnn_graph)
    cnn_fused = FusedExecutor(cnn_graph)
    cnn_layout = LayoutExecutor(cnn_graph)
    cnn_fused_layout = CombinedExecutor(cnn_graph, enable_quantization=False)
    cnn_quant = QuantizedExecutor(cnn_graph)
    cnn_combined = CombinedExecutor(cnn_graph, enable_quantization=True)

    cnn_pt = PyTorchEagerBaseline(cnn_model, harness)
    cnn_ort = ONNXRuntimeBaseline(onnx_cnn_path, harness, num_threads=num_threads)
    cnn_trt = TensorRTBaseline(onnx_cnn_path, harness)

    res_cnn_n = harness.benchmark(lambda: cnn_naive.run(cnn_input_np))
    res_cnn_f = harness.benchmark(lambda: cnn_fused.run(cnn_input_np))
    res_cnn_l = harness.benchmark(lambda: cnn_layout.run(cnn_input_np))
    res_cnn_fl = harness.benchmark(lambda: cnn_fused_layout.run(cnn_input_np))
    res_cnn_q = harness.benchmark(lambda: cnn_quant.run(cnn_input_np))
    res_cnn_c = harness.benchmark(lambda: cnn_combined.run(cnn_input_np))
    res_cnn_pt, _ = cnn_pt.run_benchmark(cnn_input_tensor)
    res_cnn_ort, _ = cnn_ort.run_benchmark(cnn_input_tensor.numpy())
    res_cnn_trt = cnn_trt.run_benchmark(cnn_input_tensor.numpy())

    n_cnn_lat = res_cnn_n.median_ms
    ort_cnn_lat = res_cnn_ort.median_ms

    sp_cf, gap_cf, lbl_cf = calculate_attribution(n_cnn_lat, res_cnn_f.median_ms, ort_cnn_lat)
    sp_cl, gap_cl, lbl_cl = calculate_attribution(n_cnn_lat, res_cnn_l.median_ms, ort_cnn_lat)
    sp_cfl, gap_cfl, lbl_cfl = calculate_attribution(n_cnn_lat, res_cnn_fl.median_ms, ort_cnn_lat)
    sp_cq, gap_cq, lbl_cq = calculate_attribution(n_cnn_lat, res_cnn_q.median_ms, ort_cnn_lat)
    sp_cc, gap_cc, lbl_cc = calculate_attribution(n_cnn_lat, res_cnn_c.median_ms, ort_cnn_lat)
    sp_cpt, gap_cpt, lbl_cpt = calculate_attribution(n_cnn_lat, res_cnn_pt.median_ms, ort_cnn_lat)

    cnn_table = [
        {"config": "Naive (Phase 1)", "p50_ms": n_cnn_lat, "speedup": 1.00, "pct_ort_gap": 0.0, "gap_label": "0.0% (reference)"},
        {"config": "+ Fusion only (Phase 2)", "p50_ms": res_cnn_f.median_ms, "speedup": sp_cf, "pct_ort_gap": gap_cf, "gap_label": lbl_cf},
        {"config": "+ Layout only (Phase 3)", "p50_ms": res_cnn_l.median_ms, "speedup": sp_cl, "pct_ort_gap": gap_cl, "gap_label": lbl_cl},
        {"config": "+ Fusion + Layout (Quantization excluded)", "p50_ms": res_cnn_fl.median_ms, "speedup": sp_cfl, "pct_ort_gap": gap_cfl, "gap_label": lbl_cfl},
        {"config": "+ Quantization only (Phase 4)", "p50_ms": res_cnn_q.median_ms, "speedup": sp_cq, "pct_ort_gap": gap_cq, "gap_label": lbl_cq},
        {"config": "+ Combined incl. Quantization (not recommended — see Phase 4 negative result)", "p50_ms": res_cnn_c.median_ms, "speedup": sp_cc, "pct_ort_gap": gap_cc, "gap_label": lbl_cc},
        {"config": "PyTorch Eager (Sanity Baseline)", "p50_ms": res_cnn_pt.median_ms, "speedup": sp_cpt, "pct_ort_gap": gap_cpt, "gap_label": lbl_cpt},
        {"config": "Real ONNX Runtime (Target Baseline)", "p50_ms": ort_cnn_lat, "speedup": calculate_attribution(n_cnn_lat, ort_cnn_lat, ort_cnn_lat)[0], "pct_ort_gap": 100.0, "gap_label": "100.0% (reference)"},
        {"config": "Real TensorRT", "p50_ms": res_cnn_trt[0].median_ms if res_cnn_trt else None, "speedup": None, "pct_ort_gap": None, "gap_label": "Skipped (GPU-gated)"},
    ]

    results["models"]["tiny_cnn"] = {
        "attribution_table": cnn_table,
        "raw_results": {
            "naive": res_cnn_n.to_dict(),
            "fused": res_cnn_f.to_dict(),
            "layout": res_cnn_l.to_dict(),
            "fused_layout": res_cnn_fl.to_dict(),
            "quantized": res_cnn_q.to_dict(),
            "combined": res_cnn_c.to_dict(),
            "pytorch_eager": res_cnn_pt.to_dict(),
            "onnxruntime": res_cnn_ort.to_dict(),
        },
    }

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 5] Attribution experiment complete -> Saved to {output_json_path}")
    return results


if __name__ == "__main__":
    run_phase5_experiment()
