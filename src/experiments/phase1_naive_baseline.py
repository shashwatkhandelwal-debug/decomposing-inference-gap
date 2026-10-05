"""Phase 1 Experiment: Naive Executor & Baseline Characterization.

Measures op-by-op naive execution performance, per-operator timing breakdown,
memory footprint, and cross-validates with PyTorch eager baseline.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict
import numpy as np
import torch

from ..graph.builder import build_from_pytorch
from ..executors.naive_executor import NaiveExecutor
from ..baselines.pytorch_eager import PyTorchEagerBaseline
from ..models.tiny_cnn import TinyCNN
from ..models.tiny_transformer import TinyTransformer
from ..profiling.timers import BenchmarkHarness
from ..profiling.memory_trace import MemoryTracker


def run_phase1_experiment(
    output_json_path: str = "results/raw_logs/phase1_naive.json",
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Runs Phase 1 baseline characterization across TinyCNN and TinyTransformer."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    harness = BenchmarkHarness(warmup_runs=25, num_trials=100, num_threads=num_threads)

    results: Dict[str, Any] = {
        "phase": 1,
        "description": "Naive Executor vs PyTorch Eager Baseline Characterization",
        "models": {},
    }

    # -------------------------------------------------------------
    # 1. Tiny CNN
    # -------------------------------------------------------------
    cnn_model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=32)
    cnn_input_tensor = torch.randn(1, 3, 32, 32)
    cnn_input_np = {"input_0": cnn_input_tensor.numpy()}

    cnn_graph = build_from_pytorch(cnn_model, cnn_input_tensor)
    cnn_executor = NaiveExecutor(cnn_graph)

    # PyTorch Eager Baseline
    cnn_pytorch = PyTorchEagerBaseline(cnn_model, harness)
    cnn_pt_res, cnn_pt_out = cnn_pytorch.run_benchmark(cnn_input_tensor)

    # Naive Executor Benchmark
    def cnn_naive_fn():
        return cnn_executor.run(cnn_input_np, enable_profiling=False)

    cnn_naive_res = harness.benchmark(cnn_naive_fn)

    # Profiling & Per-op breakdown
    for _ in range(20):
        cnn_executor.run(cnn_input_np, enable_profiling=True)
    cnn_op_report = cnn_executor.get_profile_report()

    # Memory Tracking
    _, cnn_mem_profile = MemoryTracker.measure(cnn_naive_fn)

    # Numerical Parity Check
    cnn_naive_out = cnn_naive_fn()["logits"]
    cnn_max_err = float(np.max(np.abs(cnn_naive_out - cnn_pt_out)))
    cnn_rel_err = float(np.max(np.abs(cnn_naive_out - cnn_pt_out) / (np.abs(cnn_pt_out) + 1e-7)))

    results["models"]["tiny_cnn"] = {
        "naive_latency": cnn_naive_res.to_dict(),
        "pytorch_eager_latency": cnn_pt_res.to_dict(),
        "speedup_vs_pytorch": cnn_pt_res.median_ms / max(cnn_naive_res.median_ms, 1e-9),
        "per_op_type_ms": cnn_op_report.per_op_type_ms if cnn_op_report else {},
        "per_op_type_pct": cnn_op_report.per_op_type_percent if cnn_op_report else {},
        "memory_profile_kb": {
            "peak_kb": cnn_mem_profile.peak_kb,
            "current_kb": cnn_mem_profile.current_kb,
        },
        "correctness": {
            "max_abs_error": cnn_max_err,
            "max_rel_error": cnn_rel_err,
            "numerically_valid": bool(cnn_max_err < 1e-4),
        },
    }

    # -------------------------------------------------------------
    # 2. Tiny Transformer
    # -------------------------------------------------------------
    trans_model = TinyTransformer(num_layers=2, d_model=64, n_heads=4, d_ff=128)
    trans_input_tensor = torch.randn(1, 16, 64)
    trans_input_np = {"input_0": trans_input_tensor.numpy()}

    trans_graph = build_from_pytorch(trans_model, trans_input_tensor)
    trans_executor = NaiveExecutor(trans_graph)

    # PyTorch Eager Baseline
    trans_pytorch = PyTorchEagerBaseline(trans_model, harness)
    trans_pt_res, trans_pt_out = trans_pytorch.run_benchmark(trans_input_tensor)

    # Naive Executor Benchmark
    def trans_naive_fn():
        return trans_executor.run(trans_input_np, enable_profiling=False)

    trans_naive_res = harness.benchmark(trans_naive_fn)

    # Profiling & Per-op breakdown
    for _ in range(20):
        trans_executor.run(trans_input_np, enable_profiling=True)
    trans_op_report = trans_executor.get_profile_report()

    # Memory Tracking
    _, trans_mem_profile = MemoryTracker.measure(trans_naive_fn)

    # Numerical Parity Check
    trans_naive_out = trans_naive_fn()["logits"]
    trans_max_err = float(np.max(np.abs(trans_naive_out - trans_pt_out)))
    trans_rel_err = float(np.max(np.abs(trans_naive_out - trans_pt_out) / (np.abs(trans_pt_out) + 1e-7)))

    results["models"]["tiny_transformer"] = {
        "naive_latency": trans_naive_res.to_dict(),
        "pytorch_eager_latency": trans_pt_res.to_dict(),
        "speedup_vs_pytorch": trans_pt_res.median_ms / max(trans_naive_res.median_ms, 1e-9),
        "per_op_type_ms": trans_op_report.per_op_type_ms if trans_op_report else {},
        "per_op_type_pct": trans_op_report.per_op_type_percent if trans_op_report else {},
        "memory_profile_kb": {
            "peak_kb": trans_mem_profile.peak_kb,
            "current_kb": trans_mem_profile.current_kb,
        },
        "correctness": {
            "max_abs_error": trans_max_err,
            "max_rel_error": trans_rel_err,
            "numerically_valid": bool(trans_max_err < 1e-4),
        },
    }

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 1] Baseline characterization complete -> Saved to {output_json_path}")
    return results


if __name__ == "__main__":
    run_phase1_experiment()
