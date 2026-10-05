"""Phase 3 Experiment: Memory Layout & Cache-Tiling Ablation.

Measures the impact of memory contiguity, cache blocking (tiling) across block sizes,
and pre-allocated buffer management in isolation.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import numpy as np
import torch

from ..graph.builder import build_from_pytorch
from ..executors.naive_executor import NaiveExecutor
from ..executors.layout_executor import LayoutExecutor
from ..kernels.tiled_matmul import tiled_matmul
from ..models.tiny_cnn import TinyCNN
from ..models.tiny_transformer import TinyTransformer
from ..profiling.timers import BenchmarkHarness


def run_phase3_experiment(
    output_json_path: str = "results/raw_logs/phase3_layout.json",
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Runs Phase 3 layout ablation experiments."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    harness = BenchmarkHarness(warmup_runs=25, num_trials=100, num_threads=num_threads)

    results: Dict[str, Any] = {
        "phase": 3,
        "description": "Memory Layout Optimization & Cache Tiling Ablation",
        "matmul_microbenchmarks": {},
        "tiling_sweep": [],
        "model_level_results": {},
    }

    # -------------------------------------------------------------
    # 1. MatMul Microbenchmarks (Contiguous vs Non-Contiguous vs BLAS)
    # -------------------------------------------------------------
    M, K, N = 256, 256, 256
    np.random.seed(42)
    # Create non-contiguous tensor by slicing/transposing a larger buffer
    big_buffer = np.random.randn(M * 2, K * 2).astype(np.float32)
    A_non_contig = big_buffer[::2, ::2]  # Strided view
    B = np.random.randn(K, N).astype(np.float32)

    A_contig = np.ascontiguousarray(A_non_contig)
    B_contig = np.ascontiguousarray(B)

    # Warmup JIT if numba is used
    _ = tiled_matmul(A_contig[:32, :32], B_contig[:32, :32], block_size=16)

    res_strided = harness.benchmark(lambda: np.matmul(A_non_contig, B))
    res_contig = harness.benchmark(lambda: np.matmul(A_contig, B_contig))

    results["matmul_microbenchmarks"]["strided_non_contiguous_ms"] = res_strided.median_ms
    results["matmul_microbenchmarks"]["contiguous_blas_ms"] = res_contig.median_ms
    results["matmul_microbenchmarks"]["contiguity_speedup"] = res_strided.median_ms / max(res_contig.median_ms, 1e-9)

    # -------------------------------------------------------------
    # 2. Cache-Tiling Block Size Sweep (16, 32, 64, 128) vs BLAS
    # -------------------------------------------------------------
    for bs in [16, 32, 64, 128]:
        res_tiled = harness.benchmark(
            lambda: tiled_matmul(A_contig, B_contig, block_size=bs),
            custom_trials=50,
        )
        results["tiling_sweep"].append({
            "block_size": bs,
            "median_ms": res_tiled.median_ms,
            "speedup_vs_blas": res_contig.median_ms / max(res_tiled.median_ms, 1e-9),  # < 1.0 expected
        })

    # -------------------------------------------------------------
    # 3. Model-Level Layout Executor Evaluation
    # -------------------------------------------------------------
    # TinyCNN
    cnn_model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=32)
    cnn_input_tensor = torch.randn(1, 3, 32, 32)
    cnn_input_np = {"input_0": cnn_input_tensor.numpy()}
    cnn_graph = build_from_pytorch(cnn_model, cnn_input_tensor)

    cnn_naive = NaiveExecutor(cnn_graph)
    cnn_layout = LayoutExecutor(cnn_graph, enable_buffer_pool=True)

    res_cnn_n = harness.benchmark(lambda: cnn_naive.run(cnn_input_np))
    res_cnn_l = harness.benchmark(lambda: cnn_layout.run(cnn_input_np))

    results["model_level_results"]["tiny_cnn"] = {
        "naive_p50_ms": res_cnn_n.median_ms,
        "layout_p50_ms": res_cnn_l.median_ms,
        "speedup": res_cnn_n.median_ms / max(res_cnn_l.median_ms, 1e-9),
    }

    # TinyTransformer
    trans_model = TinyTransformer(num_layers=2, d_model=64, n_heads=4, d_ff=128)
    trans_input_tensor = torch.randn(1, 16, 64)
    trans_input_np = {"input_0": trans_input_tensor.numpy()}
    trans_graph = build_from_pytorch(trans_model, trans_input_tensor)

    trans_naive = NaiveExecutor(trans_graph)
    trans_layout = LayoutExecutor(trans_graph, enable_buffer_pool=True)

    res_trans_n = harness.benchmark(lambda: trans_naive.run(trans_input_np))
    res_trans_l = harness.benchmark(lambda: trans_layout.run(trans_input_np))

    results["model_level_results"]["tiny_transformer"] = {
        "naive_p50_ms": res_trans_n.median_ms,
        "layout_p50_ms": res_trans_l.median_ms,
        "speedup": res_trans_n.median_ms / max(res_trans_l.median_ms, 1e-9),
    }

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 3] Layout ablation complete -> Saved to {output_json_path}")
    return results


if __name__ == "__main__":
    run_phase3_experiment()
