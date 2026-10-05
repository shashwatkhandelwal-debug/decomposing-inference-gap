"""Phase 7: Attribution Report and Publication Figures Generator.

Reads the experimental logs from Phases 1-6 and generates all 6 required figures:
  Fig 1: Per-operator time breakdown (stacked bar)
  Fig 2: Fusion speedup by pattern, isolated
  Fig 3: Layout/tiling latency vs block size with BLAS baseline
  Fig 4: Quantization speedup vs accuracy Pareto scatter
  Fig 5: Speedup gap attribution waterfall chart
  Fig 6: Scaling sweep gap closed vs model depth & batch size
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np


def generate_all_figures(
    logs_dir: str = "results/raw_logs",
    figures_dir: str = "results/figures",
) -> None:
    """Generates the 6 core research figures from raw experimental JSON logs."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.dpi": 300})

    # Load JSON logs
    p1_log = os.path.join(logs_dir, "phase1_naive.json")
    p2_log = os.path.join(logs_dir, "phase2_fusion.json")
    p3_log = os.path.join(logs_dir, "phase3_layout.json")
    p4_log = os.path.join(logs_dir, "phase4_quantization.json")
    p5_log = os.path.join(logs_dir, "phase5_combined.json")
    p6_log = os.path.join(logs_dir, "phase6_scaling.json")

    # -------------------------------------------------------------
    # Figure 1: Per-Operator Time Breakdown (Stacked Bar)
    # -------------------------------------------------------------
    if os.path.exists(p1_log):
        with open(p1_log) as f:
            d1 = json.load(f)

        fig, ax = plt.subplots(figsize=(8, 5))
        models = ["TinyCNN", "TinyTransformer"]
        keys = ["tiny_cnn", "tiny_transformer"]

        # Collect op types
        all_ops = set()
        for k in keys:
            all_ops.update(d1["models"][k]["per_op_type_pct"].keys())
        all_ops = sorted(list(all_ops))

        bottom = np.zeros(len(models))
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_ops)))

        for i, op in enumerate(all_ops):
            values = [d1["models"][k]["per_op_type_pct"].get(op, 0.0) for k in keys]
            ax.bar(models, values, bottom=bottom, label=op, color=colors[i], width=0.45, edgecolor="black", alpha=0.9)
            bottom += np.array(values)

        ax.set_ylabel("Percentage of Total Latency (%)", fontweight="bold")
        ax.set_title("Figure 1: Per-Operator Execution Time Breakdown (Naive Mode)", fontweight="bold", pad=15)
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Operator Type")
        plt.tight_layout()
        fig1_path = os.path.join(figures_dir, "fig1_per_op_breakdown.png")
        plt.savefig(fig1_path)
        plt.close()
        print(f"[Fig 1] Saved -> {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Isolated Fusion Speedup by Pattern
    # -------------------------------------------------------------
    if os.path.exists(p2_log):
        with open(p2_log) as f:
            d2 = json.load(f)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        patterns = ["Conv+BN+ReLU", "MatMul+Bias+Act", "Fused LayerNorm", "All Combined (CNN)", "All Combined (Trans)"]
        speedups = [
            d2["patterns_isolated"]["conv_bn_relu"]["speedup"],
            d2["patterns_isolated"]["matmul_bias_act"]["speedup"],
            d2["patterns_isolated"]["layernorm_fused"]["speedup"],
            d2["models_combined"]["tiny_cnn"]["speedup"],
            d2["models_combined"]["tiny_transformer"]["speedup"],
        ]

        bars = ax.bar(patterns, speedups, color="#2b5c8f", width=0.5, edgecolor="black", alpha=0.85)
        ax.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.5, label="Naive Baseline (1.0x)")

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}x", ha="center", va="bottom", fontweight="bold")

        ax.set_ylabel("Speedup Factor (x)", fontweight="bold")
        ax.set_title("Figure 2: Isolated Operator Fusion Speedups", fontweight="bold", pad=15)
        ax.set_xticklabels(patterns, rotation=15, ha="right")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig2_path = os.path.join(figures_dir, "fig2_fusion_speedup_by_pattern.png")
        plt.savefig(fig2_path)
        plt.close()
        print(f"[Fig 2] Saved -> {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Layout & Tiling vs BLAS
    # -------------------------------------------------------------
    if os.path.exists(p3_log):
        with open(p3_log) as f:
            d3 = json.load(f)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sweep = d3["tiling_sweep"]
        block_sizes = [str(x["block_size"]) for x in sweep]
        latencies = [x["median_ms"] for x in sweep]
        blas_lat = d3["matmul_microbenchmarks"]["contiguous_blas_ms"]

        ax.plot(block_sizes, latencies, marker="o", linewidth=2.5, markersize=8, color="#e67e22", label="Hand-Tiled MatMul")
        ax.axhline(blas_lat, color="#27ae60", linestyle="--", linewidth=2.0, label=f"OpenBLAS/MKL Baseline ({blas_lat:.3f} ms)")

        ax.set_xlabel("Tile Block Size (B x B)", fontweight="bold")
        ax.set_ylabel("Latency (ms) - Lower is Better", fontweight="bold")
        ax.set_title("Figure 3: Cache-Tiling Block Size Sweep vs Optimized BLAS", fontweight="bold", pad=15)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig3_path = os.path.join(figures_dir, "fig3_layout_tiling_vs_blas.png")
        plt.savefig(fig3_path)
        plt.close()
        print(f"[Fig 3] Saved -> {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Quantization Pareto Frontier (Speedup vs MSE Error)
    # -------------------------------------------------------------
    if os.path.exists(p4_log):
        with open(p4_log) as f:
            d4 = json.load(f)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        m_cnn = d4["models"]["tiny_cnn"]
        m_trans = d4["models"]["tiny_transformer"]

        ax.scatter([m_cnn["speedup"]], [m_cnn["mse_degradation"]], color="#8e44ad", s=140, label="TinyCNN (int8)", zorder=4, edgecolor="black")
        ax.scatter([m_trans["speedup"]], [m_trans["mse_degradation"]], color="#2980b9", s=140, label="TinyTransformer (int8)", zorder=4, edgecolor="black")

        ax.text(m_cnn["speedup"] + 0.01, m_cnn["mse_degradation"], f" CNN (MSE={m_cnn['mse_degradation']:.4f})", va="center", fontweight="bold")
        ax.text(m_trans["speedup"] + 0.01, m_trans["mse_degradation"], f" Transformer (MSE={m_trans['mse_degradation']:.4f})", va="center", fontweight="bold")

        ax.set_xlabel("Latency Speedup vs Naive fp32 (x)", fontweight="bold")
        ax.set_ylabel("Reconstruction MSE Degradation", fontweight="bold")
        ax.set_title("Figure 4: Quantization Efficiency & Precision Tradeoff", fontweight="bold", pad=15)
        ax.axvline(1.0, color="#c0392b", linestyle=":", label="fp32 Baseline Speed")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig4_path = os.path.join(figures_dir, "fig4_quantization_pareto.png")
        plt.savefig(fig4_path)
        plt.close()
        print(f"[Fig 4] Saved -> {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Attribution Waterfall Chart (TinyTransformer)
    # -------------------------------------------------------------
    if os.path.exists(p5_log):
        with open(p5_log) as f:
            d5 = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        table = d5["models"]["tiny_transformer"]["attribution_table"]
        filtered = [row for row in table if row["p50_ms"] is not None and "TensorRT" not in row["config"]]

        short_names = [
            "Naive (Phase 1)",
            "+ Fusion only",
            "+ Layout only",
            "+ Fusion + Layout (Best)",
            "+ Quant only (Negative)",
            "+ Combined w/ Quant",
            "PyTorch Eager",
            "ONNX Runtime",
        ]
        # Match lengths
        configs = short_names[:len(filtered)]
        latencies = [row["p50_ms"] for row in filtered]

        colors = ["#7f8c8d", "#3498db", "#f39c12", "#27ae60", "#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
        bars = ax.bar(range(len(configs)), latencies, color=colors[:len(configs)], width=0.55, edgecolor="black", alpha=0.9)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}ms", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax.set_ylabel("p50 Latency (ms) - Lower is Better", fontweight="bold")
        ax.set_title("Figure 5: Decomposing the Inference Speedup Gap (TinyTransformer)", fontweight="bold", pad=15)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=25, ha="right", fontweight="bold")
        ax.set_ylim(0, max(latencies) * 1.15)
        plt.tight_layout()
        fig5_path = os.path.join(figures_dir, "fig5_speedup_attribution_waterfall.png")
        plt.savefig(fig5_path)
        plt.close()
        print(f"[Fig 5] Saved -> {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6: Scaling Sweep (Depth & Batch Size vs % Gap Closed)
    # -------------------------------------------------------------
    if os.path.exists(p6_log):
        with open(p6_log) as f:
            d6 = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 6a: Depth Sweep
        depth_data = d6["depth_sweep"]
        depths = [x["num_layers"] for x in depth_data]
        ax1.plot(depths, [x["pct_gap_fused"] for x in depth_data], marker="o", label="+ Fusion", color="#3498db", linewidth=2)
        ax1.plot(depths, [x["pct_gap_layout"] for x in depth_data], marker="s", label="+ Layout", color="#f39c12", linewidth=2)
        ax1.plot(depths, [x["pct_gap_quant"] for x in depth_data], marker="^", label="+ Quantization", color="#9b59b6", linewidth=2)
        ax1.plot(depths, [x["pct_gap_combined"] for x in depth_data], marker="D", label="Combined", color="#1abc9c", linewidth=2.5)

        ax1.set_xlabel("Transformer Depth (Number of Layers)", fontweight="bold")
        ax1.set_ylabel("% of ONNX Runtime Gap Closed", fontweight="bold")
        ax1.set_title("(a) Gap Closed vs Layer Depth", fontweight="bold")
        ax1.legend()
        ax1.grid(True)

        # 6b: Batch Size Sweep
        bs_data = d6["batch_size_sweep"]
        batch_sizes = [str(x["batch_size"]) for x in bs_data]
        ax2.plot(batch_sizes, [x["pct_gap_fused"] for x in bs_data], marker="o", label="+ Fusion", color="#3498db", linewidth=2)
        ax2.plot(batch_sizes, [x["pct_gap_layout"] for x in bs_data], marker="s", label="+ Layout", color="#f39c12", linewidth=2)
        ax2.plot(batch_sizes, [x["pct_gap_quant"] for x in bs_data], marker="^", label="+ Quantization", color="#9b59b6", linewidth=2)
        ax2.plot(batch_sizes, [x["pct_gap_combined"] for x in bs_data], marker="D", label="Combined", color="#1abc9c", linewidth=2.5)

        ax2.set_xlabel("Inference Batch Size", fontweight="bold")
        ax2.set_ylabel("% of ONNX Runtime Gap Closed", fontweight="bold")
        ax2.set_title("(b) Gap Closed vs Batch Size", fontweight="bold")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Figure 6: Scaling Regime Sweeps — Optimization Contribution Dynamics", fontweight="bold", y=1.02)
        plt.tight_layout()
        fig6_path = os.path.join(figures_dir, "fig6_scaling_sweep_gap_closed.png")
        plt.savefig(fig6_path)
        plt.close()
        print(f"[Fig 6] Saved -> {fig6_path}")


if __name__ == "__main__":
    generate_all_figures()
