"""Master Benchmark Driver: Executes Phases 1 through 7 in sequential order.

Runs baseline characterization, ablation experiments, scaling sweeps,
and figure generation, reproducing all raw logs and plots.
"""

import sys
import time
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.phase1_naive_baseline import run_phase1_experiment
from src.experiments.phase2_fusion_ablation import run_phase2_experiment
from src.experiments.phase3_layout_ablation import run_phase3_experiment
from src.experiments.phase4_quantization_ablation import run_phase4_experiment
from src.experiments.phase5_combined_and_real_baselines import run_phase5_experiment
from src.experiments.phase6_scaling_sweep import run_phase6_experiment
from src.experiments.phase7_attribution_report import generate_all_figures


def main():
    print("=" * 70)
    print(" DECOMPOSING THE INFERENCE SPEEDUP GAP: MASTER BENCHMARK RUNNER")
    print("=" * 70)
    start_time = time.time()

    print("\n>>> [Phase 1/7] Naive Baseline & Op Characterization...")
    p1_res = run_phase1_experiment()

    print("\n>>> [Phase 2/7] Operator Fusion Ablation...")
    p2_res = run_phase2_experiment()

    print("\n>>> [Phase 3/7] Memory Layout & Cache Tiling Ablation...")
    p3_res = run_phase3_experiment()

    print("\n>>> [Phase 4/7] Post-Training Int8 Quantization Ablation...")
    p4_res = run_phase4_experiment()

    print("\n>>> [Phase 5/7] Combined Hand-Built vs Real Baselines...")
    p5_res = run_phase5_experiment()

    print("\n>>> [Phase 6/7] Multi-Dimensional Scaling Sweeps...")
    p6_res = run_phase6_experiment()

    print("\n>>> [Phase 7/7] Generating Research Figures & Attribution Report...")
    generate_all_figures()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" ALL PHASES COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    print(" Results saved to 'results/raw_logs/' and 'results/figures/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
