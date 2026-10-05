"""High-precision benchmarking harness with warmup, tail-latency, and environment controls.

Ensures benchmarking hygiene by standardizing warmup duration, repeated trials,
outlier trimming, and hardware environment fingerprinting.
"""

from __future__ import annotations
import gc
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import psutil
import torch


@dataclass
class BenchmarkResult:
    """Encapsulates comprehensive statistical benchmark metrics."""
    mean_ms: float
    std_ms: float
    median_ms: float  # p50
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    num_trials: int
    warmup_runs: int
    trim_percent: float
    raw_latencies_ms: List[float] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> str:
        return (
            f"Latency: p50={self.median_ms:.3f} ms | "
            f"Mean={self.mean_ms:.3f} ± {self.std_ms:.3f} ms | "
            f"p95={self.p95_ms:.3f} ms | p99={self.p99_ms:.3f} ms | "
            f"Min/Max=[{self.min_ms:.3f}, {self.max_ms:.3f}] ms (N={self.num_trials})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_environment_fingerprint() -> Dict[str, Any]:
    """Captures CPU, memory, OS, and library versions for reproducibility."""
    try:
        cpu_name = platform.processor() or "Unknown CPU"
    except Exception:
        cpu_name = "Unknown CPU"

    try:
        import onnxruntime
        ort_ver = onnxruntime.__version__
    except ImportError:
        ort_ver = "Not Installed"

    return {
        "cpu_name": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "onnxruntime_version": ort_ver,
        "torch_num_threads": torch.get_num_threads(),
    }


class BenchmarkHarness:
    """Standardized timing harness used across all experimental phases."""

    def __init__(
        self,
        warmup_runs: int = 25,
        num_trials: int = 100,
        trim_percent: float = 0.01,
        num_threads: int = 1,
    ):
        """
        Args:
            warmup_runs: Minimum iterations before timing to allow JIT warmup and cache priming.
            num_trials: Number of timed trials recorded.
            trim_percent: Fraction of top and bottom outliers trimmed before summary statistics.
            num_threads: Pinned CPU thread count.
        """
        self.warmup_runs = max(20, warmup_runs)
        self.num_trials = max(50, num_trials)
        self.trim_percent = trim_percent
        self.num_threads = num_threads
        self._apply_thread_pinning()

    def _apply_thread_pinning(self) -> None:
        """Enforces thread configuration across PyTorch and OpenMP."""
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.num_threads)
        torch.set_num_threads(self.num_threads)

    def benchmark(
        self,
        fn: Callable[[], Any],
        custom_warmup: Optional[int] = None,
        custom_trials: Optional[int] = None,
    ) -> BenchmarkResult:
        """Executes warmup, measures trials with nanosecond precision, and aggregates statistics."""
        warmup_count = custom_warmup if custom_warmup is not None else self.warmup_runs
        trials_count = custom_trials if custom_trials is not None else self.num_trials

        # 1. Warmup phase
        gc.disable()
        for _ in range(warmup_count):
            fn()

        # 2. Timed trials
        latencies_ms: List[float] = []
        for _ in range(trials_count):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            latencies_ms.append((t1 - t0) / 1e6)  # Convert ns to ms
        gc.enable()

        latencies_arr = np.array(latencies_ms, dtype=np.float64)

        # 3. Outlier trimming (e.g. top 1% and bottom 1%)
        if self.trim_percent > 0:
            k = int(len(latencies_arr) * self.trim_percent)
            if k > 0 and len(latencies_arr) > 2 * k:
                sorted_lats = np.sort(latencies_arr)
                trimmed_lats = sorted_lats[k:-k]
            else:
                trimmed_lats = latencies_arr
        else:
            trimmed_lats = latencies_arr

        return BenchmarkResult(
            mean_ms=float(np.mean(trimmed_lats)),
            std_ms=float(np.std(trimmed_lats)),
            median_ms=float(np.median(latencies_arr)),
            p95_ms=float(np.percentile(latencies_arr, 95)),
            p99_ms=float(np.percentile(latencies_arr, 99)),
            min_ms=float(np.min(latencies_arr)),
            max_ms=float(np.max(latencies_arr)),
            num_trials=trials_count,
            warmup_runs=warmup_count,
            trim_percent=self.trim_percent,
            raw_latencies_ms=[float(x) for x in latencies_arr],
            environment=get_environment_fingerprint(),
        )

    def measure_warmup_curve(self, fn: Callable[[], Any], total_runs: int = 100) -> List[float]:
        """Measures single-run latency starting from run 0 to observe warmup / cache effects."""
        latencies: List[float] = []
        gc.disable()
        for _ in range(total_runs):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)
        gc.enable()
        return latencies
