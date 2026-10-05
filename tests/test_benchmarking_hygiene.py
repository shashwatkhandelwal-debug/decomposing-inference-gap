"""Tests for the benchmarking harness to verify measurement hygiene and reliability."""

import time
import pytest
import numpy as np

from src.profiling.timers import BenchmarkHarness, get_environment_fingerprint


def test_environment_fingerprint_completeness():
    """Verify system and hardware environment fields are gathered properly."""
    env = get_environment_fingerprint()
    assert "cpu_name" in env
    assert "logical_cores" in env and env["logical_cores"] is not None and env["logical_cores"] > 0
    assert "total_ram_gb" in env and env["total_ram_gb"] > 0
    assert "torch_version" in env
    assert "os" in env


def test_warmup_detection_and_steady_state():
    """Verify that harness measures steady state after initial warmup delay."""
    # Create a synthetic function that is intentionally slow on first 5 calls
    state = {"calls": 0}

    def synthetic_jit_workload():
        state["calls"] += 1
        if state["calls"] <= 5:
            time.sleep(0.003)  # 3ms first few calls
        else:
            time.sleep(0.0005)  # 0.5ms steady state

    harness = BenchmarkHarness(warmup_runs=10, num_trials=30, trim_percent=0.0)
    result = harness.benchmark(synthetic_jit_workload)

    # Steady state median should reflect ~0.5ms, not 3ms
    assert result.median_ms < 2.0, f"Warmup did not absorb initial cold runs: p50 = {result.median_ms}ms"


def test_outlier_trimming():
    """Verify that trim_percent properly rejects extreme latency spikes."""
    state = {"trial": 0}

    def spike_workload():
        state["trial"] += 1
        if state["trial"] == 50:
            time.sleep(0.020)  # Injected massive 20ms outlier spike
        else:
            time.sleep(0.0005)

    # With trimming enabled
    harness_trimmed = BenchmarkHarness(warmup_runs=5, num_trials=100, trim_percent=0.02)
    res_trimmed = harness_trimmed.benchmark(spike_workload)

    # Without trimming
    state["trial"] = 0
    harness_untrimmed = BenchmarkHarness(warmup_runs=5, num_trials=100, trim_percent=0.0)
    res_untrimmed = harness_untrimmed.benchmark(spike_workload)

    assert res_trimmed.max_ms >= res_untrimmed.min_ms
    assert res_trimmed.mean_ms < res_untrimmed.mean_ms, (
        f"Trimmed mean ({res_trimmed.mean_ms}ms) should be lower than untrimmed ({res_untrimmed.mean_ms}ms)"
    )


def test_percentile_monotonicity():
    """Verify statistical invariants: min <= p50 <= p95 <= p99 <= max."""
    def dummy_op():
        arr = np.random.randn(100, 100)
        _ = arr @ arr

    harness = BenchmarkHarness(warmup_runs=5, num_trials=50, trim_percent=0.01)
    res = harness.benchmark(dummy_op)

    assert res.min_ms <= res.median_ms <= res.p95_ms <= res.p99_ms <= res.max_ms
    assert res.mean_ms >= res.min_ms
    assert res.std_ms >= 0.0
