"""Memory allocation and footprint tracking utilities.

Uses Python tracemalloc and tensor inspection to quantify peak memory,
current allocations, and intermediate tensor overheads.
"""

from __future__ import annotations
import gc
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


@dataclass
class MemoryProfile:
    """Detailed memory snapshot of an execution."""
    peak_kb: float
    current_kb: float
    total_allocated_kb: float

    def summary(self) -> str:
        return (
            f"Memory Footprint: Peak={self.peak_kb:.2f} KB | "
            f"Current={self.current_kb:.2f} KB"
        )


class MemoryTracker:
    """Instruments memory allocation during execution."""

    @staticmethod
    def measure(fn: Callable[[], Any]) -> Tuple[Any, MemoryProfile]:
        """Runs fn() while tracking tracemalloc peak memory usage."""
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        result = fn()

        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        profile = MemoryProfile(
            peak_kb=peak_bytes / 1024.0,
            current_kb=current_bytes / 1024.0,
            total_allocated_kb=peak_bytes / 1024.0,
        )
        return result, profile
