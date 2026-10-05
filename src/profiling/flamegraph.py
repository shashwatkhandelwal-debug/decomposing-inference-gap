"""Operator profiling and breakdown utilities.

Tracks per-node and per-op-type timing breakdown to pinpoint performance bottlenecks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class OpProfileEntry:
    op_type: str
    node_name: str
    total_time_ms: float
    percentage: float
    call_count: int = 1


@dataclass
class OpProfileReport:
    total_time_ms: float
    per_op_type_ms: Dict[str, float] = field(default_factory=dict)
    per_op_type_percent: Dict[str, float] = field(default_factory=dict)
    per_node_ms: Dict[str, float] = field(default_factory=dict)
    entries: List[OpProfileEntry] = field(default_factory=list)

    def summary_table(self) -> str:
        lines = ["Operator Type Breakdown:"]
        lines.append(f"{'Op Type':<18} | {'Total Time (ms)':<15} | {'Share (%)':<10}")
        lines.append("-" * 50)
        sorted_ops = sorted(self.per_op_type_ms.items(), key=lambda x: x[1], reverse=True)
        for op_type, time_ms in sorted_ops:
            pct = self.per_op_type_percent.get(op_type, 0.0)
            lines.append(f"{op_type:<18} | {time_ms:<15.4f} | {pct:<10.2f}%")
        lines.append("-" * 50)
        lines.append(f"{'Total':<18} | {self.total_time_ms:<15.4f} | 100.00%")
        return "\n".join(lines)


class OpProfiler:
    """Collects and aggregates per-operator timings."""

    def __init__(self):
        self.node_times_ms: Dict[str, List[float]] = {}
        self.node_op_types: Dict[str, str] = {}

    def record(self, node_name: str, op_type: str, time_ms: float) -> None:
        self.node_times_ms.setdefault(node_name, []).append(time_ms)
        self.node_op_types[node_name] = op_type

    def generate_report(self) -> OpProfileReport:
        node_avg: Dict[str, float] = {k: float(np.mean(v)) for k, v in self.node_times_ms.items()}
        total_time = sum(node_avg.values()) or 1e-9

        per_op_type_ms: Dict[str, float] = {}
        for node_name, avg_time in node_avg.items():
            op_type = self.node_op_types[node_name]
            per_op_type_ms[op_type] = per_op_type_ms.get(op_type, 0.0) + avg_time

        per_op_type_pct = {k: (v / total_time) * 100.0 for k, v in per_op_type_ms.items()}

        entries = []
        for node_name, avg_time in sorted(node_avg.items(), key=lambda x: x[1], reverse=True):
            op_type = self.node_op_types[node_name]
            entries.append(
                OpProfileEntry(
                    op_type=op_type,
                    node_name=node_name,
                    total_time_ms=avg_time,
                    percentage=(avg_time / total_time) * 100.0,
                )
            )

        return OpProfileReport(
            total_time_ms=total_time,
            per_op_type_ms=per_op_type_ms,
            per_op_type_percent=per_op_type_pct,
            per_node_ms=node_avg,
            entries=entries,
        )
