# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark report: summary statistics, display, and JSON serialization."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec.json
import numpy as np

from ..utils import monotime_to_datetime

# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

_DEFAULT_PERCENTILES = (99.9, 99, 97, 95, 90, 80, 75, 50, 25, 10, 5, 1)


def compute_summary(
    values: list[float],
    percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
    n_histogram_buckets: int = 10,
) -> dict[str, Any]:
    """Compute rollup statistics from a list of metric values.

    Returns a dict with: total, min, max, avg, std_dev, median,
    percentiles (dict), and histogram (buckets + counts).
    """
    if not values:
        return {
            "total": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "avg": 0.0,
            "std_dev": 0.0,
            "percentiles": {str(p): 0.0 for p in percentiles},
            "histogram": {"buckets": [], "counts": []},
        }

    arr = np.array(values, dtype=np.float64)
    arr.sort()

    perc_values = np.percentile(arr, percentiles, method="linear")
    perc_dict = {
        str(p): float(v) for p, v in zip(percentiles, perc_values, strict=False)
    }

    bounds = np.histogram_bin_edges(arr, bins=n_histogram_buckets)
    counts, _ = np.histogram(arr, bins=bounds)
    hist_buckets = [
        (float(bounds[i]), float(bounds[i + 1])) for i in range(len(bounds) - 1)
    ]

    return {
        "total": float(arr.sum()),
        "min": float(arr[0]),
        "max": float(arr[-1]),
        "median": float(np.median(arr)),
        "avg": float(arr.mean()),
        "std_dev": float(arr.std()),
        "percentiles": perc_dict,
        "histogram": {"buckets": hist_buckets, "counts": counts.tolist()},
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Report:
    """Summarized benchmark report."""

    version: str
    git_sha: str | None
    test_started_at: int
    n_samples_issued: int
    n_samples_completed: int
    n_samples_failed: int
    duration_ns: int | None

    # Per-metric rollup dicts (output of compute_summary)
    ttft: dict[str, Any]
    tpot: dict[str, Any]
    latency: dict[str, Any]
    output_sequence_lengths: dict[str, Any]

    @property
    def qps(self) -> float | None:
        if self.duration_ns is None or self.duration_ns <= 0:
            return None
        return self.n_samples_completed / (self.duration_ns / 1e9)

    @property
    def tps(self) -> float | None:
        if self.duration_ns is None or self.duration_ns <= 0:
            return None
        if not self.output_sequence_lengths:
            return None
        total = self.output_sequence_lengths.get("total", 0)
        if not total:
            return None
        return total / (self.duration_ns / 1e9)

    def to_json(self, save_to: os.PathLike | None = None) -> str:
        d = dataclasses.asdict(self)
        d["qps"] = self.qps
        d["tps"] = self.tps
        json_str = msgspec.json.format(
            msgspec.json.encode(dict(sorted(d.items()))), indent=2
        ).decode("utf-8")
        if save_to is not None:
            with Path(save_to).open("w") as f:
                f.write(json_str)
        return json_str

    def display(
        self,
        fn: Callable[[str], None] = print,
        summary_only: bool = False,
        newline: str = "",
    ) -> None:
        fn(f"----------------- Summary -----------------{newline}")
        fn(f"Version: {self.version}{newline}")
        if self.git_sha:
            fn(f"Git SHA: {self.git_sha}{newline}")
        if self.test_started_at > 0:
            approx = monotime_to_datetime(self.test_started_at)
            fn(f"Test started at: {approx.strftime('%Y-%m-%d %H:%M:%S')}{newline}")
        fn(f"Total samples issued: {self.n_samples_issued}{newline}")
        fn(f"Total samples completed: {self.n_samples_completed}{newline}")
        fn(f"Total samples failed: {self.n_samples_failed}{newline}")
        if self.duration_ns is not None:
            fn(f"Duration: {self.duration_ns / 1e9:.2f} seconds{newline}")
        else:
            fn(f"Duration: N/A{newline}")

        if self.qps is not None:
            fn(f"QPS: {self.qps:.2f}{newline}")
        else:
            fn(f"QPS: N/A{newline}")

        if self.tps is not None:
            fn(f"TPS: {self.tps:.2f}{newline}")

        if summary_only:
            fn(f"----------------- End of Summary -----------------{newline}")
            return

        fn(f"\n------------------- Latency Breakdowns -------------------{newline}")

        for section_name, metric_dict, unit, scale_factor in [
            ("TTFT", self.ttft, "ms", 1e-6),
            ("TPOT", self.tpot, "ms", 1e-6),
            ("Latency", self.latency, "ms", 1e-6),
            ("Output sequence lengths", self.output_sequence_lengths, "tokens", 1.0),
        ]:
            if not metric_dict:
                continue
            fn(f"{section_name}:{newline}")
            _display_metric(
                metric_dict,
                fn=fn,
                unit=unit,
                scale_factor=scale_factor,
                newline=newline,
            )
            fn(f"{newline}")

        fn(f"----------------- End of Report -----------------{newline}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_metric(
    metric_dict: dict[str, Any],
    fn: Callable[[str], None],
    unit: str = "",
    max_bar_length: int = 30,
    scale_factor: float = 1.0,
    newline: str = "",
) -> None:
    for name, key in [
        ("Min", "min"),
        ("Max", "max"),
        ("Median", "median"),
        ("Avg.", "avg"),
        ("Std Dev.", "std_dev"),
    ]:
        fn(f"  {name}: {metric_dict[key] * scale_factor:.2f} {unit}{newline}")

    fn(f"\n  Histogram:{newline}")
    buckets = metric_dict["histogram"]["buckets"]
    counts = metric_dict["histogram"]["counts"]

    if buckets:
        bucket_strs = [
            f"  [{lo * scale_factor:.2f}, {hi * scale_factor:.2f})"
            for lo, hi in buckets
        ]
        max_count = max(counts)
        normalize = max_bar_length / max_count if max_count > 0 else 1
        max_label = max(len(s) for s in bucket_strs)

        for label, count in zip(bucket_strs, counts, strict=False):
            bar = "#" * int(count * normalize)
            fn(f"  {label:>{max_label}} |{bar} {count}{newline}")

    fn(f"\n  Percentiles:{newline}")
    for p, val in metric_dict.get("percentiles", {}).items():
        fn(f"  {p:>6}: {val * scale_factor:.2f} {unit}{newline}")
