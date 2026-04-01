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

"""Build a Report from a KVStore snapshot.

Works identically for live metrics (mid-test) and final reports (post-drain).
The caller decides when to call — the function just reads the current state.
"""

from __future__ import annotations

from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    BasicKVStoreReader,
    SeriesStats,
)
from inference_endpoint.utils.version import get_version_info

from .report import Report, compute_summary


def build_report(reader: BasicKVStoreReader) -> Report:
    """Build a Report from the current KVStore state.

    Reads counters and series from the reader, computes rollup summaries
    (percentiles, histograms) for each series metric, and returns a Report.
    """
    snap = reader.snapshot()

    def _counter(key: str) -> int:
        val = snap.get(key)
        return int(val) if isinstance(val, int) else 0

    # Counters
    n_issued = _counter("n_samples_issued")
    n_completed = _counter("n_samples_completed")
    n_failed = _counter("n_samples_failed")
    duration_ns = _counter("duration_ns")
    test_started_at = _counter("test_started_at")

    # Series → summaries
    def _summarize(key: str) -> dict:
        val = snap.get(key)
        if isinstance(val, SeriesStats) and val.count > 0:
            return compute_summary(val)
        return {}

    version_info = get_version_info()

    return Report(
        version=str(version_info.get("version", "unknown")),
        git_sha=version_info.get("git_sha"),
        test_started_at=test_started_at,
        n_samples_issued=n_issued,
        n_samples_completed=n_completed,
        n_samples_failed=n_failed,
        duration_ns=duration_ns if duration_ns > 0 else None,
        ttft=_summarize("ttft_ns"),
        tpot=_summarize("tpot_ns"),
        latency=_summarize("sample_latency_ns"),
        output_sequence_lengths=_summarize("osl"),
    )
