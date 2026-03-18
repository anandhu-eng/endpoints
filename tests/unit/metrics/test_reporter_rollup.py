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

import math

import pytest
from inference_endpoint.metrics.reporter import (
    MetricsReporter,
    RollupQueryTable,
    TPOTReportingMode,
)


def test_tpot_to_histogram(events_db, fake_outputs, tokenizer, sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    expected = [
        {
            "tpot": (10211 - 10000 - 10) / len(fake_outputs[uuid1][1]),
            "count": len(fake_outputs[uuid1][1]),
        },
        {
            "tpot": (10219 - 10003 - 187) / len(fake_outputs[uuid2][1]),
            "count": len(fake_outputs[uuid2][1]),
        },
    ]
    expected.sort(key=lambda x: x["tpot"])

    bucket_boundaries = [
        expected[0]["tpot"] - 1,
        (expected[0]["tpot"] + expected[1]["tpot"]) / 2,
        expected[1]["tpot"] + 1,
    ]

    with MetricsReporter(events_db) as reporter:
        tpot_rows = reporter.derive_TPOT(
            tokenizer, reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED
        )

    # This isn't documented since it's an internal detail and should not be relied on, but `n_buckets`
    # is passed directly to np.histogram, so we can specify exact buckets to use
    buckets, counts = tpot_rows.to_histogram(n_buckets=bucket_boundaries)
    assert len(buckets) == 2
    assert len(counts) == 2

    assert buckets[0] == (bucket_boundaries[0], bucket_boundaries[1])
    assert buckets[1] == (bucket_boundaries[1], bucket_boundaries[2])
    assert counts[0] == expected[0]["count"]
    assert counts[1] == expected[1]["count"]


def test_percentile():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    table = RollupQueryTable(
        metric_type="test", from_query="", rows=[(0, v) for v in values]
    )
    assert table.percentile(50) == 5
    assert table.percentile([50, 75]) == {50: 5, 75: 7.5}
    assert table.percentile(90) == 9
    with pytest.raises(TypeError):
        table.percentile("10")
    with pytest.raises(ValueError):
        table.percentile(101)
    with pytest.raises(ValueError):
        table.percentile(-1)


def test_rollup_summarize(events_db):
    with MetricsReporter(events_db) as reporter:
        latencies = reporter.derive_sample_latency()
    summary = latencies.summarize()
    values = [10211 - 10000, 10219 - 10003]
    assert summary["total"] == sum(values)
    assert summary["min"] == min(values)
    assert summary["max"] == max(values)
    assert summary["median"] == (values[0] + values[1]) / 2
    assert summary["avg"] == (values[0] + values[1]) / 2

    deviations_squared = [(value - summary["avg"]) ** 2 for value in values]

    assert math.isclose(
        summary["std_dev"],
        math.sqrt(sum(deviations_squared) / len(values)),
        rel_tol=1e-3,
    )

    for percentile in [99.9, 99, 95, 90, 80, 75, 50, 25, 10, 5, 1]:
        s = str(percentile)
        assert s in summary["percentiles"]
        assert summary["percentiles"][s] == latencies.percentile(percentile)
