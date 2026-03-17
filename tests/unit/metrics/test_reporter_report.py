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

import io
import json

from inference_endpoint.metrics.reporter import MetricsReporter


def test_reporter_create_report(events_db, fake_outputs, tokenizer):
    with MetricsReporter(events_db) as reporter:
        report = reporter.create_report(tokenizer)

        # Expected
        ttft_rollup = reporter.derive_TTFT()
        sample_latency_rollup = reporter.derive_sample_latency()
        tpot_rollup = reporter.derive_TPOT(
            tokenizer,
            ttft_rollup=ttft_rollup,
            sample_latency_rollup=sample_latency_rollup,
        )

    assert report.n_samples_issued == 3
    assert report.n_samples_completed == 2
    assert report.duration_ns == (10300 - 5000)

    for k, expected in ttft_rollup.summarize().items():
        assert k in report.ttft
        assert report.ttft[k] == expected
    for k, expected in tpot_rollup.summarize().items():
        assert k in report.tpot
        assert report.tpot[k] == expected
    for k, expected in sample_latency_rollup.summarize().items():
        assert k in report.latency
        assert report.latency[k] == expected
    for k, expected in tpot_rollup.summarize().items():
        assert k in report.tpot
        assert report.tpot[k] == expected

    # QPS should be: completed_samples / (duration_ns / 1e9)
    expected_qps = report.n_samples_completed / (report.duration_ns / 1e9)
    assert report.qps == expected_qps

    expected_total_tokens = 0
    for output in fake_outputs.values():
        for chunk in output:
            expected_total_tokens += len(tokenizer.tokenize(chunk))
    expected_tps = expected_total_tokens / ((10300 - 5000) / 1e9)
    assert report.tps == expected_tps


def test_reporter_json(events_db):
    with MetricsReporter(events_db) as reporter:
        report = reporter.create_report()

    json_str = report.to_json()

    json_dict = json.loads(json_str)

    expected_keys = [
        "version",
        "git_sha",
        "n_samples_issued",
        "n_samples_completed",
        "n_samples_failed",
        "duration_ns",
        "ttft",
        "tpot",
        "latency",
        "output_sequence_lengths",
        "tpot_reporting_mode",
        "qps",
        "tps",
        "test_started_at",
    ]
    assert set(json_dict.keys()) == set(expected_keys)
    assert json_dict["n_samples_issued"] == report.n_samples_issued
    assert json_dict["n_samples_completed"] == report.n_samples_completed
    assert json_dict["n_samples_failed"] == report.n_samples_failed
    assert json_dict["duration_ns"] == report.duration_ns
    assert json_dict["qps"] == report.qps
    assert json_dict["tps"] == report.tps

    def _assert_rollup_summary_equal(json_dict, summary_dict):
        if summary_dict is None:
            assert json_dict is None
            return

        for k in summary_dict.keys():
            if k == "histogram":
                continue
            assert json_dict[k] == summary_dict[k]

        assert json_dict["histogram"]["buckets"] == [
            list(bucket) for bucket in summary_dict["histogram"]["buckets"]
        ]
        assert json_dict["histogram"]["counts"] == summary_dict["histogram"]["counts"]

    _assert_rollup_summary_equal(json_dict["ttft"], report.ttft)
    _assert_rollup_summary_equal(json_dict["tpot"], report.tpot)
    _assert_rollup_summary_equal(json_dict["latency"], report.latency)
    _assert_rollup_summary_equal(
        json_dict["output_sequence_lengths"], report.output_sequence_lengths
    )


def test_display_report(events_db):
    with MetricsReporter(events_db) as reporter:
        report = reporter.create_report()

    buf = io.StringIO()

    def _write_with_newline(s):
        buf.write(s + "\n")

    report.display(fn=_write_with_newline)
    s = buf.getvalue()
    lines = s.splitlines()

    assert "- Summary -" in lines[0]
    assert lines[1].startswith("Version:")
    # Git SHA may or may not be present, so Total samples issued can be on line 2 or 3
    assert any(line.startswith("Total samples issued:") for line in lines[2:4])
