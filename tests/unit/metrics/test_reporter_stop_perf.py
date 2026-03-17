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

import msgspec.json
import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.metrics.recorder import sqlite3_cursor
from inference_endpoint.metrics.reporter import MetricsReporter

EVENTS_DDL = (
    "CREATE TABLE IF NOT EXISTS events "
    "(sample_uuid VARCHAR(32), event_type VARCHAR(32), "
    "timestamp_ns INTEGER, data BLOB)"
)
EVENTS_DML = (
    "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) "
    "VALUES (?, ?, ?, ?)"
)


def _create_db(tmp_path, name, events):
    test_db = str(tmp_path / name)
    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(EVENTS_DDL)
        cursor.executemany(EVENTS_DML, events)
        conn.commit()
    return test_db


@pytest.mark.parametrize(
    "case_desc, events, expected_ts",
    [
        (
            "event present",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
            10100,
        ),
        (
            "event missing",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
            float("inf"),
        ),
    ],
)
def test_stop_perf_tracking_timestamp(tmp_path, case_desc, events, expected_ts):
    """Test stop_performance_tracking_timestamp_ns with and without the event."""
    test_db = _create_db(tmp_path, f"test_{case_desc.replace(' ', '_')}.db", events)
    with MetricsReporter(test_db) as reporter:
        assert reporter.stop_performance_tracking_timestamp_ns == expected_ts


def test_stop_perf_tracking_missing_duration(tmp_path):
    """Test that derive_duration still works when STOP_PERFORMANCE_TRACKING is absent."""
    test_db = _create_db(
        tmp_path,
        "no_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )
    with MetricsReporter(test_db) as reporter:
        assert reporter.stop_performance_tracking_timestamp_ns == float("inf")
        assert reporter.derive_duration() == 10300 - 5000


def test_derive_ttft_with_stop_perf(tmp_path, sample_uuids):
    """Test that derive_TTFT excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "ttft_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
            (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (uuid3, SampleEvent.FIRST_CHUNK.value, 10220, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
            (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        ttft_rows = reporter.derive_TTFT()

    assert len(ttft_rows) == 2
    assert ttft_rows.filter_uuid(uuid1, only_first=True) == 10
    assert ttft_rows.filter_uuid(uuid2, only_first=True) == 187
    assert ttft_rows.filter_uuid(uuid3, only_first=True) is None


def test_derive_latency_with_stop_perf(tmp_path, sample_uuids):
    """Test that derive_sample_latency excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "latency_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
            (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        latency_rows = reporter.derive_sample_latency()

    assert len(latency_rows) == 2
    assert latency_rows.filter_uuid(uuid1, only_first=True) == 10211 - 10000
    assert latency_rows.filter_uuid(uuid2, only_first=True) == 10219 - 10003
    assert latency_rows.filter_uuid(uuid3, only_first=True) is None


def test_derive_duration_stop_perf_no_samples(tmp_path):
    """Test that derive_duration uses STOP_PERFORMANCE_TRACKING timestamp when present."""
    test_db = _create_db(
        tmp_path,
        "duration_stop_perf_no_samples.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    assert duration is None


def test_derive_duration_with_stop_perf(tmp_path, sample_uuids):
    """Test that derive_duration uses STOP_PERFORMANCE_TRACKING timestamp when present."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "duration_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10213, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10216, b""),
            (uuid2, SampleEvent.COMPLETE.value, 10250, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10219, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    assert duration == 10250 - 5000


def test_duration_all_complete_after_stop(tmp_path, sample_uuids):
    """Test derive_duration when samples are issued before but all complete after STOP_PERFORMANCE_TRACKING."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "duration_all_complete_after_stop.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10010, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
            (uuid2, SampleEvent.COMPLETE.value, 10252, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10219, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    # Should use timestamp of the last COMPLETE event (uuid2 at 10252) - TEST_STARTED
    assert duration == 10252 - 5000


def test_derive_duration_no_stop_perf(tmp_path):
    """Test that derive_duration uses TEST_ENDED when STOP_PERFORMANCE_TRACKING is absent."""
    test_db = _create_db(
        tmp_path,
        "duration_no_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    assert duration == 10300 - 5000


def test_sample_statuses_with_stop_perf(tmp_path, sample_uuids):
    """Test that get_sample_statuses excludes samples issued after STOP_PERFORMANCE_TRACKING.

    Verifies:
    1. Samples issued before stop_ts are counted in total_sent
    2. Samples issued after stop_ts are NOT counted in total_sent
    3. Completed samples are only counted if they were issued before stop_ts
    4. Samples issued before stop_ts but completing after stop_ts ARE still counted as completed
    """
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "statuses_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10100, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        stats = reporter.get_sample_statuses()
        assert reporter.stop_performance_tracking_timestamp_ns == 10150

    assert stats["total_sent"] == 2
    assert stats["completed"] == 2
    assert stats["in_flight"] == 0


def test_statuses_excludes_late_issued(tmp_path, sample_uuids):
    """Test that completed samples issued after STOP_PERFORMANCE_TRACKING are not counted.

    Specifically tests the edge case where a sample is issued after stop_ts and completes,
    ensuring it's not included in the completed count.
    """
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)
    uuid4 = sample_uuids(4)

    test_db = _create_db(
        tmp_path,
        "late_completion.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.COMPLETE.value, 10100, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (uuid4, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10205, b""),
            (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
            (uuid4, SampleEvent.COMPLETE.value, 10260, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        stats = reporter.get_sample_statuses()

    assert stats["total_sent"] == 2
    assert stats["completed"] == 1
    assert stats["in_flight"] == 1


def test_osl_with_stop_perf(tmp_path, sample_uuids, tokenizer):
    """Test that get_output_sequence_lengths excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "osl_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (
                uuid1,
                SampleEvent.COMPLETE.value,
                10211,
                msgspec.json.encode({"output": ["Hello, ", "world"]}),
            ),
            (
                uuid2,
                SampleEvent.COMPLETE.value,
                10219,
                msgspec.json.encode({"output": ["And ", "goodbye."]}),
            ),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (
                uuid3,
                SampleEvent.COMPLETE.value,
                10250,
                msgspec.json.encode({"output": ["Extra ", "sample"]}),
            ),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        osl_rollup = reporter.get_output_sequence_lengths(tokenizer)

    assert len(osl_rollup) == 2
    assert uuid1 in osl_rollup
    assert uuid2 in osl_rollup
    assert uuid3 not in osl_rollup


def test_create_report_with_stop_perf(tmp_path, sample_uuids, tokenizer):
    """Test that create_report respects STOP_PERFORMANCE_TRACKING for all metrics."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    test_db = _create_db(
        tmp_path,
        "report_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
            (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
            (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
            (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
            (
                uuid1,
                SampleEvent.COMPLETE.value,
                10211,
                msgspec.json.encode({"output": ["Hello, ", "world"]}),
            ),
            (
                uuid2,
                SampleEvent.COMPLETE.value,
                10219,
                msgspec.json.encode({"output": ["And ", "goodbye."]}),
            ),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
            (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
            (uuid3, SampleEvent.FIRST_CHUNK.value, 10220, b""),
            (
                uuid3,
                SampleEvent.COMPLETE.value,
                10250,
                msgspec.json.encode({"output": ["Extra ", "sample"]}),
            ),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        report = reporter.create_report(tokenizer)

    assert report.n_samples_issued == 2
    assert report.n_samples_completed == 2
    assert report.duration_ns == 10219 - 5000

    expected_qps = 2 / ((10219 - 5000) / 1e9)
    assert report.qps == expected_qps

    assert report.latency["total"] == (10211 - 10000) + (10219 - 10003)


def test_report_zero_samples_before_stop(tmp_path):
    """Test that create_report shows 'Duration: N/A' when 0 samples issued before STOP_PERFORMANCE_TRACKING."""
    test_db = _create_db(
        tmp_path,
        "zero_samples_stop_perf.db",
        [
            ("", SessionEvent.TEST_STARTED.value, 5000, b""),
            ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
            ("", SessionEvent.TEST_ENDED.value, 10300, b""),
        ],
    )

    with MetricsReporter(test_db) as reporter:
        report = reporter.create_report()

    assert report.n_samples_issued == 0
    assert report.n_samples_completed == 0
    assert report.duration_ns is None

    buf = io.StringIO()

    def _write_with_newline(s):
        buf.write(s + "\n")

    report.display(fn=_write_with_newline)
    display_output = buf.getvalue()

    assert "Duration: N/A" in display_output
    assert "(no performance samples were issued)" in display_output
