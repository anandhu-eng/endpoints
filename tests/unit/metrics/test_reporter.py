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

import msgspec.json
import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.metrics.recorder import sqlite3_cursor
from inference_endpoint.metrics.reporter import MetricsReporter, TPOTReportingMode


def test_sample_counting(events_db):
    with MetricsReporter(events_db) as reporter:
        stats = reporter.get_sample_statuses()
        assert stats["completed"] == 2
        assert stats["in_flight"] == 1


def test_error_counting(events_db):
    """get_error_count returns distinct failed samples, not raw ERROR event count.

    The fixture has 3 ERROR events all belonging to uuid3, so the count should be 1.
    """
    with MetricsReporter(events_db) as reporter:
        assert reporter.get_error_count() == 1


def test_derive_ttft(events_db, sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        ttft_rows = reporter.derive_TTFT()
    assert len(ttft_rows) == 2
    assert ttft_rows[0].metric_type == "ttft"
    assert ttft_rows[1].metric_type == "ttft"
    assert ttft_rows.filter_uuid(uuid1, only_first=True) == 10
    assert ttft_rows.filter_uuid(uuid2, only_first=True) == 187
    assert ttft_rows.filter_uuid("asdf", only_first=True) is None
    assert ttft_rows.filter_uuid("asdf", only_first=False) == ()


def test_derive_tpot(events_db, sample_uuids, fake_outputs, tokenizer):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        tpot_rows = reporter.derive_TPOT(
            tokenizer, reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED
        )

    # From test_derive_sample_latency and ttft:
    expected_tpot1 = (10211 - 10000 - 10) / len(fake_outputs[uuid1][1])
    expected_tpot2 = (10219 - 10003 - 187) / len(fake_outputs[uuid2][1])

    tpot1 = tpot_rows.filter_uuid(uuid1, only_first=False)
    tpot2 = tpot_rows.filter_uuid(uuid2, only_first=False)
    assert len(tpot1) == len(fake_outputs[uuid1][1])
    assert len(tpot2) == len(fake_outputs[uuid2][1])
    assert all(tpot == expected_tpot1 for tpot in tpot1)
    assert all(tpot == expected_tpot2 for tpot in tpot2)


def test_derive_tpot_with_string_output(tmp_path, sample_uuids, tokenizer):
    """Test that derive_TPOT handles a plain string output gracefully.

    A single-string output has only one chunk, so TPOT cannot be computed.
    The reporter should not raise an exception and should return None.
    """
    test_db = str(tmp_path / "test_string_output.db")
    uuid1 = sample_uuids(1)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10211,
                    msgspec.json.encode({"output": "the final answer"}),
                ),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        tpot_rows = reporter.derive_TPOT(tokenizer)

    # A single-string output produces only 1 chunk — TPOT requires at least 2
    assert tpot_rows is None


def test_derive_tpot_with_list_reasoning(tmp_path, sample_uuids, tokenizer):
    """Test that derive_TPOT computes TPOT when string output is paired with a list reasoning sequence.

    The fix wraps string outputs into a single-element list so they can be combined with
    reasoning chunks. Without the fix, the string output causes the sample to be silently
    skipped before reasoning is considered, so TPOT returns None even though there are
    enough chunks (output + reasoning) to compute it.
    """
    test_db = str(tmp_path / "test_string_output_with_reasoning.db")
    uuid1 = sample_uuids(1)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10211,
                    msgspec.json.encode(
                        {"output": "the answer", "reasoning": ["thought step"]}
                    ),
                ),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        tpot_rows = reporter.derive_TPOT(tokenizer)

    # String output ("the answer") + list reasoning (["thought step"]) = 2 chunks total,
    # which is enough for TPOT computation.
    assert tpot_rows is not None
    assert len(tpot_rows) == 1


def test_derive_sample_latency(events_db, sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        sample_latency_rows = reporter.derive_sample_latency()

    assert len(sample_latency_rows) == 2
    latency1, latency2 = tuple(sorted(sample_latency_rows, key=lambda x: x.sample_uuid))
    assert latency1.metric_type == "sample_latency"
    assert latency1.sample_uuid == uuid1
    assert latency1.metric_value == 10211 - 10000

    assert latency2.metric_type == "sample_latency"
    assert latency2.sample_uuid == uuid2
    assert latency2.metric_value == 10219 - 10003


def test_derive_duration(events_db):
    with MetricsReporter(events_db) as reporter:
        duration = reporter.derive_duration()
    assert duration == (10300 - 5000)


def test_derive_duration_malformed(tmp_path):
    test_db_path = str(tmp_path / "bad_events.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("", SessionEvent.TEST_STARTED.value, 11000, b""),
                ("", SessionEvent.TEST_ENDED.value, 12000, b""),
            ],
        )
        conn.commit()

    with pytest.raises(
        RuntimeError, match=r"Multiple .*TEST_.* events found - 2 events"
    ):
        with MetricsReporter(test_db_path) as reporter:
            reporter.derive_duration()


@pytest.mark.parametrize(
    "case_desc, events, expected_duration, expect_raises",
    [
        (
            "multiple starts",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_STARTED.value, 6000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
            10300 - 6000,
            False,
        ),
        (
            "multiple ends",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("", SessionEvent.TEST_ENDED.value, 12000, b""),
            ],
            12000 - 5000,
            False,
        ),
        (
            "ended not last timestamp",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("some_uuid", SampleEvent.COMPLETE.value, 15000, b""),
            ],
            15000 - 5000,
            False,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_derive_duration_lenient(
    tmp_path, case_desc, events, expected_duration, expect_raises
):
    """Test derive_duration with check_malformed=False for various malformed scenarios."""
    test_db_path = str(tmp_path / f"{case_desc.replace(' ', '_')}.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            events,
        )
        conn.commit()

    with MetricsReporter(test_db_path) as reporter:
        duration = reporter.derive_duration(check_malformed=False)

    assert duration == expected_duration


def test_derive_duration_ended_not_last_raises(tmp_path):
    """Test that check_malformed=True raises when TEST_ENDED is not the max timestamp."""
    test_db_path = str(tmp_path / "test_ended_not_last.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("some_uuid", SampleEvent.COMPLETE.value, 15000, b""),
            ],
        )
        conn.commit()

    with pytest.raises(
        RuntimeError,
        match=r"TEST_ENDED exists .* but is not the maximum timestamp in database",
    ):
        with MetricsReporter(test_db_path) as reporter:
            reporter.derive_duration(check_malformed=True)
