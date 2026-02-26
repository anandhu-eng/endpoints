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

"""Unit tests for AsyncEventReporter — inflight tracking, idle notification, data wrapping."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from inference_endpoint.async_utils.transport.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.metrics.async_reporter import AsyncEventReporter


@pytest.fixture
def mock_publisher():
    pub = MagicMock()
    pub.publish = MagicMock()
    return pub


@pytest.fixture
def reporter(mock_publisher):
    return AsyncEventReporter(mock_publisher, "test_session_001")


class TestInflightTracking:
    def test_issued_increments(self, reporter, mock_publisher):
        assert reporter.n_inflight_samples == 0
        reporter.record_event(
            SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid="a"
        )
        assert reporter.n_inflight_samples == 1

    def test_complete_decrements(self, reporter, mock_publisher):
        reporter.record_event(
            SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid="a"
        )
        reporter.record_event(
            SampleEventType.COMPLETE, time.monotonic_ns(), sample_uuid="a"
        )
        assert reporter.n_inflight_samples == 0

    def test_multiple_inflight(self, reporter, mock_publisher):
        for i in range(5):
            reporter.record_event(
                SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid=str(i)
            )
        assert reporter.n_inflight_samples == 5
        for i in range(3):
            reporter.record_event(
                SampleEventType.COMPLETE, time.monotonic_ns(), sample_uuid=str(i)
            )
        assert reporter.n_inflight_samples == 2

    def test_session_events_dont_change_inflight(self, reporter, mock_publisher):
        reporter.record_event(SessionEventType.STARTED, time.monotonic_ns())
        assert reporter.n_inflight_samples == 0
        reporter.record_event(SessionEventType.ENDED, time.monotonic_ns())
        assert reporter.n_inflight_samples == 0


class TestIdleNotification:
    def test_idle_fires_when_inflight_hits_zero(self, mock_publisher):
        idle = asyncio.Event()
        r = AsyncEventReporter(mock_publisher, "s1", notify_idle=idle)
        r.should_check_idle = True

        r.record_event(SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid="a")
        assert not idle.is_set()

        r.record_event(SampleEventType.COMPLETE, time.monotonic_ns(), sample_uuid="a")
        assert idle.is_set()

    def test_idle_not_fired_when_check_disabled(self, mock_publisher):
        idle = asyncio.Event()
        r = AsyncEventReporter(mock_publisher, "s1", notify_idle=idle)
        # should_check_idle defaults to False
        r.record_event(SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid="a")
        r.record_event(SampleEventType.COMPLETE, time.monotonic_ns(), sample_uuid="a")
        assert not idle.is_set()

    def test_idle_not_fired_without_notify_event(self, mock_publisher):
        r = AsyncEventReporter(mock_publisher, "s1", notify_idle=None)
        r.should_check_idle = True
        r.record_event(SampleEventType.ISSUED, time.monotonic_ns(), sample_uuid="a")
        r.record_event(SampleEventType.COMPLETE, time.monotonic_ns(), sample_uuid="a")
        # Should not raise — just a no-op


class TestDataWrapping:
    def test_none_data_becomes_empty_dict(self, reporter, mock_publisher):
        reporter.record_event(SessionEventType.STARTED, time.monotonic_ns())
        record: EventRecord = mock_publisher.publish.call_args[0][0]
        assert record.data == {}

    def test_dict_data_passed_through(self, reporter, mock_publisher):
        reporter.record_event(
            SampleEventType.COMPLETE,
            time.monotonic_ns(),
            sample_uuid="a",
            data={"response": "ok"},
        )
        record: EventRecord = mock_publisher.publish.call_args[0][0]
        assert record.data == {"response": "ok"}

    def test_non_dict_data_wrapped_in_raw(self, reporter, mock_publisher):
        reporter.record_event(
            SampleEventType.COMPLETE,
            time.monotonic_ns(),
            sample_uuid="a",
            data="raw_string_value",
        )
        record: EventRecord = mock_publisher.publish.call_args[0][0]
        assert record.data == {"_raw": "raw_string_value"}


class TestPublish:
    def test_publishes_event_record(self, reporter, mock_publisher):
        ts = time.monotonic_ns()
        reporter.record_event(SampleEventType.ISSUED, ts, sample_uuid="uid1")
        mock_publisher.publish.assert_called_once()
        record: EventRecord = mock_publisher.publish.call_args[0][0]
        assert record.event_type == SampleEventType.ISSUED
        assert record.timestamp_ns == ts
        assert record.sample_uuid == "uid1"

    def test_connection_name(self, reporter):
        assert (
            str(reporter.connection_name)
            == "/dev/shm/mlperf_testsession_test_session_001.db"
        )
