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

"""Unit tests for EventRecord and related types (serialization / deserialization)."""

import time

import pytest
from inference_endpoint.core.record import (
    TOPIC_FRAME_SIZE,
    ErrorEventType,
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
    decode_event_record,
    encode_event_record,
)
from inference_endpoint.core.types import ErrorData, PromptData, TextModelOutput


class TestEventType:
    def test_category_base_raises(self):
        with pytest.raises(AttributeError):
            EventType.category()

    @pytest.mark.parametrize(
        "subclass, expected_category",
        [
            (SessionEventType, "session"),
            (ErrorEventType, "error"),
            (SampleEventType, "sample"),
        ],
    )
    def test_subclass_category(self, subclass, expected_category):
        assert subclass.category() == expected_category

    @pytest.mark.parametrize(
        "event, expected_topic, expected_value",
        [
            (SessionEventType.STARTED, "session.started", "started"),
            (SampleEventType.COMPLETE, "sample.complete", "complete"),
            (ErrorEventType.GENERIC, "error.generic", "generic"),
        ],
    )
    def test_topic_and_value(self, event, expected_topic, expected_value):
        assert event.topic == expected_topic
        assert event.value == expected_value
        assert isinstance(event, EventType)


class TestEventRecordConstruction:
    def test_construction_defaults(self):
        before = time.monotonic_ns()
        record = EventRecord(event_type=SessionEventType.STARTED)
        after = time.monotonic_ns()
        assert before <= record.timestamp_ns <= after
        assert record.sample_uuid == ""
        assert record.data is None


class TestEncodeEventRecord:
    def test_encode_returns_topic_and_payload(self):
        """encode_event_record returns (topic_bytes_padded, payload) for single-frame ZMQ."""
        data = TextModelOutput(output="test-output")
        record = EventRecord(
            event_type=SampleEventType.ISSUED,
            sample_uuid="test-uuid",
            data=data,
        )
        topic_bytes, payload = encode_event_record(record)
        assert isinstance(topic_bytes, bytes)
        assert len(topic_bytes) == TOPIC_FRAME_SIZE
        assert topic_bytes.rstrip(b"\x00") == b"sample.issued"
        assert isinstance(payload, bytes)
        decoded = decode_event_record(payload)
        assert decoded.sample_uuid == "test-uuid"
        assert decoded.data == data

    @pytest.mark.parametrize(
        "event, expected_topic",
        [
            (SessionEventType.STARTED, "session.started"),
            (SessionEventType.ENDED, "session.ended"),
            (SampleEventType.COMPLETE, "sample.complete"),
            (ErrorEventType.GENERIC, "error.generic"),
        ],
    )
    def test_topic_bytes_padding(self, event, expected_topic):
        """Topic is null-padded to TOPIC_FRAME_SIZE."""
        topic_bytes, _ = encode_event_record(EventRecord(event_type=event))
        assert len(topic_bytes) == TOPIC_FRAME_SIZE
        assert topic_bytes.rstrip(b"\x00") == expected_topic.encode("utf-8")


class TestEventRecordRoundTrip:
    @pytest.mark.parametrize(
        "case_desc, event_type, uuid, data",
        [
            ("session no data", SessionEventType.STARTED, "sess-1", None),
            (
                "sample with output",
                SampleEventType.COMPLETE,
                "sample-42",
                TextModelOutput(output="output text"),
            ),
            (
                "sample with reasoning",
                SampleEventType.COMPLETE,
                "sample-42",
                TextModelOutput(output="out", reasoning="reason"),
            ),
            (
                "prompt data text",
                SampleEventType.ISSUED,
                "sample-99",
                PromptData(text="What is AI?"),
            ),
            (
                "prompt data tokens",
                SampleEventType.ISSUED,
                "sample-100",
                PromptData(token_ids=(101, 202, 303)),
            ),
            (
                "error data",
                ErrorEventType.LOADGEN,
                "",
                ErrorData(error_type="LoadgenError", error_message="error details"),
            ),
            ("defaults only", SessionEventType.ENDED, "", None),
        ],
    )
    def test_round_trip(self, case_desc, event_type, uuid, data):
        record = EventRecord(event_type=event_type, sample_uuid=uuid, data=data)
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == event_type.topic
        assert decoded.sample_uuid == uuid
        assert decoded.data == data
        assert decoded.timestamp_ns == record.timestamp_ns

    def test_explicit_timestamp_preserved(self):
        ts = 1234567890
        record = EventRecord(event_type=SampleEventType.ISSUED, timestamp_ns=ts)
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.timestamp_ns == ts
