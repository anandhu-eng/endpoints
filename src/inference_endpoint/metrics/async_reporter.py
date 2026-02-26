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

"""Lightweight event reporter for the single-loop benchmark architecture.

Publishes EventRecords via ZmqEventRecordPublisher (sync ZMQ PUB NOBLOCK).
The background AsyncEventRecorder (in async_recorder.py) subscribes and writes
to SQLite, keeping the same schema for MetricsReporter compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_endpoint.async_utils.transport.record import (
    EventRecord,
    EventType,
    SampleEventType,
)

if TYPE_CHECKING:
    from inference_endpoint.async_utils.transport.zmq.pubsub import (
        ZmqEventRecordPublisher,
    )

logger = logging.getLogger(__name__)


class AsyncEventReporter:
    """Event reporter for the single-loop benchmark architecture.

    Uses EventType (from async_utils/transport/record.py) natively — no
    compatibility layer with the old Event enum. Publishes EventRecords
    via ZmqEventRecordPublisher (sync ZMQ PUB NOBLOCK).

    The background AsyncEventRecorder subscribes and writes to SQLite,
    keeping the same schema for MetricsReporter compatibility.

    Usage:
        publisher = ZmqEventRecordPublisher(addr, zmq_ctx, loop=loop)

        with AsyncEventRecorder(session_id, publisher.bind_address) as writer:
            reporter = AsyncEventReporter(publisher, session_id)
            reporter.record_event(SessionEventType.STARTED, time.monotonic_ns())
            reporter.record_event(SampleEventType.ISSUED, ts, sample_uuid=uid)
            reporter.record_event(SampleEventType.COMPLETE, ts, sample_uuid=uid)
            reporter.record_event(SessionEventType.ENDED, time.monotonic_ns())
    """

    __slots__ = (
        "_publisher",
        "session_id",
        "n_inflight_samples",
        "should_check_idle",
        "notify_idle",
    )

    def __init__(
        self,
        publisher: ZmqEventRecordPublisher,
        session_id: str,
        notify_idle: asyncio.Event | None = None,
    ):
        from inference_endpoint.async_utils.transport.zmq.pubsub import (
            ZmqEventRecordPublisher as _Pub,
        )

        self._publisher: _Pub = publisher
        self.session_id = session_id
        self.n_inflight_samples: int = 0
        self.should_check_idle: bool = False
        self.notify_idle = notify_idle

    @property
    def connection_name(self) -> Path:
        """SQLite database path (same as EventRecorder)."""
        return Path(f"/dev/shm/mlperf_testsession_{self.session_id}.db")

    def record_event(
        self,
        ev_type: EventType,
        timestamp_ns: int,
        sample_uuid: str = "",
        data: Any = None,
    ) -> None:
        """Record an event by publishing via ZMQ PUB (sync, non-blocking).

        Args:
            ev_type: EventType from async_utils/transport/record.py
                     (SessionEventType, SampleEventType, ErrorEventType).
            timestamp_ns: Monotonic nanosecond timestamp.
            sample_uuid: UUID of the sample (empty for session-level events).
            data: Optional event data (Any — serialized by subscriber, matching
                  old EventRecorder behavior).
        """
        # Update inflight sample tracking
        if ev_type == SampleEventType.ISSUED:
            self.n_inflight_samples += 1
        elif ev_type == SampleEventType.COMPLETE:
            self.n_inflight_samples -= 1

        # Wrap non-dict data for EventRecord.data (which requires dict)
        event_data: dict[str, Any]
        if data is None:
            event_data = {}
        elif isinstance(data, dict):
            event_data = data
        else:
            event_data = {"_raw": data}

        self._publisher.publish(
            EventRecord(
                event_type=ev_type,
                timestamp_ns=timestamp_ns,
                sample_uuid=sample_uuid,
                data=event_data,
            )
        )

        # Check idle
        if (
            self.should_check_idle
            and self.notify_idle is not None
            and self.n_inflight_samples == 0
        ):
            self.notify_idle.set()
