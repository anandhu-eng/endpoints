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

"""Background process for SQLite event recording via ZMQ PUB/SUB.

Replaces the EventRecorder's writer thread with a separate OS process.
Uses the existing ZmqEventRecordSubscriber infrastructure from async_utils.

The main process publishes EventRecords via ZmqEventRecordPublisher (ZMQ PUB,
non-blocking). This module runs a subscriber in a separate OS process that
receives events and batch-writes them to SQLite at /dev/shm.

The SQLite schema is identical to EventRecorder's, ensuring full compatibility
with MetricsReporter. A mapping table converts pub-sub EventType topics to
the old event_type string values expected by MetricsReporter.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.synchronize
import sqlite3
from pathlib import Path

import orjson
import uvloop

from inference_endpoint.async_utils.transport.record import (
    EventRecord,
    SessionEventType,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordSubscriber,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EventType topic → old event_type value for SQLite (MetricsReporter compat)
# ---------------------------------------------------------------------------

_TOPIC_TO_SQLITE_EVENT_TYPE: dict[str, str] = {
    # Session events
    "session.started": "test_started",
    "session.ended": "test_ended",
    "session.stop_loadgen": "loadgen_stop",
    "session.stop_performance_tracking": "stop_performance_tracking",
    # Sample events
    "sample.issued": "loadgen_issue_called",
    "sample.complete": "complete",
    "sample.recv_first": "first_chunk_received",
    "sample.recv_non_first": "non_first_chunk_received",
    "sample.client_send": "http_request_issued",
    "sample.client_resp_done": "http_response_completed",
    "sample.transport_sent": "zmq_response_sent",
    "sample.transport_recv": "zmq_request_received",
    # Error events
    "error.generic": "error",
    "error.loadgen": "error",
    "error.session": "error",
    "error.client": "error",
}

# SQLite schema (same as EventRecorder)
_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS events ("
    "sample_uuid TEXT, event_type TEXT, timestamp_ns INTEGER, data BLOB)"
)
_INSERT = "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)"


# ---------------------------------------------------------------------------
# SQLite Writer Subscriber
# ---------------------------------------------------------------------------


class _SqliteWriterSubscriber(ZmqEventRecordSubscriber):
    """Subscriber that writes EventRecords to SQLite, compatible with MetricsReporter.

    Runs in a background process with its own uvloop. Processes received
    EventRecords by converting them to the SQLite format and batch-inserting.
    """

    def __init__(
        self,
        db_path: str,
        txn_buffer_size: int,
        done_event: asyncio.Event,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self.txn_buffer_size = txn_buffer_size
        self._done_event = done_event

        # Open SQLite connection
        self._conn = sqlite3.connect(db_path)
        self._cur = self._conn.cursor()
        self._cur.execute(_CREATE_TABLE)
        self._conn.commit()

        self._sql_buffer: list[tuple[str, str, int, bytes]] = []

    def _commit_buffer(self) -> None:
        if self._sql_buffer:
            self._cur.executemany(_INSERT, self._sql_buffer)
            self._conn.commit()
            self._sql_buffer.clear()

    async def process(self, records: list[EventRecord]) -> None:
        """Process received EventRecords — convert and buffer for SQLite."""
        for record in records:
            topic: str = record.event_type.topic  # type: ignore[attr-defined]
            sqlite_event_type = _TOPIC_TO_SQLITE_EVENT_TYPE.get(topic) or topic

            # Serialize data matching old EventRecorder format:
            # - If data has "_raw" key, serialize the raw value directly
            #   (this handles non-dict data like strings passed via record_event)
            # - Otherwise serialize the full dict
            data_bytes = b""
            if record.data:
                raw = record.data.get("_raw")
                if raw is not None:
                    data_bytes = orjson.dumps(raw)
                else:
                    data_bytes = orjson.dumps(record.data)

            self._sql_buffer.append(
                (record.sample_uuid, sqlite_event_type, record.timestamp_ns, data_bytes)
            )

            # Check for session ended → signal done after flush
            if record.event_type == SessionEventType.ENDED:
                self._commit_buffer()
                self._done_event.set()
                return

        if len(self._sql_buffer) >= self.txn_buffer_size:
            self._commit_buffer()

    def close(self) -> None:
        """Flush remaining events, create indexes for fast reads, and close."""
        if not self.is_closed:
            self._commit_buffer()
            # Create indexes after all writes — speeds up MetricsReporter queries
            try:
                self._cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)"
                )
                self._cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sample_uuid ON events(sample_uuid)"
                )
                self._cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_type_uuid ON events(event_type, sample_uuid)"
                )
                self._conn.commit()
            except Exception:
                pass  # non-critical — queries still work, just slower
            self._cur.close()
            self._conn.close()
        super().close()


# ---------------------------------------------------------------------------
# Background process entry point
# ---------------------------------------------------------------------------


def _subscriber_main(
    publisher_address: str,
    db_path: str,
    txn_buffer_size: int,
    ready_event: multiprocessing.synchronize.Event | None = None,
) -> None:
    """Entry point for the background subscriber process.

    Creates a uvloop, connects a subscriber to the publisher's address,
    and writes events to SQLite until SESSION_ENDED is received.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent controls lifecycle

    async def _run():
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)

        done_event = asyncio.Event()

        # Reset ManagedZMQContext singleton inherited from parent via fork().
        # The forked singleton holds a stale ZMQ context that doesn't work
        # correctly in the child process.
        ManagedZMQContext._instance = None
        zmq_ctx = ManagedZMQContext(io_threads=1)

        subscriber = _SqliteWriterSubscriber(
            db_path=db_path,
            txn_buffer_size=txn_buffer_size,
            done_event=done_event,
            connect_address=publisher_address,
            zmq_context=zmq_ctx,
            loop=loop,
            topics=None,  # Subscribe to all topics
        )

        # Start receiving
        subscriber.start()

        # Signal readiness to the main process
        if ready_event is not None:
            ready_event.set()

        # Wait for SESSION_ENDED or timeout
        try:
            await asyncio.wait_for(done_event.wait(), timeout=3600)
        except TimeoutError:
            logger.warning("Subscriber timed out waiting for SESSION_ENDED")
        finally:
            subscriber.close()
            zmq_ctx.cleanup()

    uvloop.run(_run())


# ---------------------------------------------------------------------------
# Process manager (used by the main process)
# ---------------------------------------------------------------------------


class AsyncEventRecorder:
    """Manages a background process that subscribes to EventPublisherService
    and writes events to SQLite.

    Usage (context manager):
        with AsyncEventRecorder(session_id, publisher.bind_address, sub_settle_s=0.5) as writer:
            recorder = AsyncEventReporter(publisher, session_id)
            # ... benchmark runs, publisher.publish() events ...
            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())

    Usage (manual):
        writer = AsyncEventRecorder(session_id, publisher.bind_address)
        writer.start(sub_settle_s=0.5)
        # ... benchmark runs, publisher.publish() events ...
        writer.stop()
    """

    def __init__(
        self,
        session_id: str,
        publisher_address: str,
        txn_buffer_size: int = 1000,
        sub_settle_s: float = 0.3,
        start_timeout: float = 10.0,
        stop_timeout: float = 10.0,
    ):
        self.session_id = session_id
        self.publisher_address = publisher_address
        self.txn_buffer_size = txn_buffer_size
        self.sub_settle_s = sub_settle_s
        self.start_timeout = start_timeout
        self.stop_timeout = stop_timeout
        self.db_path = Path(f"/dev/shm/mlperf_testsession_{session_id}.db")
        self._process: multiprocessing.Process | None = None

    def __enter__(self) -> AsyncEventRecorder:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # On error, use a short timeout — the subscriber may not have received
        # SESSION_ENDED, so a long wait would just delay the inevitable kill.
        timeout = self.stop_timeout if exc_type is None else 2.0
        self.stop(timeout=timeout)

    def start(
        self, timeout: float | None = None, sub_settle_s: float | None = None
    ) -> None:
        """Start the background subscriber process.

        Blocks until the subscriber signals readiness (connected and subscribed),
        then waits for ZMQ PUB/SUB subscription to propagate.

        NOTE: This uses blocking waits (not async) because ZMQ's PUB/SUB handshake
        requires the I/O threads to run uninterrupted. Called once during setup,
        not on the hot path.

        Args:
            timeout: Max seconds to wait for subscriber readiness (default: self.start_timeout).
            sub_settle_s: Extra blocking sleep for ZMQ subscription propagation (default: self.sub_settle_s).

        Raises:
            TimeoutError: If subscriber doesn't become ready in time.
        """
        import time

        timeout = timeout if timeout is not None else self.start_timeout
        sub_settle_s = sub_settle_s if sub_settle_s is not None else self.sub_settle_s

        ready_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_subscriber_main,
            args=(
                self.publisher_address,
                str(self.db_path),
                self.txn_buffer_size,
                ready_event,
            ),
            daemon=True,
            name=f"EventWriter-{self.session_id[:8]}",
        )
        self._process.start()

        # Blocking wait for subprocess to connect and subscribe
        if not ready_event.wait(timeout=timeout):
            if self._process.is_alive():
                self._process.kill()
            raise TimeoutError(
                f"AsyncEventRecorder did not become ready within {timeout}s"
            )

        # Blocking sleep for ZMQ PUB/SUB subscription propagation
        time.sleep(sub_settle_s)

        logger.debug(
            f"AsyncEventRecorder started (pid={self._process.pid}, "
            f"db={self.db_path})"
        )

    def stop(self, timeout: float | None = None) -> None:
        """Wait for the background process to finish (it stops on SESSION_ENDED)."""
        timeout = timeout if timeout is not None else self.stop_timeout
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("AsyncEventRecorder did not stop, killing")
                self._process.kill()
                self._process.join(timeout=2.0)
        logger.debug("AsyncEventRecorder stopped")

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
