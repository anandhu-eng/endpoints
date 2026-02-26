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

"""
Performance tests for cross-process event recording pipeline.

Measures end-to-end throughput of the AsyncEventRecorder → ZmqEventRecordPublisher
(main process) → EventWriterProcess (child process, SQLite at /dev/shm) pipeline.

This is the actual architecture used by _run_benchmark_async in benchmark.py.
The ablation script (scripts/ablation_event_recording.py) only measures single-process
ZMQ PUB throughput — this test measures the full cross-process path including SQLite writes.

Run with:
    pytest tests/performance/test_recorder_subprocess.py -v -s -m performance
"""

import asyncio
import gc
import sqlite3
import time
import uuid

import pytest
import uvloop
from inference_endpoint.async_utils.transport.record import (
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordPublisher
from inference_endpoint.metrics.recorder_subprocess import (
    AsyncEventRecorder,
    EventWriterProcess,
)

# =============================================================================
# Config
# =============================================================================

# Subscriber settle time — ZMQ PUB/SUB subscription propagation
SUB_SETTLE_S = 0.5


# =============================================================================
# Helpers
# =============================================================================


def _count_rows(db_path: str) -> int:
    """Count total rows in the events table."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]
    finally:
        conn.close()


def _count_rows_by_type(db_path: str) -> dict[str, int]:
    """Count rows grouped by event_type."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
        )
        return dict(cur.fetchall())
    finally:
        conn.close()


# =============================================================================
# Test: Unbatched cross-process throughput (current architecture)
# =============================================================================


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("n_samples", [10_000, 100_000, 500_000])
def test_cross_process_recorder_throughput(n_samples: int):
    """Measure cross-process event recording throughput (unbatched).

    Simulates the real benchmark hot path: for each sample, records
    DATA_LOAD + ISSUE_CALLED (sender side) and COMPLETE (receiver side).
    That's 3 events per sample — the minimum for non-streaming mode.

    Reports:
    - Publish rate (events/s from main process)
    - End-to-end rate (events written to SQLite / elapsed)
    - Per-event cost (ns/call)
    """

    async def _run() -> tuple[int, float, float]:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

        session_id = f"perf_test_{uuid.uuid4().hex[:8]}"

        with ManagedZMQContext.scoped(io_threads=4) as zmq_ctx:
            pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
            publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

            writer = EventWriterProcess(session_id, publisher.bind_address)
            writer.start(sub_settle_s=SUB_SETTLE_S)

            idle_event = asyncio.Event()
            recorder = AsyncEventRecorder(publisher, session_id, notify_idle=idle_event)

            # Suppress GC during measurement
            gc_was_enabled = gc.isenabled()
            gc.collect()
            gc.disable()

            # --- Hot loop: 3 events per sample ---
            recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

            start_ns = time.monotonic_ns()

            for _ in range(n_samples):
                sample_uuid = uuid.uuid4().hex
                ts = time.monotonic_ns()

                # Sender-side events
                recorder.record_event(
                    SampleEventType.ISSUED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.ISSUE_CALLED,
                    ts,
                    sample_uuid=sample_uuid,
                )

                # Receiver-side event (in real benchmark, this comes later)
                recorder.record_event(
                    SampleEventType.COMPLETE,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                    data={"response": "ok"},
                )

            publish_elapsed_ns = time.monotonic_ns() - start_ns

            # Restore GC
            if gc_was_enabled:
                gc.enable()

            # Signal end and wait for subscriber to finish writing
            recorder.record_event(
                SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
            )
            recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())
            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())

            # Allow time for ZMQ to deliver remaining messages
            time.sleep(1.0)
            writer.stop(timeout=30.0)

            total_elapsed_ns = time.monotonic_ns() - start_ns

            publisher.close()

        return n_samples, publish_elapsed_ns, total_elapsed_ns

    n_samples_actual, publish_ns, total_ns = uvloop.run(_run())

    n_events = n_samples_actual * 3  # 3 events per sample
    # +4 session events (STARTED, STOP_PERF, STOP_LOADGEN, ENDED)
    n_total_events = n_events + 4

    publish_s = publish_ns / 1e9
    total_s = total_ns / 1e9
    pub_rate = n_events / publish_s
    pub_ns_per_call = publish_ns / n_events
    e2e_rate = n_total_events / total_s

    print(
        f"\n  Cross-process recorder (unbatched):"
        f"\n    Samples:        {n_samples_actual:>10,}"
        f"\n    Events:         {n_events:>10,} ({n_events // n_samples_actual} per sample)"
        f"\n    Publish time:   {publish_s:>10.3f} s"
        f"\n    Publish rate:   {pub_rate:>10,.0f} events/s"
        f"\n    Per-event cost: {pub_ns_per_call:>10,.0f} ns/call"
        f"\n    E2E time:       {total_s:>10.3f} s"
        f"\n    E2E rate:       {e2e_rate:>10,.0f} events/s"
    )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("n_samples", [10_000, 100_000, 500_000])
def test_cross_process_recorder_throughput_with_verification(n_samples: int):
    """Same as above but also verifies all events land in SQLite.

    Slower due to SQLite read after benchmark, but confirms correctness.
    """
    session_id = f"perf_verify_{uuid.uuid4().hex[:8]}"

    async def _run() -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

        with ManagedZMQContext.scoped(io_threads=4) as zmq_ctx:
            pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
            publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

            writer = EventWriterProcess(session_id, publisher.bind_address)
            writer.start(sub_settle_s=SUB_SETTLE_S)

            idle_event = asyncio.Event()
            recorder = AsyncEventRecorder(publisher, session_id, notify_idle=idle_event)

            gc_was_enabled = gc.isenabled()
            gc.collect()
            gc.disable()

            recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

            start_ns = time.monotonic_ns()

            for _ in range(n_samples):
                sample_uuid = uuid.uuid4().hex
                ts = time.monotonic_ns()

                recorder.record_event(
                    SampleEventType.ISSUED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.ISSUE_CALLED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.COMPLETE,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                    data={"response": "ok"},
                )

            publish_elapsed_ns = time.monotonic_ns() - start_ns

            if gc_was_enabled:
                gc.enable()

            recorder.record_event(
                SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
            )
            recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())
            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())

            time.sleep(1.0)
            writer.stop(timeout=30.0)

            total_elapsed_ns = time.monotonic_ns() - start_ns

            publisher.close()

        return publish_elapsed_ns, total_elapsed_ns

    publish_ns, total_ns = uvloop.run(_run())

    n_events = n_samples * 3
    n_session_events = 4  # STARTED, STOP_PERF, STOP_LOADGEN, ENDED
    n_total_events = n_events + n_session_events

    publish_s = publish_ns / 1e9
    total_s = total_ns / 1e9
    pub_rate = n_events / publish_s
    pub_ns_per_call = publish_ns / n_events

    # Verify SQLite
    db_path = f"/dev/shm/mlperf_testsession_{session_id}.db"
    row_count = _count_rows(db_path)
    rows_by_type = _count_rows_by_type(db_path)

    print(
        f"\n  Cross-process recorder (with verification):"
        f"\n    Samples:          {n_samples:>10,}"
        f"\n    Events published: {n_total_events:>10,}"
        f"\n    Events in SQLite: {row_count:>10,}"
        f"\n    Publish rate:     {pub_rate:>10,.0f} events/s"
        f"\n    Per-event cost:   {pub_ns_per_call:>10,.0f} ns/call"
        f"\n    Total time:       {total_s:>10.3f} s"
        f"\n    Event breakdown:  {dict(rows_by_type)}"
    )

    # Correctness checks
    assert (
        row_count == n_total_events
    ), f"Expected {n_total_events} events in SQLite, got {row_count}"
    assert rows_by_type.get("loadgen_data_load", 0) == n_samples
    assert rows_by_type.get("loadgen_issue_called", 0) == n_samples
    assert rows_by_type.get("complete", 0) == n_samples
    assert rows_by_type.get("test_started", 0) == 1
    assert rows_by_type.get("test_ended", 0) == 1

    # Cleanup
    import os

    os.unlink(db_path)


# =============================================================================
# Test: Streaming mode (5 events per sample)
# =============================================================================


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("n_samples", [10_000, 100_000])
def test_cross_process_recorder_streaming_throughput(n_samples: int):
    """Measure throughput with streaming events (5 events/sample).

    Simulates: DATA_LOAD, ISSUE_CALLED, RECV_FIRST, RECV_NON_FIRST, COMPLETE.
    This is the typical per-sample event count for online streaming benchmarks.
    """
    session_id = f"perf_stream_{uuid.uuid4().hex[:8]}"

    async def _run() -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

        with ManagedZMQContext.scoped(io_threads=4) as zmq_ctx:
            pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
            publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

            writer = EventWriterProcess(session_id, publisher.bind_address)
            writer.start(sub_settle_s=SUB_SETTLE_S)

            idle_event = asyncio.Event()
            recorder = AsyncEventRecorder(publisher, session_id, notify_idle=idle_event)

            gc_was_enabled = gc.isenabled()
            gc.collect()
            gc.disable()

            recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

            start_ns = time.monotonic_ns()

            for _ in range(n_samples):
                sample_uuid = uuid.uuid4().hex
                ts = time.monotonic_ns()

                # Sender-side
                recorder.record_event(
                    SampleEventType.ISSUED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.ISSUE_CALLED,
                    ts,
                    sample_uuid=sample_uuid,
                )

                # Receiver-side (streaming)
                recorder.record_event(
                    SampleEventType.RECV_FIRST,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                    data={"response_chunk": "first"},
                )
                recorder.record_event(
                    SampleEventType.RECV_NON_FIRST,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.COMPLETE,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                    data={"response": "ok"},
                )

            publish_elapsed_ns = time.monotonic_ns() - start_ns

            if gc_was_enabled:
                gc.enable()

            recorder.record_event(
                SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
            )
            recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())
            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())

            time.sleep(1.0)
            writer.stop(timeout=30.0)

            total_elapsed_ns = time.monotonic_ns() - start_ns
            publisher.close()

        return publish_elapsed_ns, total_elapsed_ns

    publish_ns, total_ns = uvloop.run(_run())

    events_per_sample = 5
    n_events = n_samples * events_per_sample
    n_session_events = 4
    n_total_events = n_events + n_session_events

    publish_s = publish_ns / 1e9
    total_s = total_ns / 1e9
    pub_rate = n_events / publish_s
    pub_ns_per_call = publish_ns / n_events

    # Verify SQLite
    db_path = f"/dev/shm/mlperf_testsession_{session_id}.db"
    row_count = _count_rows(db_path)

    print(
        f"\n  Cross-process recorder (streaming, {events_per_sample} events/sample):"
        f"\n    Samples:          {n_samples:>10,}"
        f"\n    Events published: {n_total_events:>10,}"
        f"\n    Events in SQLite: {row_count:>10,}"
        f"\n    Publish rate:     {pub_rate:>10,.0f} events/s"
        f"\n    Per-event cost:   {pub_ns_per_call:>10,.0f} ns/call"
        f"\n    Total time:       {total_s:>10.3f} s"
    )

    assert (
        row_count == n_total_events
    ), f"Expected {n_total_events} events in SQLite, got {row_count}"

    # Cleanup
    import os

    os.unlink(db_path)


# =============================================================================
# Test: Implied QPS capacity
# =============================================================================


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_implied_qps_capacity():
    """Calculate the max QPS supported by event recording overhead.

    At 3 events/sample (non-streaming), the max QPS is:
        max_qps = publish_rate / 3

    This test runs a quick measurement and reports the implied QPS ceiling.
    This is the theoretical max before event recording becomes the bottleneck.
    """
    n_samples = 100_000
    session_id = f"perf_qps_{uuid.uuid4().hex[:8]}"

    async def _run() -> float:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

        with ManagedZMQContext.scoped(io_threads=4) as zmq_ctx:
            pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
            publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

            writer = EventWriterProcess(session_id, publisher.bind_address)
            writer.start(sub_settle_s=SUB_SETTLE_S)

            recorder = AsyncEventRecorder(publisher, session_id)

            gc_was_enabled = gc.isenabled()
            gc.collect()
            gc.disable()

            recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

            start_ns = time.monotonic_ns()

            for _ in range(n_samples):
                sample_uuid = uuid.uuid4().hex
                ts = time.monotonic_ns()

                recorder.record_event(
                    SampleEventType.ISSUED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.ISSUE_CALLED,
                    ts,
                    sample_uuid=sample_uuid,
                )
                recorder.record_event(
                    SampleEventType.COMPLETE,
                    time.monotonic_ns(),
                    sample_uuid=sample_uuid,
                    data={"response": "ok"},
                )

            elapsed_ns = time.monotonic_ns() - start_ns

            if gc_was_enabled:
                gc.enable()

            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
            time.sleep(1.0)
            writer.stop(timeout=30.0)
            publisher.close()

        return elapsed_ns

    elapsed_ns = uvloop.run(_run())

    n_events = n_samples * 3
    elapsed_s = elapsed_ns / 1e9
    events_per_s = n_events / elapsed_s
    ns_per_event = elapsed_ns / n_events
    implied_max_qps_3ev = events_per_s / 3
    implied_max_qps_5ev = events_per_s / 5

    print(
        f"\n  Implied QPS capacity from event recording:"
        f"\n    Publish rate:     {events_per_s:>10,.0f} events/s"
        f"\n    Per-event cost:   {ns_per_event:>10,.0f} ns"
        f"\n    Max QPS (3 ev/s): {implied_max_qps_3ev:>10,.0f} (non-streaming)"
        f"\n    Max QPS (5 ev/s): {implied_max_qps_5ev:>10,.0f} (streaming)"
        f"\n"
        f"\n    Note: Real benchmark has additional overhead (UUID gen, data load,"
        f"\n    HTTP client send/recv) so actual achievable QPS is lower."
    )

    # Cleanup
    import os

    db_path = f"/dev/shm/mlperf_testsession_{session_id}.db"
    if os.path.exists(db_path):
        os.unlink(db_path)
