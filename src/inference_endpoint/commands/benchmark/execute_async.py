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

"""Async benchmark runner — single uvloop, no threads in the main process.

Architecture:
  - HTTPEndpointClient on the running loop (no separate loop thread)
  - ZmqEventRecordPublisher for non-blocking event publishing
  - AsyncEventRecorder in a background process for SQLite writes
  - Scheduler.__aiter__() for drift-correcting online timing
  - Unified receiver: await recv() wakeup + poll() drain
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from tqdm import tqdm

from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordPublisher,
)
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    LoadPatternType,
    SystemDefaults,
)
from inference_endpoint.core.record import SampleEventType, SessionEventType
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.exceptions import ExecutionError
from inference_endpoint.load_generator.scheduler import (
    Scheduler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.metrics.async_recorder import AsyncEventRecorder
from inference_endpoint.metrics.async_reporter import AsyncEventReporter
from inference_endpoint.metrics.reporter import MetricsReporter

from .execute import BenchmarkContext, ResponseCollector

logger = logging.getLogger(__name__)

_NO_RECORD = os.environ.get("NO_RECORD", "")


# ── Runtime state ────────────────────────────────────────────────────────


@dataclass
class _BenchmarkRuntime:
    """Mutable state shared across sender/receiver coroutines."""

    http_client: HTTPEndpointClient
    recorder: AsyncEventReporter
    scheduler: Scheduler
    collector: ResponseCollector
    dataloader: Dataset
    uuid_to_index: dict[str, int] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    send_done: bool = False
    send_n: int = 0
    stop_requested: bool = False

    def issue_sample(self, s_idx: int, ds: Dataset, uuid_map: dict[str, int]) -> None:
        sample_uuid = self.rng.randbytes(16).hex()
        sample_data = ds.load_sample(s_idx)
        if not _NO_RECORD:
            self.recorder.record_event(
                SampleEventType.ISSUED,
                time.monotonic_ns(),
                sample_uuid=sample_uuid,
            )
        self.http_client.issue(Query(id=sample_uuid, data=sample_data))
        uuid_map[sample_uuid] = s_idx

    def handle_response(self, result: QueryResult | StreamChunk) -> None:
        ts = time.monotonic_ns()
        if isinstance(result, StreamChunk):
            ev = (
                SampleEventType.RECV_FIRST
                if (result.metadata or {}).get("first_chunk", False)
                else SampleEventType.RECV_NON_FIRST
            )
            self.recorder.record_event(ev, ts, sample_uuid=result.id)
        elif isinstance(result, QueryResult):
            if result.error is not None:
                logger.error(f"Error in request {result.id}: {result.error}")
            self.recorder.record_event(
                SampleEventType.COMPLETE,
                ts,
                sample_uuid=result.id,
                data=result.response_output,
            )
            self.collector.on_complete_hook(result)
            self.scheduler.notify_complete()


# ── Sender factories ────────────────────────────────────────────────────


def _make_sender(
    rt: _BenchmarkRuntime,
    load_pattern: LoadPatternType,
) -> Any:
    """Return a sender coroutine matched to the load pattern."""

    if load_pattern == LoadPatternType.MAX_THROUGHPUT:

        async def _sender() -> None:
            sent = 0
            for s_idx, _ in rt.scheduler:
                if rt.stop_requested:
                    break
                rt.issue_sample(s_idx, rt.dataloader, rt.uuid_to_index)
                sent += 1
                if sent % 1000 == 0:
                    await asyncio.sleep(0)
            rt.send_n = sent
            rt.send_done = True

    elif load_pattern == LoadPatternType.CONCURRENCY:

        async def _sender() -> None:
            sent = 0
            async for s_idx in rt.scheduler:
                if rt.stop_requested:
                    break
                rt.issue_sample(s_idx, rt.dataloader, rt.uuid_to_index)
                sent += 1
                if sent % 1000 == 0:
                    await asyncio.sleep(0)
            rt.send_n = sent
            rt.send_done = True

    else:
        # Poisson: scheduler.__aiter__ handles timing
        async def _sender() -> None:
            sent = 0
            async for s_idx in rt.scheduler:
                if rt.stop_requested:
                    break
                rt.issue_sample(s_idx, rt.dataloader, rt.uuid_to_index)
                sent += 1
            rt.send_n = sent
            rt.send_done = True

    return _sender


# ── Receiver ─────────────────────────────────────────────────────────────


async def _receiver(rt: _BenchmarkRuntime) -> None:
    """Unified receiver: async recv wakeup + sync poll drain."""
    recv_n = 0
    while True:
        result = await rt.http_client.recv()
        if result is None:
            break
        if _NO_RECORD:
            recv_n += 1
            rt.scheduler.notify_complete()
            while rt.http_client.poll() is not None:
                recv_n += 1
                rt.scheduler.notify_complete()
        else:
            rt.handle_response(result)
            while (r := rt.http_client.poll()) is not None:
                rt.handle_response(r)
        if rt.send_done and (
            (_NO_RECORD and recv_n >= rt.send_n)
            or (not _NO_RECORD and rt.recorder.n_inflight_samples <= 0)
            or rt.stop_requested
        ):
            break


# ── Accuracy phase ───────────────────────────────────────────────────────


async def _run_accuracy_phase(
    rt: _BenchmarkRuntime,
    acc_ds: Dataset,
    acc_scheduler: Scheduler,
) -> None:
    """Run one accuracy dataset: max-throughput sender + receiver."""
    acc_uuid_map: dict[str, int] = {}
    done = False

    async def sender() -> None:
        nonlocal done
        sent = 0
        for s_idx, _ in acc_scheduler:
            if rt.stop_requested:
                break
            rt.issue_sample(s_idx, acc_ds, acc_uuid_map)
            sent += 1
            if sent % 1000 == 0:
                await asyncio.sleep(0)
        done = True

    async def receiver() -> None:
        while True:
            result = await rt.http_client.recv()
            if result is None:
                break
            rt.handle_response(result)
            while (r := rt.http_client.poll()) is not None:
                rt.handle_response(r)
            if done and rt.recorder.n_inflight_samples <= 0:
                break

    await asyncio.gather(sender(), receiver())


# ── Resource setup / teardown ────────────────────────────────────────────


def _build_http_config(ctx: BenchmarkContext) -> HTTPClientConfig:
    config = ctx.config
    api_type: APIType = config.endpoint_config.api_type
    return HTTPClientConfig(
        endpoint_urls=[
            urljoin(e, api_type.default_route())
            for e in config.endpoint_config.endpoints
        ],
        api_type=api_type,
        num_workers=config.settings.client.workers,
        record_worker_events=config.settings.client.record_worker_events,
        event_logs_dir=ctx.report_dir,
        log_level=config.settings.client.log_level,
        cpu_affinity=ctx.affinity_plan,
        warmup_connections=config.settings.client.warmup_connections,
        max_connections=config.settings.client.max_connections,
        api_key=config.endpoint_config.api_key,
    )


def _generate_report(
    recorder: AsyncEventReporter,
    ctx: BenchmarkContext,
) -> Any:
    """Read SQLite and generate metrics report."""
    try:
        with MetricsReporter(
            recorder.connection_name, client_type="sqlite"
        ) as reporter:
            report = reporter.create_report(ctx.tokenizer)
            report.display(fn=print, summary_only=True)
            if ctx.report_dir:
                report.to_json(save_to=Path(ctx.report_dir) / "result_summary.json")
                with open(Path(ctx.report_dir) / "report.txt", "w") as f:
                    report.display(fn=f.write, summary_only=False, newline="\n")
                reporter.dump_to_json(Path(ctx.report_dir) / "events.jsonl")
                logger.info(f"Report saved to: {ctx.report_dir}")
            return report
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return None


def _cleanup(
    loop: asyncio.AbstractEventLoop,
    pbar: tqdm | None,
    recorder: AsyncEventReporter | None,
    writer: AsyncEventRecorder | None,
    http_client: HTTPEndpointClient | None,
    publisher: ZmqEventRecordPublisher | None,
    zmq_ctx: ManagedZMQContext | None,
    session_ended: bool,
) -> None:
    """Best-effort cleanup — each step guarded individually."""
    try:
        loop.remove_signal_handler(signal.SIGINT)
    except Exception:
        pass

    if pbar:
        try:
            pbar.close()
        except Exception:
            pass

    if recorder and not session_ended:
        try:
            recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
        except Exception:
            pass

    if writer:
        try:
            writer.stop()
        except Exception:
            pass

    if http_client:
        try:
            http_client.shutdown()
        except Exception:
            pass

    try:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except (OSError, AttributeError):
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if publisher:
        try:
            publisher.close()
        except Exception:
            pass

    if zmq_ctx:
        try:
            zmq_ctx.cleanup()
        except Exception:
            pass


# ── Main entry point ─────────────────────────────────────────────────────


async def run_benchmark_async(
    ctx: BenchmarkContext,
) -> tuple[Any, ResponseCollector]:
    """Execute benchmark on a single uvloop — no threads in the main process."""
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

    config = ctx.config
    zmq_ctx: ManagedZMQContext | None = None
    http_client: HTTPEndpointClient | None = None
    publisher: ZmqEventRecordPublisher | None = None
    writer: AsyncEventRecorder | None = None
    recorder: AsyncEventReporter | None = None
    pbar: tqdm | None = None
    collector = ResponseCollector(collect_responses=ctx.collect_responses)
    session_ended = False
    report = None

    try:
        # ── Resources ────────────────────────────────────────────────
        zmq_ctx = ManagedZMQContext(io_threads=4)

        logger.info(f"Connecting: {config.endpoint_config.endpoints}")
        http_client = await asyncio.to_thread(
            HTTPEndpointClient,
            _build_http_config(ctx),
            loop=loop,
            zmq_context=zmq_ctx,
        )

        session_id = f"cli_benchmark_{uuid.uuid4().hex[:8]}"
        pub_socket_name = f"ev_pub_{session_id[:8]}"
        publisher = ZmqEventRecordPublisher(pub_socket_name, zmq_ctx, loop=loop)

        writer = AsyncEventRecorder(
            session_id, publisher.bind_address, sub_settle_s=0.5, stop_timeout=5.0
        )
        writer.start()

        idle_event = asyncio.Event()
        recorder = AsyncEventReporter(publisher, session_id, notify_idle=idle_event)

        pbar = tqdm(
            desc=f"{config.model_params.name} (Streaming: {ctx.enable_streaming})",
            total=ctx.total_samples,
            smoothing=0,
        )
        collector = ResponseCollector(
            collect_responses=ctx.collect_responses, pbar=pbar
        )

        rt = _BenchmarkRuntime(
            http_client=http_client,
            recorder=recorder,
            scheduler=ctx.scheduler,
            collector=collector,
            dataloader=ctx.dataloader,
        )

        # ── SIGINT ───────────────────────────────────────────────────
        def on_sigint() -> None:
            logger.warning("Interrupt received, stopping benchmark...")
            rt.stop_requested = True
            rt.send_done = True

        loop.add_signal_handler(signal.SIGINT, on_sigint)

        # ── Performance phase ────────────────────────────────────────
        recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

        sender_coro = _make_sender(rt, config.settings.load_pattern.type)
        logger.info("Running...")
        await asyncio.gather(sender_coro(), _receiver(rt))
        loop.remove_signal_handler(signal.SIGINT)

        recorder.record_event(
            SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
        )
        logger.info("All performance samples issued")

        # ── Accuracy phases ──────────────────────────────────────────
        if ctx.accuracy_datasets and not rt.stop_requested:
            for acc_ds in ctx.accuracy_datasets:
                ds_name = getattr(
                    acc_ds.__class__, "DATASET_ID", acc_ds.__class__.__name__
                )
                acc_rt = RuntimeSettings(
                    metric_target=ctx.rt_settings.metric_target,
                    reported_metrics=ctx.rt_settings.reported_metrics,
                    min_duration_ms=0,
                    max_duration_ms=None,
                    n_samples_from_dataset=acc_ds.num_samples(),
                    n_samples_to_issue=acc_ds.num_samples() * acc_ds.repeats,
                    min_sample_count=acc_ds.num_samples() * acc_ds.repeats,
                    rng_sched=ctx.rt_settings.rng_sched,
                    rng_sample_index=ctx.rt_settings.rng_sample_index,
                    load_pattern=ctx.rt_settings.load_pattern,
                )
                acc_sched = ctx.scheduler.__class__(
                    acc_rt, WithoutReplacementSampleOrder
                )

                logger.info(f"Running accuracy phase: {ds_name}")
                await _run_accuracy_phase(rt, acc_ds, acc_sched)

            logger.info("All accuracy samples issued")

        # ── Drain + finalize ─────────────────────────────────────────
        recorder.should_check_idle = True
        recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())

        if recorder.n_inflight_samples > 0 and not rt.stop_requested:
            try:
                await asyncio.wait_for(
                    idle_event.wait(),
                    timeout=config.timeout or SystemDefaults.DEFAULT_TIMEOUT,
                )
            except TimeoutError:
                logger.warning(
                    f"Timed out waiting for {recorder.n_inflight_samples} inflight samples"
                )

        recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
        session_ended = True
        writer.stop()

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
    except ExecutionError:
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise ExecutionError(f"Benchmark execution failed: {e}") from e
    finally:
        _cleanup(
            loop,
            pbar,
            recorder,
            writer,
            http_client,
            publisher,
            zmq_ctx,
            session_ended,
        )

        # Reset CPU affinity is in _cleanup; generate report here
        if recorder:
            report = _generate_report(recorder, ctx)

    return (report, collector)
