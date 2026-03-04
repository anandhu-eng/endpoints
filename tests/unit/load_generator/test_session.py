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

import random
from pathlib import Path
from unittest.mock import patch

import inference_endpoint.metrics as metrics
import pytest
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import Sample, SampleEventHandler
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.session import BenchmarkSession
from inference_endpoint.metrics.reporter import MetricsReporter
from tqdm import tqdm

from tests.test_helpers import (
    DummyDataLoader,
    PooledSampleIssuer,
)

# The following are tests for PooledSampleIssuer in test_helpers.py. If these tests pass
# but session.py tests fail, it's probably not the PooledSampleIssuer's fault.


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_pooled_issuer_exception_propagation(record_event_mock):
    record_event_mock.return_value = None

    """Test that exceptions in worker threads are properly propagated to the main thread."""

    def failing_compute(sample):
        raise ValueError("Worker thread error!")

    issuer = PooledSampleIssuer(compute_func=failing_compute, n_workers=2)

    sample1 = Sample(b"sample1")
    sample2 = Sample(b"sample2")

    # Submit some work that will fail
    issuer.issue(sample1)
    issuer.issue(sample2)

    # Shutdown should raise the exception from the worker thread
    with pytest.raises(ValueError, match="Worker thread error!"):
        issuer.shutdown()


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_pooled_issuer_futures_cleanup(record_event_mock):
    record_event_mock.return_value = None

    """Test that completed futures are cleaned up to prevent memory leaks."""
    import time

    def slow_compute(sample):
        time.sleep(0.01)  # Small delay
        return [sample.decode("utf-8")]

    issuer = PooledSampleIssuer(compute_func=slow_compute, n_workers=4)

    # Submit 250 samples (should trigger cleanup at 100 and 200)
    for _ in range(250):
        issuer.issue(Sample(b"sample"))

    # Let some time pass first
    time.sleep(0.2)

    # Manually check errors to trigger cleanup
    issuer.check_errors()

    for _ in range(250):
        issuer.issue(Sample(b"sample"))

    issuer.shutdown()

    # After shutdown, all futures should be cleared
    assert len(issuer.futures) == 0, "Futures not cleared after shutdown"


# session.py tests


def test_session_start(clean_sample_event_hooks):
    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=100,
        n_samples_to_issue=10_000,
        min_sample_count=100,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    def compute_digits_of_square(n: int):
        yield from str(n**2)

    dl = DummyDataLoader(n_samples=100)
    sample_issuer = PooledSampleIssuer(compute_digits_of_square)
    sched = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)

    class ProgressBarHook:
        def __init__(self, pbar: tqdm | None = None):
            self.pbar = pbar

        def __call__(self, _):
            if isinstance(self.pbar, tqdm):
                self.pbar.update(1)

        def set_pbar(self, pbar: tqdm):
            self.pbar = pbar

    pbar_hook = ProgressBarHook()
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, pbar_hook)

    with tqdm(desc="pytest_test_session_start", total=10_000, unit="samples") as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            dl,
            sample_issuer,
            sched,
            name="pytest_test_session_start",
        )
        events_db_path = sess.event_recorder.connection_name
        sess.wait_for_test_end()

        # Shutdown the sample issuer to ensure proper cleanup and error propagation
        sample_issuer.shutdown()

    assert Path(events_db_path).exists()
    with MetricsReporter(events_db_path) as reporter:
        stats = reporter.get_sample_statuses()
        assert stats["total_sent"] == 10_000
        assert stats["completed"] == 10_000
        assert stats["in_flight"] == 0


def test_session_with_prefill_dataset(clean_sample_event_hooks, tmp_path):
    """Test that a prefill dataset's samples are issued but excluded from reporting."""
    n_perf_samples = 200
    n_prefill_samples = 50

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=0,
        max_duration_ms=10_000,
        n_samples_from_dataset=100,
        n_samples_to_issue=n_perf_samples,
        min_sample_count=100,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    def compute_digits_of_square(n: int):
        yield from str(n**2)

    dl = DummyDataLoader(n_samples=100)
    prefill_dl = DummyDataLoader(n_samples=n_prefill_samples)
    sample_issuer = PooledSampleIssuer(compute_digits_of_square)
    sched = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)

    report_dir = tmp_path / "report"

    sess = BenchmarkSession.start(
        rt_settings,
        dl,
        sample_issuer,
        sched,
        prefill_dataset=prefill_dl,
        name="pytest_test_session_prefill",
        report_dir=report_dir,
        dump_events_log=True,
    )
    events_db_path = sess.event_recorder.connection_name
    sess.wait_for_test_end()
    sample_issuer.shutdown()

    # The report should only contain performance samples (not prefill)
    assert sess.report is not None
    assert sess.report.n_samples_issued == n_perf_samples
    assert sess.report.n_samples_completed == n_perf_samples

    # Prefill should NOT appear in sample_uuid_map
    assert sess.sample_uuid_map is not None
    all_uuids_in_map = set()
    for mapping in sess.sample_uuid_map.values():
        all_uuids_in_map.update(mapping.keys())
    assert len(all_uuids_in_map) == n_perf_samples

    # The raw events DB should contain ALL samples (perf + prefill)
    assert Path(events_db_path).exists()
    with MetricsReporter(events_db_path) as reporter:
        raw_stats = reporter.get_sample_statuses()
    assert raw_stats["completed"] == n_perf_samples + n_prefill_samples

    # events.jsonl should also contain prefill events (dumped before exclusion)
    events_jsonl = report_dir / "events.jsonl"
    assert events_jsonl.exists()
    events_text = events_jsonl.read_text()
    assert events_text.count('"loadgen_issue_called"') == n_perf_samples + n_prefill_samples
