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

"""End-to-end oracle test: verify responses match expected dataset outputs.

Uses the async BenchmarkSession to issue all samples to a mock oracle server,
then checks each response against the expected ground-truth output.
"""

import asyncio
import random
from pathlib import Path
from urllib.parse import urljoin

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.core.record import EventRecord
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnRemap,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import (
    HttpClientSampleIssuer,
)
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseType,
)


class _NoOpPublisher:
    def publish(self, event_record: EventRecord) -> None:
        pass


async def _run_benchmark(
    server_url: str,
    dataloader: Dataset,
    rt_settings: RuntimeSettings,
) -> tuple[dict[str, int], dict[str, str]]:
    """Run a benchmark and return (uuid_to_index, responses).

    Uses the async BenchmarkSession with MAX_THROUGHPUT strategy.
    Responses are collected via the on_sample_complete callback.
    """
    loop = asyncio.get_running_loop()

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(server_url, "/v1/chat/completions")],
        warmup_connections=0,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)

    responses: dict[str, str] = {}

    def on_complete(result: QueryResult) -> None:
        responses[result.id] = result.get_response_output_string()

    session = BenchmarkSession(
        issuer=issuer,
        event_publisher=_NoOpPublisher(),
        loop=loop,
        on_sample_complete=on_complete,
    )

    phases = [
        PhaseConfig(
            "performance",
            rt_settings,
            dataloader,
            PhaseType.PERFORMANCE,
        ),
    ]

    try:
        result = await session.run(phases)
    finally:
        await http_client.shutdown_async()

    perf = result.perf_results[0]
    return perf.uuid_to_index, responses


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_generator_full_run_mock_http_oracle_server(
    mock_http_oracle_server,
    ds_pickle_dataset_path,
    hf_model_name,
):
    dummy_dataloader = Dataset.load_from_file(
        ds_pickle_dataset_path,
        transforms=[
            ColumnRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    n_samples = dummy_dataloader.num_samples()
    assert n_samples > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1_000,
        max_duration_ms=10_000_000,
        n_samples_from_dataset=n_samples,
        n_samples_to_issue=n_samples,
        min_sample_count=1,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    uuid_to_index, responses = await _run_benchmark(
        mock_http_oracle_server.url, dummy_dataloader, rt_settings
    )

    # Verify all samples received responses
    assert (
        len(responses) == n_samples
    ), f"Expected {n_samples} responses, got {len(responses)}"

    # Build expected outputs from dataset
    expected: dict[int, str] = {}
    for i in range(n_samples):
        entry = dummy_dataloader.load_sample(i)
        expected[i] = entry["output"]

    # Verify each response matches the expected oracle output
    for sample_uuid, resp in responses.items():
        sample_index = uuid_to_index[sample_uuid]
        assert resp == expected[sample_index], (
            f"Sample {sample_uuid} (index={sample_index}): "
            f"expected {expected[sample_index][:30]!r}, got {resp[:30]!r}"
        )


async def _run_load_generator_full_run_url(
    url: str,
    dataset_path: Path,
    hf_model_name: str,
) -> None:
    """Helper for docker server tests."""
    dummy_dataloader = Dataset.load_from_file(
        dataset_path,
        transforms=[
            ColumnRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    n_samples = dummy_dataloader.num_samples()
    assert n_samples > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(50),
        [metrics.Throughput(50)],
        min_duration_ms=100,
        max_duration_ms=1_000,
        n_samples_from_dataset=n_samples,
        n_samples_to_issue=n_samples,
        min_sample_count=1,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    _, responses = await _run_benchmark(url, dummy_dataloader, rt_settings)
    assert len(responses) == n_samples


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_vllm_docker_server(
    vllm_docker_server,
    ds_pickle_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        vllm_docker_server.url,
        ds_pickle_dataset_path,
        hf_model_name,
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_sglang_docker_server(
    sglang_docker_server,
    ds_pickle_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        sglang_docker_server.url,
        ds_pickle_dataset_path,
        hf_model_name,
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_trtllm_docker_server(
    trtllm_docker_server,
    ds_pickle_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        trtllm_docker_server.url,
        ds_pickle_dataset_path,
        hf_model_name,
    )
