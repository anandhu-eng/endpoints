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
TODO: PoC only, subject to change!

Benchmark command implementation."""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.commands.utils import get_default_report_path
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    ClientSettings,
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    Metrics,
    ModelParams,
    OSLDistribution,
    RuntimeConfig,
    Settings,
    StreamingMode,
    SystemDefaults,
    TestMode,
    TestType,
)
from inference_endpoint.config.schema import (
    Dataset as DatasetConfig,
)
from inference_endpoint.config.yaml_loader import ConfigError, ConfigLoader
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.cpu_affinity import pin_loadgen
from inference_endpoint.endpoint_client.http_client import (
    AsyncHttpEndpointClient,
    HTTPEndpointClient,
)
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.evaluation import Extractor
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import (
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.scheduler import Scheduler

# Suppress HuggingFace warnings about missing PyTorch/TensorFlow
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class ResponseCollector:
    """Collects query responses and errors for accuracy evaluation and reporting.

    TODO (to be deprecated): This is a temporary testing/validation feature. Once the full
    reporter functionality is implemented, this class will be deprecated in
    favor of the comprehensive reporting system.

    This collector acts as a callback handler for completed queries, tracking:
    - Total count of completed queries
    - Response outputs (when collect_responses is True)
    - Error messages from failed queries

    Used primarily in accuracy evaluation mode (TestMode.ACC/BOTH) where
    responses need to be stored for later comparison against ground truth.

    Attributes:
        collect_responses: Whether to store full response text (memory intensive).
        responses: Map of query_id -> response_output for successful queries.
        errors: List of error messages from failed queries.
        count: Total number of completed queries (success + failure).
    """

    def __init__(self, collect_responses: bool = False, pbar: tqdm | None = None):
        """Initialize response collector.

        Args:
            collect_responses: If True, stores full response text for each query.
                              If False, only tracks counts and errors (saves memory).
        """
        self.collect_responses = collect_responses
        self.responses: dict[str, str] = {}
        self.errors: list[str] = []
        self.count = 0

        self.pbar = pbar

    def on_complete_hook(self, result: QueryResult):
        """Callback invoked when a query completes (success or failure).

        This method is registered with SampleEventHandler and called automatically
        when a COMPLETE event fires. It updates internal counters and optionally
        stores the response text.

        Args:
            result: QueryResult containing response data or error information.
        """
        self.count += 1
        if result.error:
            self.errors.append(f"Sample {result.id}: {result.error}")
            if self.pbar:
                self.pbar.set_postfix(refresh=True, errors=len(self.errors))
        elif self.collect_responses:
            self.responses[result.id] = result.get_response_output_string()

        if self.pbar:
            self.pbar.update(1)


@dataclass
class AccuracyConfiguration:
    scorer: type[Scorer]
    extractor: type[Extractor]
    dataset_name: str
    dataset: Dataset
    report_dir: os.PathLike
    ground_truth_column: str | None
    num_repeats: int


@dataclass
class BenchmarkSetup:
    """All prepared state needed by a benchmark runner."""

    config: BenchmarkConfig
    tokenizer: Any
    report_dir: Path
    dataloader: Dataset
    scheduler: Scheduler
    rt_settings: RuntimeSettings
    http_config: HTTPClientConfig
    total_samples: int
    collect_responses: bool
    enable_streaming: bool
    affinity_plan: Any
    accuracy_datasets: list[Dataset]
    eval_configs: list[AccuracyConfiguration]
    model_name: str
    load_pattern_type: LoadPatternType
    endpoints: list[str]
    test_mode: TestMode
    benchmark_mode: TestType | None


def setup_benchmark(
    config: BenchmarkConfig,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> BenchmarkSetup:
    """Common setup for both sync and async benchmark runners.

    Handles: CPU affinity, tokenizer, report dir, streaming, datasets,
    scheduler, HTTP client config. Returns a BenchmarkSetup with all
    prepared state.
    """
    collect_responses = test_mode in [TestMode.ACC, TestMode.BOTH]

    # CPU affinity
    affinity_plan = None
    if config.enable_cpu_affinity:
        affinity_plan = pin_loadgen(config.settings.client.workers)

    # Model name
    model_name = config.model_params.name
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model
        config.model_params.name = model_name
    if not model_name:
        raise InputValidationError("No model name provided")

    # Report directory
    report_dir = (
        Path(config.report_dir) if config.report_dir else get_default_report_path()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    # Tokenizer
    tokenizer = None
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
        logger.warning("Continuing without tokenizer (report metrics may be limited)")

    # Streaming mode
    streaming_mode = config.model_params.streaming
    if streaming_mode == StreamingMode.ON:
        enable_streaming = True
    elif streaming_mode == StreamingMode.OFF:
        enable_streaming = False
    else:
        enable_streaming = benchmark_mode == TestType.ONLINE
        config.model_params.streaming = (
            StreamingMode.ON if enable_streaming else StreamingMode.OFF
        )

    # Datasets
    accuracy_configs = [d for d in config.datasets if d.type == DatasetType.ACCURACY]
    performance_configs = [
        d for d in config.datasets if d.type == DatasetType.PERFORMANCE
    ]
    if not performance_configs and not accuracy_configs:
        raise InputValidationError("No performance or accuracy datasets provided")

    # Accuracy datasets
    accuracy_datasets: list[Dataset] = []
    eval_configs: list[AccuracyConfiguration] = []
    for acc_config in accuracy_configs:
        assert (
            acc_config.accuracy_config is not None
        ), f"accuracy_config must be set for dataset {acc_config.name}"
        assert (
            acc_config.accuracy_config.eval_method is not None
        ), f"eval_method must be set for dataset {acc_config.name}"
        assert (
            acc_config.accuracy_config.extractor is not None
        ), f"extractor must be set for dataset {acc_config.name}"

        dataset = DataLoaderFactory.create_loader(
            acc_config, num_repeats=acc_config.accuracy_config.num_repeats
        )
        accuracy_datasets.append(dataset)
        eval_configs.append(
            AccuracyConfiguration(
                Scorer.get(acc_config.accuracy_config.eval_method),
                Extractor.get(acc_config.accuracy_config.extractor),
                acc_config.name,
                dataset,
                report_dir,
                acc_config.accuracy_config.ground_truth,
                acc_config.accuracy_config.num_repeats,
            )
        )
        dataset.load(
            api_type=config.endpoint_config.api_type,
            model_params=config.model_params,
        )
        logger.info(f"Loaded {dataset} - {dataset.num_samples()} samples")

    if not accuracy_configs:
        logger.info("No accuracy datasets provided")

    if len(performance_configs) > 1:
        logger.warning(
            "Multiple performance datasets provided, only the first one will be used"
        )

    # Performance dataset
    try:
        dataloader = DataLoaderFactory.create_loader(performance_configs[0])
        dataloader.load(
            api_type=config.endpoint_config.api_type, model_params=config.model_params
        )
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except FileNotFoundError as e:
        raise InputValidationError(
            f"Dataset file not found: {performance_configs[0].path}"
        ) from e
    except Exception as e:
        raise SetupError(f"Failed to load dataset: {e}") from e

    # Runtime settings + scheduler
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    load_pattern_type = config.settings.load_pattern.type

    total_samples = rt_settings.total_samples_to_issue()
    if accuracy_datasets:
        total_samples += sum(
            dataset.num_samples() * dataset.repeats for dataset in accuracy_datasets
        )

    try:
        scheduler_class = Scheduler.get_implementation(load_pattern_type)
        scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)
        logger.info(
            f"Scheduler: {scheduler_class.__name__} (pattern: {load_pattern_type.value})"
        )
    except KeyError as e:
        raise SetupError(str(e)) from e

    logger.info(
        f"Mode: {test_mode}, Target QPS: {config.settings.load_pattern.target_qps}, Responses: {collect_responses}"
    )
    logger.info(
        f"Min Duration: {rt_settings.min_duration_ms / 1000:.1f}s, Expected samples: {total_samples}"
    )

    # HTTP client config
    endpoints = config.endpoint_config.endpoints
    assert endpoints is not None
    api_type: APIType = config.endpoint_config.api_type

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(e, api_type.default_route()) for e in endpoints],
        api_type=api_type,
        num_workers=config.settings.client.workers,
        record_worker_events=config.settings.client.record_worker_events,
        event_logs_dir=report_dir,
        log_level=config.settings.client.log_level,
        cpu_affinity=affinity_plan,
        warmup_connections=config.settings.client.warmup_connections,
        max_connections=config.settings.client.max_connections,
        api_key=config.endpoint_config.api_key,
    )

    return BenchmarkSetup(
        config=config,
        tokenizer=tokenizer,
        report_dir=report_dir,
        dataloader=dataloader,
        scheduler=scheduler,
        rt_settings=rt_settings,
        http_config=http_config,
        total_samples=total_samples,
        collect_responses=collect_responses,
        enable_streaming=enable_streaming,
        affinity_plan=affinity_plan,
        accuracy_datasets=accuracy_datasets,
        eval_configs=eval_configs,
        model_name=model_name,
        load_pattern_type=load_pattern_type,
        endpoints=endpoints,
        test_mode=test_mode,
        benchmark_mode=benchmark_mode,
    )


def post_benchmark(
    setup: BenchmarkSetup,
    report: Any,
    response_collector: ResponseCollector,
) -> None:
    """Shared post-benchmark processing: accuracy scoring, results JSON, error summary."""
    if report is None:
        logger.error(
            "Session report missing — benchmark reporter failed to produce results"
        )
        raise ExecutionError(
            "Session report missing — cannot produce benchmark results"
        )

    elapsed_time = report.duration_ns / 1e9
    total = report.n_samples_issued
    success_count = report.n_samples_completed
    estimated_qps = report.qps or 0.0

    logger.info(f"Completed in {elapsed_time:.1f}s")
    logger.info(f"Results: {success_count}/{total} successful")
    logger.info(f"Estimated QPS: {estimated_qps:.1f}")

    # Accuracy scoring
    accuracy_scores: dict[str, Any] = {}
    for eval_config in setup.eval_configs:
        scorer_instance = eval_config.scorer(
            eval_config.dataset_name,
            eval_config.dataset,
            eval_config.report_dir,
            extractor=eval_config.extractor,
            ground_truth_column=eval_config.ground_truth_column,
        )
        score, n_repeats = scorer_instance.score()
        assert eval_config.dataset.data is not None
        accuracy_scores[eval_config.dataset_name] = {
            "dataset_name": eval_config.dataset_name,
            "num_samples": len(eval_config.dataset.data),
            "extractor": eval_config.extractor.__name__,
            "ground_truth_column": eval_config.ground_truth_column,
            "score": score,
            "n_repeats": n_repeats,
        }
        logger.info(
            f"Score for {eval_config.dataset_name}: {score} ({n_repeats} repeats)"
        )

    # Error summary
    if response_collector.errors:
        logger.warning(f"Errors: {len(response_collector.errors)}")
        if setup.config.verbose:
            for error in response_collector.errors[:3]:
                logger.warning(f"  {error}")
            if len(response_collector.errors) > 3:
                logger.warning(f"  ... +{len(response_collector.errors) - 3} more")

    # Results JSON
    try:
        results: dict[str, Any] = {
            "config": {
                "endpoint": setup.endpoints,
                "mode": setup.test_mode.value
                if hasattr(setup.test_mode, "value")
                else str(setup.test_mode),
                "target_qps": setup.config.settings.load_pattern.target_qps,
            },
            "results": {
                "total": total,
                "successful": success_count,
                "failed": total - success_count,
                "elapsed_time": elapsed_time,
                "qps": estimated_qps,
            },
        }
        if accuracy_scores:
            results["accuracy_scores"] = accuracy_scores
        if setup.collect_responses:
            results["responses"] = response_collector.responses
        if response_collector.errors:
            results["errors"] = response_collector.errors
        results_path = setup.report_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {results_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")


async def run_benchmark_command(args: argparse.Namespace) -> None:
    """Run performance benchmark in offline, online, or YAML-configured mode.

    This is the main entry point for the benchmark command. It:
    1. Determines benchmark mode (offline/online/from-config)
    2. Builds or loads configuration (CLI args vs YAML)
    3. Validates configuration comprehensively
    4. Determines response collection strategy based on test mode
    5. Delegates to _run_benchmark() for execution

    Benchmark modes:
    - offline: Max throughput test (all queries at t=0)
    - online: Sustained QPS test (Poisson/concurrency-based scheduling)
    - from-config: Load all settings from YAML file

    Test modes (--mode):
    - perf: Performance metrics only (no response storage)
    - acc: Accuracy evaluation (collect responses)
    - both: Both performance and accuracy

    Args:
        args: Parsed command line arguments from argparse.
              Expected attributes vary by benchmark_mode:
              - offline/online: endpoint, model, dataset, qps, workers, etc.
              - from-config: config (YAML path)

    Raises:
        InputValidationError: If configuration is invalid or args are missing.
        SetupError: If initialization fails (dataset loading, connection, etc.).
        ExecutionError: If benchmark execution fails after successful setup.
    """

    # Determine benchmark mode
    benchmark_mode_str = getattr(
        args, "benchmark_mode", None
    )  # "offline", "online", "from-config", or None

    # Three subcommands:
    # - benchmark offline: CLI mode
    # - benchmark online: CLI mode
    # - benchmark from-config: YAML mode
    # Argparse enforces all arg validity per mode

    if benchmark_mode_str == "from-config":
        # ===== YAML MODE - Load from config file =====
        config_path = args.config  # Required by argparse
        try:
            effective_config: BenchmarkConfig = ConfigLoader.load_yaml(
                Path(config_path)
            )

            # Only auxiliary params allowed (output)
            mode_str = getattr(args, "mode", None)
            test_mode = (
                TestMode(mode_str)
                if mode_str
                else (
                    TestMode.BOTH
                    if effective_config.type == TestType.SUBMISSION
                    else TestMode.PERF
                )
            )

            # Get benchmark mode from config
            benchmark_mode = effective_config.get_benchmark_mode()
            if not benchmark_mode:
                raise InputValidationError(
                    "SUBMISSION configs must specify 'benchmark_mode' (offline or online)"
                )
        except ConfigError as e:
            logger.error(f"Config error: {e}")
            raise InputValidationError(f"Config error: {e}") from e

    elif benchmark_mode_str in ("offline", "online"):
        # ===== CLI MODE - Build config from CLI params =====
        benchmark_mode = TestType(benchmark_mode_str)  # TestType values are lowercase
        effective_config = _build_config_from_cli(args, benchmark_mode_str)
        test_mode = (
            TestMode(args.mode) if getattr(args, "mode", None) else TestMode.PERF
        )

    else:
        # Shouldn't happen with current argparse structure
        raise InputValidationError(
            "Unknown benchmark mode. Use: offline, online, or from-config"
        )

    # Validate configuration
    try:
        ConfigLoader.validate_config(effective_config, benchmark_mode)
    except ConfigError as e:
        logger.exception("Config validation error")
        raise InputValidationError(str(e)) from e

    # Common setup
    setup = setup_benchmark(effective_config, test_mode, benchmark_mode)

    # Select execution engine and run
    use_async = os.environ.get("INFERENCE_ENDPOINT_ASYNC", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_async:
        logger.info("Using single-loop async benchmark engine (uvloop)")
        from inference_endpoint.commands.benchmark_async import (
            run_benchmark as run_benchmark_async,
        )

        report, response_collector = await run_benchmark_async(setup)
    else:
        from inference_endpoint.commands.benchmark_sync import (
            run_benchmark as run_benchmark_sync,
        )

        report, response_collector = run_benchmark_sync(setup)

    # Shared post-processing: accuracy scoring, results.json, error summary
    post_benchmark(setup, report, response_collector)


def _build_config_from_cli(
    args: argparse.Namespace, benchmark_mode: str
) -> BenchmarkConfig:
    """Build BenchmarkConfig from CLI arguments (CLI mode only).

    Args:
        args: Parsed CLI arguments
        benchmark_mode: "online" or "offline"

    Returns:
        BenchmarkConfig built from CLI params

    Raises:
        InputValidationError: If required params missing
    """
    # Determine load pattern (CLI override or mode default)
    if load_pattern_arg := getattr(args, "load_pattern", None):
        load_pattern_type = LoadPatternType(load_pattern_arg)
    else:
        match benchmark_mode:
            case "offline":
                load_pattern_type = LoadPatternType.MAX_THROUGHPUT
            case "online" if getattr(args, "concurrency", None):
                load_pattern_type = LoadPatternType.CONCURRENCY
            case "online":
                load_pattern_type = LoadPatternType.POISSON
    report_dir = getattr(
        args,
        "report_dir",
        get_default_report_path(),
    )
    timeout = getattr(args, "timeout", None)
    verbose_level = getattr(args, "verbose", 0)
    api_type = APIType(getattr(args, "api_type", "openai"))
    # Build BenchmarkConfig from CLI params
    return BenchmarkConfig(
        name=f"cli_{benchmark_mode}",
        version="1.0",
        type=TestType.OFFLINE if benchmark_mode == "offline" else TestType.ONLINE,
        datasets=[
            DatasetConfig(
                name=args.dataset.stem,
                type=DatasetType.PERFORMANCE,
                path=str(args.dataset),
                format=None,  # Will be inferred by DataLoaderFactory
            )
        ],
        settings=Settings(
            load_pattern=LoadPattern(
                type=load_pattern_type,
                target_qps=getattr(args, "target_qps", None),
                target_concurrency=getattr(args, "concurrency", None),
            ),
            runtime=RuntimeConfig(
                min_duration_ms=args.duration * 1000
                if args.duration
                else 0,  # Default: 0 (sample count determined by n_samples_to_issue or dataset size)
                max_duration_ms=1800000,
                n_samples_to_issue=getattr(
                    args, "num_samples", None
                ),  # Map --num-samples to config
                scheduler_random_seed=42,
                dataloader_random_seed=42,
            ),
            client=ClientSettings(
                workers=args.workers if args.workers else -1,
                log_level="DEBUG" if verbose_level >= 2 else "INFO",
                warmup_connections=getattr(args, "warmup_connections", -1),
                max_connections=getattr(args, "max_connections", None) or -1,
            ),
        ),
        model_params=ModelParams(
            name=args.model,
            temperature=0.7,
            max_new_tokens=args.max_output_tokens if args.max_output_tokens else 1024,
            osl_distribution=OSLDistribution(
                min=args.min_output_tokens if args.min_output_tokens else 1,
                max=args.max_output_tokens if args.max_output_tokens else 2048,
            )
            if (args.min_output_tokens or args.max_output_tokens)
            else None,
            streaming=StreamingMode(getattr(args, "streaming", "auto")),
        ),
        endpoint_config=EndpointConfig(
            endpoints=[e.strip() for e in args.endpoints.split(",") if e.strip()],
            api_key=args.api_key,
            api_type=api_type,
        ),
        metrics=Metrics(),
        report_dir=report_dir,
        timeout=timeout,
        verbose=verbose_level > 0,
    )


def _run_benchmark(
    config: BenchmarkConfig,
    collect_responses: bool,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> None:
    """Execute the actual benchmark with full lifecycle management.

    This function orchestrates the complete benchmark execution:
    1. Load tokenizer for the target model
    2. Load and validate dataset using DataLoaderFactory
    3. Setup runtime settings and scheduler
    4. Create HTTP endpoint client with multiprocessing workers
    5. Run benchmark session with signal handling
    6. Collect and report results
    7. Clean up resources (always, even on error)

    Architecture notes:
    - This is a SYNCHRONOUS function (not async) because HTTPEndpointClient
      manages its own event loop in a separate thread
    - Uses blocking operations: sess.wait_for_test_end()
    - Signal handling: SIGINT (Ctrl+C) gracefully stops benchmark
    - Cleanup: Always executes via finally block

    Streaming behavior:
    - Enabled automatically for online mode (for TTFT metrics)
    - Disabled for offline mode (max throughput focus)

    Args:
        args: Command arguments containing output paths, verbosity, etc.
        config: Validated BenchmarkConfig (immutable Pydantic model).
               Contains all benchmark parameters from CLI or YAML.
        collect_responses: Whether to store full response text.
                          True for accuracy modes (TestMode.ACC/BOTH).
        test_mode: What to collect - PERF (metrics only), ACC (responses),
                  or BOTH (metrics + responses).
        benchmark_mode: Execution mode - OFFLINE (max throughput) or
                       ONLINE (sustained QPS). Affects streaming and scheduling.

    Raises:
        InputValidationError: If model/dataset cannot be loaded or validated.
        SetupError: If connection to endpoint fails or resources unavailable.
        ExecutionError: If benchmark execution fails after successful setup.
        KeyboardInterrupt: If user interrupts with Ctrl+C (re-raised for CLI handler).
    """
    # CPU affinity: compute plan and pin loadgen
    affinity_plan = None
    if config.enable_cpu_affinity:
        affinity_plan = pin_loadgen(config.settings.client.workers)

    # Load tokenizer if model name is provided
    # Priority: CLI args (offline/online modes) > config submission_ref (from-config mode)
    tokenizer = None
    model_name = config.model_params.name
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model
        config.model_params.name = model_name

    if config.report_dir:
        report_dir = Path(config.report_dir)
    else:
        report_dir = get_default_report_path()

    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    if model_name:
        try:
            logger.info(f"Loading tokenizer for model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            logger.warning(
                "Continuing without tokenizer (report metrics may be limited)"
            )
    else:
        # Throw exception if no model name is provided
        raise InputValidationError("No model name provided")

    # Determine if streaming should be enabled based on config
    streaming_mode = config.model_params.streaming

    if streaming_mode == StreamingMode.ON:
        enable_streaming = True
        logger.info("Streaming: enabled (forced via --streaming=on)")
    elif streaming_mode == StreamingMode.OFF:
        enable_streaming = False
        logger.info("Streaming: disabled (forced via --streaming=off)")
    else:  # StreamingMode.AUTO
        enable_streaming = benchmark_mode == TestType.ONLINE
        if enable_streaming:
            logger.info("Streaming: enabled (auto, online mode)")
            config.model_params.streaming = StreamingMode.ON
        else:
            logger.info("Streaming: disabled (auto, offline mode)")
            config.model_params.streaming = StreamingMode.OFF

    # Get dataset - from CLI or from config
    # TODO: Dataset Logic is not yet fully implemented

    accuracy_configs = [
        config for config in config.datasets if config.type == DatasetType.ACCURACY
    ]
    performance_configs = [
        config for config in config.datasets if config.type == DatasetType.PERFORMANCE
    ]
    if not performance_configs and not accuracy_configs:
        raise InputValidationError("No performance or accuracy datasets provided")
    accuracy_datasets = []
    eval_configs = []
    if len(accuracy_configs) > 0:
        # Pack the evaluation parameters for each accuracy dataset
        for acc_config in accuracy_configs:
            # Type narrowing: ensure accuracy_config is not None
            assert (
                acc_config.accuracy_config is not None
            ), f"accuracy_config must be set for dataset {acc_config.name}"
            # Type narrowing: ensure required fields are not None
            assert (
                acc_config.accuracy_config.eval_method is not None
            ), f"eval_method must be set for dataset {acc_config.name}"
            assert (
                acc_config.accuracy_config.extractor is not None
            ), f"extractor must be set for dataset {acc_config.name}"

            dataset = DataLoaderFactory.create_loader(
                acc_config, num_repeats=acc_config.accuracy_config.num_repeats
            )
            accuracy_datasets.append(dataset)
            # TODO add tests and defaults
            eval_configs.append(
                AccuracyConfiguration(
                    Scorer.get(acc_config.accuracy_config.eval_method),
                    Extractor.get(acc_config.accuracy_config.extractor),
                    acc_config.name,
                    dataset,
                    report_dir,
                    acc_config.accuracy_config.ground_truth,
                    acc_config.accuracy_config.num_repeats,
                )
            )
            dataset.load(
                api_type=config.endpoint_config.api_type,
                model_params=config.model_params,
            )
            logger.info(f"Loaded {dataset} - {dataset.num_samples()} samples")

    else:
        logger.info("No accuracy datasets provided")
    if len(performance_configs) > 1:
        logger.warning(
            "Multiple performance datasets provided, only the first one will be used"
        )

    try:
        dataloader = DataLoaderFactory.create_loader(
            performance_configs[0]
        )  # Do not repeat perf datasets
        dataloader.load(
            api_type=config.endpoint_config.api_type, model_params=config.model_params
        )
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {performance_configs[0].path}")
        raise InputValidationError(
            f"Dataset file not found: {performance_configs[0].path}"
        ) from e
    except Exception as e:
        logger.error("Dataset load failed")
        raise SetupError(f"Failed to load dataset: {e}") from e

    # Setup runtime settings using factory method
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    target_qps = config.settings.load_pattern.target_qps
    load_pattern_type = config.settings.load_pattern.type

    # Calculate and display expected sample count
    total_samples = rt_settings.total_samples_to_issue()
    if accuracy_datasets is not None:
        total_samples += sum(
            [dataset.num_samples() * dataset.repeats for dataset in accuracy_datasets]
        )
    duration_s = rt_settings.min_duration_ms / 1000

    logger.info(
        f"Mode: {test_mode}, Target QPS: {target_qps}, Responses: {collect_responses}"
    )
    logger.info(f"Min Duration: {duration_s:.1f}s, Expected samples: {total_samples}")

    # Create scheduler using __init_subclass__ registry
    try:
        scheduler_class = Scheduler.get_implementation(load_pattern_type)
        scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)
        logger.info(
            f"Scheduler: {scheduler_class.__name__} (pattern: {load_pattern_type.value})"
        )
    except KeyError as e:
        logger.exception("Scheduler not available")
        raise SetupError(str(e)) from e

    # Setup response collector
    pbar = tqdm(
        desc=f"{model_name} (Streaming: {enable_streaming})",
        total=total_samples,
        smoothing=0,  # smoothing=0 shows average instead of EMA
    )
    response_collector = ResponseCollector(
        collect_responses=collect_responses, pbar=pbar
    )
    SampleEventHandler.register_hook(
        SampleEvent.COMPLETE, response_collector.on_complete_hook
    )

    # Create endpoint client
    endpoints = config.endpoint_config.endpoints
    assert endpoints is not None
    num_workers = config.settings.client.workers

    logger.info(f"Connecting: {endpoints}")
    # Scope ZMQ context so transport and sockets are cleaned up when the block exits.
    with ManagedZMQContext.scoped() as zmq_ctx:
        tmp_dir = tempfile.mkdtemp(prefix="inference_endpoint_")

        try:
            api_type: APIType = config.endpoint_config.api_type
            assert api_type is not None
            http_config = HTTPClientConfig(
                endpoint_urls=[urljoin(e, api_type.default_route()) for e in endpoints],
                api_type=api_type,
                num_workers=num_workers,
                record_worker_events=config.settings.client.record_worker_events,
                event_logs_dir=report_dir,
                log_level=config.settings.client.log_level,
                cpu_affinity=affinity_plan,
                warmup_connections=config.settings.client.warmup_connections,
                max_connections=config.settings.client.max_connections,
                api_key=config.endpoint_config.api_key,
            )
            http_client = HTTPEndpointClient(http_config, zmq_context=zmq_ctx)
            sample_issuer = HttpClientSampleIssuer(http_client)

        except Exception as e:
            logger.error("Connection failed")
            raise SetupError(f"Failed to connect to endpoint: {e}") from e

        # Run benchmark
        logger.info("Running...")

        sess = None
        try:
            sess = BenchmarkSession.start(
                rt_settings,
                dataloader,
                sample_issuer,
                scheduler,
                name=f"cli_benchmark_{uuid.uuid4().hex[0:8]}",
                report_dir=report_dir,
                tokenizer_override=tokenizer,
                accuracy_datasets=accuracy_datasets,
                max_shutdown_timeout_s=config.timeout
                if config.timeout
                else SystemDefaults.DEFAULT_TIMEOUT,
                dump_events_log=True,
            )

            # Wait for test end with ability to interrupt
            def signal_handler(signum, frame):
                logger.warning("Interrupt signal received, stopping benchmark...")
                # Raise KeyboardInterrupt to break out of wait_for_test_end()
                raise KeyboardInterrupt()

            # Install our handler, save old one
            old_handler = signal.signal(signal.SIGINT, signal_handler)
            try:
                sess.wait_for_test_end()
            finally:
                # Always restore original handler
                signal.signal(signal.SIGINT, old_handler)
                accuracy_scores = {}
            for eval_config in eval_configs:
                scorer_instance = eval_config.scorer(
                    eval_config.dataset_name,
                    eval_config.dataset,
                    eval_config.report_dir,
                    extractor=eval_config.extractor,
                    ground_truth_column=eval_config.ground_truth_column,
                )
                score, n_repeats = scorer_instance.score()
                assert eval_config.dataset.data is not None
                accuracy_scores[eval_config.dataset_name] = {
                    "dataset_name": eval_config.dataset_name,
                    "num_samples": len(eval_config.dataset.data),
                    "extractor": eval_config.extractor.__name__,
                    "ground_truth_column": eval_config.ground_truth_column,
                    "score": score,
                    "n_repeats": n_repeats,
                }
                logger.info(
                    f"Score for {eval_config.dataset_name}: {score} ({n_repeats} repeats)"
                )

            # Prefer authoritative metrics from the session report
            report = getattr(sess, "report", None)
            if report is None:
                logger.error(
                    "Session report missing — benchmark reporter failed to produce results"
                )
                raise ExecutionError(
                    "Session report missing — cannot produce benchmark results"
                )

            elapsed_time = report.duration_ns / 1e9
            total = report.n_samples_issued
            success_count = report.n_samples_completed

            # qps will be None if duration was 0, so fall back to 0.0
            estimated_qps = report.qps or 0.0

            # Report results
            logger.info(f"Completed in {elapsed_time:.1f}s")
            logger.info(f"Results: {success_count}/{total} successful")
            logger.info(f"Estimated QPS: {estimated_qps:.1f}")

            if response_collector.errors:
                logger.warning(f"Errors: {len(response_collector.errors)}")
                if config.verbose:
                    for error in response_collector.errors[:3]:
                        logger.warning(f"  {error}")
                    if len(response_collector.errors) > 3:
                        logger.warning(
                            f"  ... +{len(response_collector.errors) - 3} more"
                        )

            try:
                results: dict[str, Any] = {
                    "config": {
                        "endpoint": endpoints,
                        "mode": test_mode,
                        "target_qps": target_qps,
                    },
                    "results": {
                        "total": total,
                        "successful": success_count,
                        "failed": total - success_count,
                        "elapsed_time": elapsed_time,
                        "qps": estimated_qps,
                    },
                }
                if accuracy_scores:
                    results["accuracy_scores"] = accuracy_scores
                if collect_responses:
                    results["responses"] = response_collector.responses
                # Always save all errors (useful for debugging)
                if response_collector.errors:
                    results["errors"] = response_collector.errors
                # Save results to JSON file
                results_path = report_dir / "results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved: {results_path}")
            except Exception as e:
                logger.error(f"Save failed: {e}")

        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            raise
        except ExecutionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logger.error("Benchmark failed")
            raise ExecutionError(f"Benchmark execution failed: {e}") from e
        finally:
            # Cleanup - always execute
            logger.info("Cleaning up...")
            try:
                if sess is not None:
                    sess.stop()
                pbar.close()
                sample_issuer.shutdown()
                http_client.shutdown()
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                if config.verbose:
                    logger.warning(f"Cleanup error: {e}")


# =============================================================================
# Single-loop async benchmark (uvloop)
# =============================================================================


async def _run_benchmark_async(
    config: BenchmarkConfig,
    collect_responses: bool,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> None:
    """Execute benchmark on a single uvloop — no threads in the main process.

    Architecture:
    - AsyncHttpEndpointClient on the running loop (not sync wrapper)
    - ZmqEventRecordPublisher for event recording (sync ZMQ PUB NOBLOCK)
    - EventWriterProcess in background for SQLite writes
    - Online: loop.call_at() callback chain for Poisson scheduling
    - Offline: tight send loop + sleep(0) every 1000
    - Unified receiver: poll() + sleep(0) when idle (benchmark_httpclient.py pattern)
    """
    from inference_endpoint.async_utils.transport.record import (
        SampleEventType,
        SessionEventType,
    )
    from inference_endpoint.async_utils.transport.zmq.pubsub import (
        ZmqEventRecordPublisher,
    )
    from inference_endpoint.core.types import Query, StreamChunk
    from inference_endpoint.metrics.recorder_subprocess import (
        AsyncEventRecorder,
        EventWriterProcess,
    )
    from inference_endpoint.metrics.reporter import MetricsReporter

    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

    # ── Setup (same as _run_benchmark) ────────────────────────────────────

    affinity_plan = None
    if config.enable_cpu_affinity:
        affinity_plan = pin_loadgen(config.settings.client.workers)

    tokenizer = None
    model_name = config.model_params.name
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model
        config.model_params.name = model_name

    report_dir = (
        Path(config.report_dir) if config.report_dir else get_default_report_path()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    if model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
    else:
        raise InputValidationError("No model name provided")

    streaming_mode = config.model_params.streaming
    enable_streaming = streaming_mode == StreamingMode.ON or (
        streaming_mode == StreamingMode.AUTO and benchmark_mode == TestType.ONLINE
    )
    if streaming_mode == StreamingMode.AUTO:
        config.model_params.streaming = (
            StreamingMode.ON if enable_streaming else StreamingMode.OFF
        )

    # Dataset loading
    performance_configs = [
        c for c in config.datasets if c.type == DatasetType.PERFORMANCE
    ]
    if not performance_configs:
        raise InputValidationError("No performance datasets provided")
    dataloader = DataLoaderFactory.create_loader(performance_configs[0])
    dataloader.load(
        api_type=config.endpoint_config.api_type, model_params=config.model_params
    )

    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    load_pattern_type = config.settings.load_pattern.type
    total_samples = rt_settings.total_samples_to_issue()

    scheduler_class = Scheduler.get_implementation(load_pattern_type)
    scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)

    # ── Create HTTP client on the running loop ────────────────────────────

    endpoints = config.endpoint_config.endpoints
    assert endpoints is not None
    api_type_val: APIType = config.endpoint_config.api_type

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(e, api_type_val.default_route()) for e in endpoints],
        api_type=api_type_val,
        num_workers=config.settings.client.workers,
        record_worker_events=config.settings.client.record_worker_events,
        event_logs_dir=report_dir,
        log_level=config.settings.client.log_level,
        cpu_affinity=affinity_plan,
        warmup_connections=config.settings.client.warmup_connections,
        max_connections=config.settings.client.max_connections,
        api_key=config.endpoint_config.api_key,
    )

    # Declare resources upfront for finally block — safe to check even if
    # setup fails partway through (all start as None).
    zmq_ctx: ManagedZMQContext | None = None
    http_client: AsyncHttpEndpointClient | None = None
    publisher: ZmqEventRecordPublisher | None = None
    writer: EventWriterProcess | None = None
    recorder: AsyncEventRecorder | None = None
    pbar: tqdm | None = None
    session_ended = False
    stop_requested = False

    try:
        # ── Resource creation ─────────────────────────────────────────────

        zmq_ctx = ManagedZMQContext(io_threads=4)

        # Client construction blocks on run_coroutine_threadsafe, so use to_thread
        http_client = await asyncio.to_thread(
            AsyncHttpEndpointClient, http_config, loop=loop, zmq_context=zmq_ctx
        )

        # ── Event recording infrastructure ────────────────────────────────

        session_id = f"cli_benchmark_{uuid.uuid4().hex[:8]}"
        pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
        publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

        writer = EventWriterProcess(session_id, publisher.bind_address)
        writer.start(sub_settle_s=0.5)  # blocking: waits for subscriber readiness

        idle_event = asyncio.Event()
        recorder = AsyncEventRecorder(publisher, session_id, notify_idle=idle_event)

        # ── Progress bar + response collector ─────────────────────────────

        pbar = tqdm(
            desc=f"{model_name} (Streaming: {enable_streaming})",
            total=total_samples,
            smoothing=0,
        )
        response_collector = ResponseCollector(
            collect_responses=collect_responses, pbar=pbar
        )

        # ── Signal handling ───────────────────────────────────────────────

        send_done = False
        stop_requested = False
        uuid_to_index: dict[str, int] = {}

        def on_sigint():
            nonlocal stop_requested, send_done
            logger.warning("Interrupt received, stopping benchmark...")
            stop_requested = True
            send_done = True  # unblock receiver drain check

        loop.add_signal_handler(signal.SIGINT, on_sigint)

        # ── Send + Receive ────────────────────────────────────────────────

        recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

        def handle_response(result):
            """Process a received response (QueryResult or StreamChunk)."""
            ts = time.monotonic_ns()
            match result:
                case StreamChunk(is_complete=False):
                    metadata = result.metadata or {}
                    if metadata.get("first_chunk", False):
                        recorder.record_event(
                            SampleEventType.RECV_FIRST,
                            ts,
                            sample_uuid=result.id,
                            data={"response_chunk": result.response_chunk},
                        )
                    else:
                        recorder.record_event(
                            SampleEventType.RECV_NON_FIRST,
                            ts,
                            sample_uuid=result.id,
                        )
                case QueryResult(error=err):
                    if err is not None:
                        logger.error(f"Error in request {result.id}: {err}")
                    recorder.record_event(
                        SampleEventType.COMPLETE,
                        ts,
                        sample_uuid=result.id,
                        data=result.response_output,
                    )
                    response_collector.on_complete_hook(result)
                    scheduler.notify_complete()

        async def receiver():
            """Unified receiver: poll() + sleep(0) when idle."""
            while True:
                result = http_client.poll()
                if result is not None:
                    handle_response(result)
                else:
                    if send_done and (
                        recorder.n_inflight_samples <= 0 or stop_requested
                    ):
                        break
                    await asyncio.sleep(0)

        def issue_sample(s_idx: int) -> str:
            """Issue a single sample — shared by all send paths."""
            sample_uuid = uuid.uuid4().hex
            sample_data = dataloader.load_sample(s_idx)
            ts = time.monotonic_ns()
            recorder.record_event(SampleEventType.ISSUED, ts, sample_uuid=sample_uuid)
            http_client.issue(Query(id=sample_uuid, data=sample_data))
            uuid_to_index[sample_uuid] = s_idx
            return sample_uuid

        if load_pattern_type == LoadPatternType.MAX_THROUGHPUT:
            # Offline: tight send loop, yield every 1000
            async def sender():
                nonlocal send_done
                sent = 0
                for s_idx, _ in scheduler:
                    if stop_requested:
                        break
                    issue_sample(s_idx)
                    sent += 1
                    if sent % 1000 == 0:
                        await asyncio.sleep(0)
                send_done = True
        else:
            # Online (Poisson / Concurrency): scheduler.__aiter__ handles timing
            async def sender():
                nonlocal send_done
                async for s_idx in scheduler:
                    if stop_requested:
                        break
                    issue_sample(s_idx)
                send_done = True

        await asyncio.gather(sender(), receiver())

        # Restore default SIGINT so Ctrl+C raises KeyboardInterrupt during
        # post-benchmark reporting and cleanup (no more flag-only handler).
        loop.remove_signal_handler(signal.SIGINT)

        # ── Post-benchmark ────────────────────────────────────────────────

        recorder.record_event(
            SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
        )
        recorder.should_check_idle = True
        recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())

        if recorder.n_inflight_samples > 0 and not stop_requested:
            try:
                await asyncio.wait_for(idle_event.wait(), timeout=config.timeout or 300)
            except TimeoutError:
                logger.warning(
                    f"Timed out waiting for {recorder.n_inflight_samples} inflight samples"
                )

        recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
        session_ended = True
        writer.stop()

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise ExecutionError(f"Benchmark execution failed: {e}") from e
    finally:
        # Each step guarded individually — failure in one doesn't skip the rest.
        # Order: signal → pbar → session ended → writer → client → report → zmq

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
                writer.stop(timeout=5.0)
            except Exception:
                pass

        if http_client:
            try:
                await http_client.shutdown()
            except Exception:
                pass

        # Reset CPU affinity — loadgen pinning no longer needed,
        # use all cores for report tokenization
        try:
            os.sched_setaffinity(0, range(os.cpu_count() or 1))
        except (OSError, AttributeError):
            pass
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        if recorder:
            try:
                with MetricsReporter(
                    recorder.connection_name, client_type="sqlite"
                ) as reporter:
                    report = reporter.create_report(tokenizer)
                    report.display(fn=print, summary_only=True)

                    if report_dir:
                        report.to_json(save_to=report_dir / "result_summary.json")
                        with open(report_dir / "report.txt", "w") as f:
                            report.display(fn=f.write, summary_only=False, newline="\n")
                        reporter.dump_to_json(report_dir / "events.jsonl")
                        logger.info(f"Report saved to: {report_dir}")
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")

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
