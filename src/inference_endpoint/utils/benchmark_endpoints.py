#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end benchmark runner.

Launches a MaxThroughputServer and runs the full benchmark pipeline
to validate the runtime works correctly. Tests offline, poisson, and
concurrency modes with either the async or sync (threaded) runtime.

Usage::

    # Async path (default) — offline only
    python -m inference_endpoint.utils.benchmark_endpoints

    # Sync (threaded) path — offline only
    python -m inference_endpoint.utils.benchmark_endpoints --sync

    # All three load patterns, both paths
    python -m inference_endpoint.utils.benchmark_endpoints --all
    python -m inference_endpoint.utils.benchmark_endpoints --all --sync

    # Against an external endpoint
    python -m inference_endpoint.utils.benchmark_endpoints --endpoint http://host:8080

    # Custom parameters
    python -m inference_endpoint.utils.benchmark_endpoints --samples 1000 --target-qps 500 -w 8

    # Streaming mode
    python -m inference_endpoint.utils.benchmark_endpoints --stream --all
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _build_config(
    endpoint_url: str,
    *,
    mode: str,
    model: str,
    dataset_path: str,
    samples: int,
    target_qps: float,
    target_concurrency: int,
    workers: int,
    streaming: bool,
    report_dir: str,
    timeout: float,
):
    """Build a BenchmarkConfig for the given mode."""
    from inference_endpoint.config.schema import (
        ClientSettings,
        Dataset,
        EndpointConfig,
        LoadPattern,
        LoadPatternType,
        ModelParams,
        OfflineBenchmarkConfig,
        OnlineBenchmarkConfig,
        OnlineSettings,
        StreamingMode,
    )

    endpoint_config = EndpointConfig(endpoints=[endpoint_url])
    model_params = ModelParams(
        name=model,
        streaming=StreamingMode.ON if streaming else StreamingMode.OFF,
    )
    client = ClientSettings(workers=workers)
    perf_dataset = Dataset(
        path=dataset_path,
        samples=samples,
        parser={"prompt": "text_input"},
    )

    if mode == "offline":
        from inference_endpoint.config.schema import OfflineSettings

        return OfflineBenchmarkConfig(
            endpoint_config=endpoint_config,
            model_params=model_params,
            datasets=[perf_dataset],
            report_dir=report_dir,
            timeout=timeout,
            settings=OfflineSettings(client=client),
        )

    if mode == "poisson":
        load_pattern = LoadPattern(type=LoadPatternType.POISSON, target_qps=target_qps)
    elif mode == "concurrency":
        load_pattern = LoadPattern(
            type=LoadPatternType.CONCURRENCY,
            target_concurrency=target_concurrency,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return OnlineBenchmarkConfig(
        endpoint_config=endpoint_config,
        model_params=model_params,
        datasets=[perf_dataset],
        report_dir=report_dir,
        timeout=timeout,
        settings=OnlineSettings(load_pattern=load_pattern, client=client),
    )


def _run_async(config, test_mode) -> None:
    """Run benchmark via the async runtime (single uvloop, no threads)."""
    from inference_endpoint.async_utils.runner import run_async
    from inference_endpoint.commands.benchmark.execute import (
        finalize_benchmark,
        setup_benchmark,
    )
    from inference_endpoint.commands.benchmark.execute_async import (
        run_benchmark_async,
    )

    ctx = setup_benchmark(config, test_mode)
    report, collector = run_async(run_benchmark_async(ctx))
    finalize_benchmark(ctx, report, collector)


def _run_sync(config, test_mode) -> None:
    """Run benchmark via the old threaded runtime (BenchmarkSession + EventRecorder)."""
    from inference_endpoint.commands.benchmark.execute import (
        finalize_benchmark,
        run_benchmark_threaded,
        setup_benchmark,
    )

    ctx = setup_benchmark(config, test_mode)
    report, collector = run_benchmark_threaded(ctx)
    finalize_benchmark(ctx, report, collector)


def _run_one(
    endpoint_url: str,
    mode: str,
    *,
    sync: bool,
    model: str,
    dataset_path: str,
    samples: int,
    target_qps: float,
    target_concurrency: int,
    workers: int,
    streaming: bool,
    timeout: float,
) -> bool:
    """Run a single benchmark mode. Returns True on success."""
    from inference_endpoint.config.schema import TestMode

    runner_name = "sync" if sync else "async"

    with tempfile.TemporaryDirectory(
        prefix=f"bench_{mode}_{runner_name}_"
    ) as report_dir:
        try:
            config = _build_config(
                endpoint_url,
                mode=mode,
                model=model,
                dataset_path=dataset_path,
                samples=samples,
                target_qps=target_qps,
                target_concurrency=target_concurrency,
                workers=workers,
                streaming=streaming,
                report_dir=report_dir,
                timeout=timeout,
            )
            if sync:
                _run_sync(config, TestMode.PERF)
            else:
                _run_async(config, TestMode.PERF)
            return True
        except Exception as e:
            print(f"  [{mode}/{runner_name}] FAIL: {e}", file=sys.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="E2E benchmark smoke test (async & sync runtimes)"
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="External endpoint URL. If not set, launches MaxThroughputServer.",
    )
    parser.add_argument("--model", default="max-tp")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset path. Default: tests/datasets/dummy_1k.pkl",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--target-qps", type=float, default=100.0)
    parser.add_argument("--target-concurrency", type=int, default=8)
    parser.add_argument("-w", "--workers", type=int, default=2)
    parser.add_argument("--server-workers", type=int, default=2)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three modes (offline, poisson, concurrency)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use the old threaded runtime (BenchmarkSession) instead of async",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if dataset_path is None:
        default_path = "tests/datasets/dummy_1k.pkl"
        if os.path.exists(default_path):
            dataset_path = default_path
        else:
            print(f"Default dataset not found: {default_path}", file=sys.stderr)
            sys.exit(1)

    modes = ["offline", "poisson", "concurrency"] if args.all else ["offline"]
    runner_name = "sync" if args.sync else "async"

    server = None
    if args.endpoint:
        endpoint_url = args.endpoint
        print(f"Using external endpoint: {endpoint_url}")
    else:
        from inference_endpoint.testing.max_throughput_server import (
            MaxThroughputServer,
        )

        server = MaxThroughputServer(
            port=0,
            num_workers=args.server_workers,
            stream=args.stream,
            quiet=True,
        )
        server.start()
        endpoint_url = f"{server.url}/v1/chat/completions"
        print(f"MaxThroughputServer @ {server.url}")

    print(f"Runtime: {runner_name}")

    results: dict[str, bool] = {}
    try:
        for mode in modes:
            label = f"{mode}/{runner_name}"
            t0 = time.monotonic()
            ok = _run_one(
                endpoint_url,
                mode,
                sync=args.sync,
                model=args.model,
                dataset_path=dataset_path,
                samples=args.samples,
                target_qps=args.target_qps,
                target_concurrency=args.target_concurrency,
                workers=args.workers,
                streaming=args.stream,
                timeout=args.timeout,
            )
            elapsed = time.monotonic() - t0
            status = "PASS" if ok else "FAIL"
            results[label] = ok
            print(f"  {label}: {status} ({elapsed:.1f}s)")
    finally:
        if server:
            server.stop()

    # Summary
    print("\n--- Results ---")
    all_ok = True
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: {status}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
