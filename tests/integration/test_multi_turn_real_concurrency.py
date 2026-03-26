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

"""Real integration tests with actual HTTP requests for multi-turn concurrency."""

import json
import random
import tempfile
import time
from pathlib import Path

import pytest
from inference_endpoint import metrics
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    ConversationMode,
    LoadPattern,
    LoadPatternType,
    MultiTurnConfig,
)
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.scheduler import MultiTurnScheduler


class MultiTurnSampleIssuer(HttpClientSampleIssuer):
    """Sample issuer for multi-turn testing."""

    def __init__(self, endpoint_url: str, zmq_context: ManagedZMQContext, num_workers: int = 8):
        self.http_config = HTTPClientConfig(
            endpoint_urls=[endpoint_url],
            num_workers=num_workers,
        )
        super().__init__(HTTPEndpointClient(self.http_config, zmq_context=zmq_context))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
def test_50_concurrent_conversations_real_endpoint(tmp_path):
    """Test 50 concurrent conversations with real HTTP requests to model endpoint.

    This is a TRUE integration test:
    - Creates 50 conversations × 3 turns = 150 user messages
    - Makes 150 actual HTTP requests to port 8868
    - Uses full benchmark infrastructure (workers, ZMQ, etc.)
    - Measures real end-to-end latency with model inference
    """
    endpoint_url = "http://localhost:8868/v1/chat/completions"

    print("\n" + "=" * 70)
    print("REAL INTEGRATION TEST: 50 Concurrent Conversations")
    print("=" * 70)

    num_conversations = 50
    turns_per_conversation = 3

    # Create dataset
    dataset_path = tmp_path / "concurrent_50.jsonl"
    conversations = []

    print(f"\nCreating dataset: {num_conversations} conversations × {turns_per_conversation} turns")

    for conv_idx in range(num_conversations):
        conv_id = f"conv_{conv_idx:03d}"

        for turn_idx in range(turns_per_conversation):
            turn = turn_idx * 2 + 1  # User turns: 1, 3, 5

            # User message
            conversations.append({
                "conversation_id": conv_id,
                "turn": turn,
                "role": "user",
                "content": f"What is {turn_idx + 1} + {turn_idx + 2}?",
                "system": "You are a helpful math assistant" if turn_idx == 0 else None,
                "max_new_tokens": 32,
            })

            # Assistant placeholder (dataset format requirement)
            conversations.append({
                "conversation_id": conv_id,
                "turn": turn + 1,
                "role": "assistant",
                "content": f"The answer is {(turn_idx + 1) + (turn_idx + 2)}.",
            })

    with open(dataset_path, "w") as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset created: {len(conversations)} lines")
    print(f"User messages (HTTP requests): {num_conversations * turns_per_conversation}")

    # Load dataset
    dataset = MultiTurnDataset.load_from_file(str(dataset_path), format=DatasetFormat.JSONL)
    dataset.load()

    print(f"\nDataset loaded:")
    print(f"  Total samples: {dataset.num_samples()}")
    print(f"  Conversations: {dataset.conversation_metadata['num_conversations']}")
    print(f"  Max turns: {dataset.conversation_metadata['max_turns_per_conv']}")

    # Setup runtime settings
    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1,
        max_duration_ms=300000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
        multi_turn_config=MultiTurnConfig(
            enabled=True,
            mode=ConversationMode.PARALLEL,
            turn_timeout_s=60.0,
        ),
    )

    print(f"\nBenchmark configuration:")
    print(f"  Workers: 8")
    print(f"  Mode: {runtime_settings.multi_turn_config.mode}")
    print(f"  Samples to issue: {runtime_settings.n_samples_to_issue}")

    # Create conversation manager and scheduler
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings=runtime_settings,
        sample_order_cls=WithoutReplacementSampleOrder,
        conversation_manager=conversation_manager,
        dataset_metadata=dataset.conversation_metadata,
    )

    print("\n" + "=" * 70)
    print("Starting benchmark...")
    print("=" * 70)

    start_time = time.time()

    with ManagedZMQContext.scoped() as zmq_ctx:
        # Create sample issuer
        sample_issuer = MultiTurnSampleIssuer(endpoint_url, zmq_ctx)

        try:
            # Inject conversation manager into event handler
            SampleEventHandler.set_conversation_manager(conversation_manager)

            # Start benchmark session
            session = BenchmarkSession.start(
                runtime_settings=runtime_settings,
                dataset=dataset,
                sample_issuer=sample_issuer,
                scheduler=scheduler,
                name="concurrent_50_test",
                max_shutdown_timeout_s=120,
            )

            # Wait for completion
            print("\nBenchmark running...")
            session.wait_for_test_end()

            elapsed_time = time.time() - start_time

            print("\n" + "=" * 70)
            print("BENCHMARK COMPLETED")
            print("=" * 70)

            # Get statistics from conversation manager
            total_conversations = len(conversation_manager._conversations)

            print(f"\nConversation Manager Statistics:")
            print(f"  Total conversations tracked: {total_conversations}")

            # Verify all conversations completed
            completed_conversations = 0
            incomplete_conversations = []

            for conv_id, state in conversation_manager._conversations.items():
                # After N user turns (1, 3, 5, ..., 2N-1) and N assistant turns (2, 4, 6, ..., 2N),
                # current_turn should be 2N (last assistant turn completed)
                expected_turn = turns_per_conversation * 2  # After 3 turns: turn 6
                if state.current_turn == expected_turn:
                    completed_conversations += 1
                else:
                    incomplete_conversations.append(
                        f"{conv_id}: turn {state.current_turn} (expected {expected_turn})"
                    )

            print(f"  Completed conversations: {completed_conversations}/{total_conversations}")

            if incomplete_conversations:
                print(f"\nIncomplete conversations: {len(incomplete_conversations)}")
                for incomplete in incomplete_conversations[:5]:
                    print(f"    - {incomplete}")

            # Performance metrics
            total_user_turns = num_conversations * turns_per_conversation

            print(f"\nPerformance Metrics:")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  HTTP requests made: {total_user_turns}")
            print(f"  Throughput: {total_user_turns / elapsed_time:.2f} requests/sec")
            print(f"  Average latency: {elapsed_time / total_user_turns * 1000:.2f}ms per request")

            # Verify correctness
            print(f"\nVerification:")
            print(f"  ✓ Expected conversations: {num_conversations}")
            print(f"  ✓ Actual conversations: {total_conversations}")
            print(f"  ✓ Completed: {completed_conversations}")

            assert total_conversations == num_conversations, \
                f"Expected {num_conversations} conversations, got {total_conversations}"

            assert completed_conversations >= num_conversations * 0.95, \
                f"Less than 95% conversations completed: {completed_conversations}/{num_conversations}"

            print("\n✅ TEST PASSED")
            print("=" * 70)

        finally:
            # Cleanup
            sample_issuer.shutdown()
            sample_issuer.http_client.shutdown()
            SampleEventHandler.clear_hooks()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
def test_100_concurrent_conversations_real_endpoint(tmp_path):
    """Test 100 concurrent conversations with real HTTP requests.

    Scaled up version: 100 conversations × 3 turns = 300 HTTP requests
    """
    endpoint_url = "http://localhost:8868/v1/chat/completions"

    print("\n" + "=" * 70)
    print("REAL INTEGRATION TEST: 100 Concurrent Conversations")
    print("=" * 70)

    num_conversations = 100
    turns_per_conversation = 3

    # Create dataset
    dataset_path = tmp_path / "concurrent_100.jsonl"
    conversations = []

    print(f"\nCreating dataset: {num_conversations} conversations × {turns_per_conversation} turns")

    for conv_idx in range(num_conversations):
        conv_id = f"conv_{conv_idx:03d}"

        for turn_idx in range(turns_per_conversation):
            turn = turn_idx * 2 + 1

            conversations.append({
                "conversation_id": conv_id,
                "turn": turn,
                "role": "user",
                "content": f"Conversation {conv_idx}, turn {turn_idx}: Hello",
                "system": "You are a test assistant" if turn_idx == 0 else None,
                "max_new_tokens": 16,
            })

            conversations.append({
                "conversation_id": conv_id,
                "turn": turn + 1,
                "role": "assistant",
                "content": "Response",
            })

    with open(dataset_path, "w") as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset created with {num_conversations * turns_per_conversation} HTTP requests")

    # Load dataset
    dataset = MultiTurnDataset.load_from_file(str(dataset_path), format=DatasetFormat.JSONL)
    dataset.load()

    # Setup
    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1,
        max_duration_ms=300000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
        multi_turn_config=MultiTurnConfig(
            enabled=True,
            mode=ConversationMode.PARALLEL,
            turn_timeout_s=60.0,
        ),
    )

    print(f"\nBenchmark configuration:")
    print(f"  Workers: 16")

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings=runtime_settings,
        sample_order_cls=WithoutReplacementSampleOrder,
        conversation_manager=conversation_manager,
        dataset_metadata=dataset.conversation_metadata,
    )

    print("\n" + "=" * 70)
    print("Starting benchmark...")
    print("=" * 70)

    start_time = time.time()

    with ManagedZMQContext.scoped() as zmq_ctx:
        sample_issuer = MultiTurnSampleIssuer(endpoint_url, zmq_ctx, num_workers=16)

        try:
            SampleEventHandler.set_conversation_manager(conversation_manager)

            session = BenchmarkSession.start(
                runtime_settings=runtime_settings,
                dataset=dataset,
                sample_issuer=sample_issuer,
                scheduler=scheduler,
                name="concurrent_100_test",
                max_shutdown_timeout_s=120,
            )

            print("\nBenchmark running...")
            session.wait_for_test_end()

            elapsed_time = time.time() - start_time

            print("\n" + "=" * 70)
            print("BENCHMARK COMPLETED")
            print("=" * 70)

            total_conversations = len(conversation_manager._conversations)
            total_requests = num_conversations * turns_per_conversation

            print(f"\nResults:")
            print(f"  Total conversations: {total_conversations}/{num_conversations}")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  HTTP requests: {total_requests}")
            print(f"  Throughput: {total_requests / elapsed_time:.2f} requests/sec")

            # Verify
            # After N user turns and N assistant turns, current_turn should be 2N
            expected_turn = turns_per_conversation * 2
            completed = sum(
                1 for state in conversation_manager._conversations.values()
                if state.current_turn == expected_turn
            )

            print(f"  Completed conversations: {completed}/{num_conversations}")

            assert total_conversations == num_conversations
            assert completed >= num_conversations * 0.95

            print("\n✅ TEST PASSED")
            print("=" * 70)

        finally:
            sample_issuer.shutdown()
            sample_issuer.http_client.shutdown()
            SampleEventHandler.clear_hooks()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)  # Disable pytest timeout for this long-running test
def test_4096_concurrent_conversations_real_endpoint(tmp_path):
    """Test 4096 concurrent conversations with real HTTP requests.

    Extreme scale test: 4096 conversations × 3 turns = 12,288 HTTP requests

    This tests the full system under very high concurrency:
    - Real model inference at port 8868
    - Full worker infrastructure (64 workers)
    - Turn sequencing across thousands of conversations
    - ZMQ transport under load
    """
    endpoint_url = "http://localhost:8868/v1/chat/completions"

    print("\n" + "=" * 70)
    print("EXTREME SCALE TEST: 4096 Concurrent Conversations")
    print("=" * 70)

    num_conversations = 4096
    turns_per_conversation = 3

    # Create dataset
    dataset_path = tmp_path / "concurrent_4096.jsonl"
    conversations = []

    print(f"\nCreating dataset: {num_conversations} conversations × {turns_per_conversation} turns")
    print("This will result in 12,288 HTTP requests to the model endpoint")

    for conv_idx in range(num_conversations):
        conv_id = f"conv_{conv_idx:04d}"

        for turn_idx in range(turns_per_conversation):
            turn = turn_idx * 2 + 1

            conversations.append({
                "conversation_id": conv_id,
                "turn": turn,
                "role": "user",
                "content": f"Question {turn_idx + 1}",
                "system": "You are a test assistant" if turn_idx == 0 else None,
                "max_new_tokens": 16,  # Smaller tokens for faster inference
            })

            conversations.append({
                "conversation_id": conv_id,
                "turn": turn + 1,
                "role": "assistant",
                "content": "Response",
            })

    with open(dataset_path, "w") as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset created with {num_conversations * turns_per_conversation} HTTP requests")

    # Load dataset
    dataset = MultiTurnDataset.load_from_file(str(dataset_path), format=DatasetFormat.JSONL)
    dataset.load()

    # Setup
    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1,
        max_duration_ms=1800000,  # 30 minutes max
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
        multi_turn_config=MultiTurnConfig(
            enabled=True,
            mode=ConversationMode.PARALLEL,
            turn_timeout_s=300.0,  # Increased timeout for extreme scale
        ),
    )

    print(f"\nBenchmark configuration:")
    print(f"  Workers: 64")
    print(f"  Mode: {runtime_settings.multi_turn_config.mode}")
    print(f"  Max duration: {runtime_settings.max_duration_ms / 1000}s")
    print(f"  Turn timeout: {runtime_settings.multi_turn_config.turn_timeout_s}s")

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings=runtime_settings,
        sample_order_cls=WithoutReplacementSampleOrder,
        conversation_manager=conversation_manager,
        dataset_metadata=dataset.conversation_metadata,
    )

    print("\n" + "=" * 70)
    print("Starting benchmark...")
    print("⚠️  This will take approximately 10-15 minutes")
    print("=" * 70)

    start_time = time.time()

    with ManagedZMQContext.scoped() as zmq_ctx:
        sample_issuer = MultiTurnSampleIssuer(endpoint_url, zmq_ctx, num_workers=64)

        try:
            SampleEventHandler.set_conversation_manager(conversation_manager)

            session = BenchmarkSession.start(
                runtime_settings=runtime_settings,
                dataset=dataset,
                sample_issuer=sample_issuer,
                scheduler=scheduler,
                name="concurrent_4096_test",
                max_shutdown_timeout_s=600,  # 10 minutes for shutdown
            )

            print("\nBenchmark running...")

            # Progress monitoring
            import threading

            stop_monitor = threading.Event()

            def progress_monitor():
                while not stop_monitor.is_set():
                    time.sleep(10)
                    if not stop_monitor.is_set():
                        elapsed = time.time() - start_time
                        completed = len([s for s in conversation_manager._conversations.values()
                                       if s.current_turn >= 2])
                        print(f"  [{elapsed:.0f}s] Conversations with ≥1 turn complete: {completed}/{num_conversations}")

            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()

            session.wait_for_test_end()

            # Stop progress monitor
            stop_monitor.set()
            monitor_thread.join(timeout=1)

            elapsed_time = time.time() - start_time

            print("\n" + "=" * 70)
            print("BENCHMARK COMPLETED")
            print("=" * 70)

            total_conversations = len(conversation_manager._conversations)
            total_requests = num_conversations * turns_per_conversation

            print(f"\nResults:")
            print(f"  Total conversations: {total_conversations}/{num_conversations}")
            print(f"  Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} minutes)")
            print(f"  HTTP requests: {total_requests}")
            print(f"  Throughput: {total_requests / elapsed_time:.2f} requests/sec")

            # Verify
            expected_turn = turns_per_conversation * 2
            completed = sum(
                1 for state in conversation_manager._conversations.values()
                if state.current_turn == expected_turn
            )

            incomplete = total_conversations - completed

            print(f"  Completed conversations: {completed}/{num_conversations} ({100 * completed / num_conversations:.1f}%)")
            if incomplete > 0:
                print(f"  ⚠️  Incomplete conversations: {incomplete}")

            # Get some statistics on turn distribution
            turn_distribution = {}
            for state in conversation_manager._conversations.values():
                turn_distribution[state.current_turn] = turn_distribution.get(state.current_turn, 0) + 1

            print(f"\nTurn distribution:")
            for turn in sorted(turn_distribution.keys()):
                count = turn_distribution[turn]
                print(f"    Turn {turn}: {count} conversations ({100 * count / total_conversations:.1f}%)")

            # More lenient assertions for extreme scale
            assert total_conversations >= num_conversations * 0.95, \
                f"Expected at least 95% conversations tracked, got {total_conversations}/{num_conversations}"

            assert completed >= num_conversations * 0.90, \
                f"Less than 90% conversations completed: {completed}/{num_conversations}"

            print("\n✅ TEST PASSED (90% completion threshold for extreme scale)")
            print("=" * 70)

        finally:
            sample_issuer.shutdown()
            sample_issuer.http_client.shutdown()
            SampleEventHandler.clear_hooks()
