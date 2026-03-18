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

"""Tests for TokenizePool thread-safety and correctness.

Uses a FakeTokenizer to avoid downloading models from HuggingFace.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    TokenizePool,
)


class FakeTokenizer:
    """Simple whitespace tokenizer that avoids HuggingFace downloads."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture(autouse=True)
def _mock_tokenizer():
    """Patch _get_thread_tokenizer so TokenizePool never hits the network."""
    fake = FakeTokenizer()
    with patch(
        "inference_endpoint.async_utils.services.metrics_aggregator.token_metrics._get_thread_tokenizer",
        return_value=fake,
    ):
        yield


@pytest.mark.unit
class TestTokenizePool:
    def test_tokenize_returns_tokens(self):
        with TokenizePool("fake", n_workers=1) as pool:
            tokens = pool.tokenize("Hello world")
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)

    def test_token_count_returns_int(self):
        with TokenizePool("fake", n_workers=1) as pool:
            count = pool.token_count("Hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_count_matches_tokenize(self):
        with TokenizePool("fake", n_workers=1) as pool:
            text = "The quick brown fox jumps over the lazy dog"
            tokens = pool.tokenize(text)
            count = pool.token_count(text)
            assert count > 0
            assert len(tokens) > 0

    def test_multiple_workers(self):
        with TokenizePool("fake", n_workers=4) as pool:
            results = [pool.token_count(f"Sentence number {i}") for i in range(10)]
            assert all(isinstance(r, int) and r > 0 for r in results)

    def test_concurrent_thread_safe(self):
        with TokenizePool("fake", n_workers=2) as pool:
            texts = [f"This is test sentence number {i}" for i in range(20)]
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(pool.token_count, t) for t in texts]
                results = [f.result() for f in futures]
            assert len(results) == 20
            assert all(isinstance(r, int) and r > 0 for r in results)

    @pytest.mark.parametrize(
        "case_desc, action, error_type, error_msg",
        [
            ("close idempotent", "close_twice", None, ""),
            ("use after close", "tokenize_after_close", RuntimeError, "closed"),
            ("n_workers=0", "zero_workers", ValueError, "n_workers"),
        ],
    )
    def test_error_cases(self, case_desc, action, error_type, error_msg):
        if action == "close_twice":
            pool = TokenizePool("fake", n_workers=1)
            pool.close()
            pool.close()
        elif action == "tokenize_after_close":
            pool = TokenizePool("fake", n_workers=1)
            pool.close()
            with pytest.raises(error_type, match=error_msg):
                pool.tokenize("hello")
        elif action == "zero_workers":
            with pytest.raises(error_type, match=error_msg):
                TokenizePool("fake", n_workers=0)

    @pytest.mark.asyncio
    async def test_token_count_async(self):
        loop = asyncio.get_running_loop()
        with TokenizePool("fake", n_workers=1) as pool:
            count = await pool.token_count_async("Hello world", loop)
            assert isinstance(count, int)
            assert count > 0

    def test_context_manager(self):
        with TokenizePool("fake", n_workers=1) as pool:
            assert pool.token_count("test") > 0
        with pytest.raises(RuntimeError, match="closed"):
            pool.tokenize("test")
