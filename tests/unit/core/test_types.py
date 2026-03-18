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
Unit tests for core type serialization using msgspec.msgpack.

Tests verify that Query, QueryResult, and StreamChunk can be properly
serialized and deserialized with various field combinations.
"""

import time

import msgspec
import pytest
from inference_endpoint.core.types import (
    ErrorData,
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)


class TestErrorData:
    """Test ErrorData string representation."""

    @pytest.mark.parametrize(
        "case_desc, error_type, error_message, expected_str",
        [
            (
                "with message",
                "ValueError",
                "invalid value",
                "ValueError: invalid value",
            ),
            ("empty message", "TimeoutError", "", "TimeoutError"),
        ],
    )
    def test_error_data_str(self, case_desc, error_type, error_message, expected_str):
        err = ErrorData(error_type=error_type, error_message=error_message)
        assert str(err) == expected_str


class TestQuerySerialization:
    """Test Query msgspec.msgpack serialization with various field combinations."""

    @pytest.mark.parametrize(
        "case_desc, query_kwargs",
        [
            ("empty defaults", {}),
            (
                "simple data",
                {
                    "data": {
                        "prompt": "Hello, world!",
                        "model": "gpt-4",
                        "max_tokens": 100,
                    }
                },
            ),
            (
                "with headers",
                {
                    "data": {"prompt": "Test"},
                    "headers": {
                        "Authorization": "Bearer token123",
                        "Content-Type": "application/json",
                    },
                },
            ),
            (
                "complex nested data",
                {
                    "data": {
                        "prompt": "Complex prompt",
                        "model": "gpt-4",
                        "parameters": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "frequency_penalty": 0.0,
                        },
                        "messages": [
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "Hello"},
                        ],
                        "stream": True,
                        "n": 1,
                    }
                },
            ),
            ("custom id", {"id": "custom-query-id-12345", "data": {"test": "value"}}),
            ("custom timestamp", {"created_at": 1234567890.123456}),
            (
                "all fields populated",
                {
                    "id": "test-query-001",
                    "data": {"prompt": "Full test", "max_tokens": 50},
                    "headers": {"X-Custom": "header-value"},
                    "created_at": 1700000000.0,
                },
            ),
        ],
    )
    def test_query_roundtrip(self, case_desc, query_kwargs):
        query = Query(**query_kwargs)
        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.id == query.id
        assert decoded.data == query.data
        assert decoded.headers == query.headers
        assert decoded.created_at == query.created_at
        assert isinstance(decoded.id, str) and len(decoded.id) > 0

    def test_query_multiple_roundtrips(self):
        """Test Query survives multiple serialization roundtrips."""
        original = Query(data={"test": "data"}, headers={"auth": "token"})

        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=Query)

        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=Query)

        assert decoded2.id == original.id
        assert decoded2.data == original.data
        assert decoded2.headers == original.headers
        assert decoded2.created_at == original.created_at


class TestQueryResultSerialization:
    """Test QueryResult msgspec.msgpack serialization with various field combinations."""

    @pytest.mark.parametrize(
        "case_desc, response_output, expected_output, expected_reasoning",
        [
            (
                "string response",
                TextModelOutput(output="This is a complete response from the model."),
                "This is a complete response from the model.",
                None,
            ),
            (
                "tuple output",
                TextModelOutput(
                    output=("First chunk", "Second chunk", "Final chunk"),
                    reasoning=None,
                ),
                ("First chunk", "Second chunk", "Final chunk"),
                None,
            ),
            (
                "list converts to tuple",
                TextModelOutput(
                    output=["Chunk 1", "Chunk 2", "Chunk 3"], reasoning=None
                ),
                ("Chunk 1", "Chunk 2", "Chunk 3"),
                None,
            ),
            (
                "output only list",
                TextModelOutput(
                    output=["First chunk", "rest of output"], reasoning=None
                ),
                ("First chunk", "rest of output"),
                None,
            ),
            (
                "output and reasoning",
                TextModelOutput(
                    output="Final output text",
                    reasoning=["First reasoning chunk", "rest of reasoning"],
                ),
                "Final output text",
                ("First reasoning chunk", "rest of reasoning"),
            ),
            (
                "empty list to tuple",
                TextModelOutput(output=[], reasoning=None),
                (),
                None,
            ),
            (
                "empty string",
                TextModelOutput(output=""),
                "",
                None,
            ),
            (
                "empty tuple",
                TextModelOutput(output=(), reasoning=None),
                (),
                None,
            ),
        ],
    )
    def test_response_output_roundtrip(
        self, case_desc, response_output, expected_output, expected_reasoning
    ):
        result = QueryResult(id=f"query-{case_desc}", response_output=response_output)
        assert result.response_output.output == expected_output

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == expected_output
        assert decoded.response_output.reasoning == expected_reasoning

    def test_query_result_minimal(self):
        """Test QueryResult with minimal required fields."""
        result = QueryResult(id="query-123")

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.id == "query-123"
        assert decoded.response_output is None
        assert decoded.metadata == {}
        assert decoded.error is None
        assert isinstance(decoded.completed_at, int)

    def test_query_result_with_metadata(self):
        """Test QueryResult with comprehensive metadata."""
        result = QueryResult(
            id="query-meta",
            response_output=TextModelOutput(output="Response text"),
            metadata={
                "model": "gpt-4",
                "tokens_used": 150,
                "finish_reason": "stop",
                "latency_ms": 234.5,
                "cache_hit": False,
                "provider": "openai",
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["model"] == "gpt-4"
        assert decoded.metadata["tokens_used"] == 150
        assert decoded.metadata["finish_reason"] == "stop"
        assert decoded.metadata["latency_ms"] == 234.5
        assert decoded.metadata["cache_hit"] is False

    def test_query_result_with_error(self):
        """Test QueryResult with ErrorData."""
        result = QueryResult(
            id="query-error",
            error=ErrorData(
                error_type="TimeoutError",
                error_message="Connection timeout after 30 seconds",
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.error is not None
        assert decoded.error.error_type == "TimeoutError"
        assert decoded.error.error_message == "Connection timeout after 30 seconds"
        assert decoded.response_output is None

    def test_query_result_error_with_partial(self):
        """Test QueryResult with both error and partial response."""
        result = QueryResult(
            id="query-partial",
            response_output=TextModelOutput(output="Partial response before error"),
            error=ErrorData(
                error_type="ConnectionError",
                error_message="Server disconnected during streaming",
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.response_output == TextModelOutput(
            output="Partial response before error"
        )
        assert decoded.error is not None
        assert decoded.error.error_message == "Server disconnected during streaming"

    def test_query_result_all_fields(self):
        """Test QueryResult with all fields fully populated."""
        result = QueryResult(
            id="query-full",
            response_output=TextModelOutput(
                output=("Chunk 1", "Chunk 2"), reasoning=None
            ),
            metadata={
                "model": "llama-2-70b",
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
            error=None,
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.id == "query-full"
        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("Chunk 1", "Chunk 2")
        assert decoded.metadata["total_tokens"] == 150
        assert decoded.error is None

    def test_query_result_immutability(self):
        """Test QueryResult is frozen and cannot be modified."""
        result = QueryResult(
            id="query-frozen", response_output=TextModelOutput(output="Original text")
        )

        with pytest.raises(AttributeError):
            result.response_output = "Modified text"

        with pytest.raises(AttributeError):
            result.error = ErrorData(error_type="x", error_message="y")

    def test_query_result_auto_completed_at(self):
        """Test QueryResult completed_at is automatically set in __post_init__."""
        before = time.monotonic_ns()
        result = QueryResult(id="query-timestamp")
        after = time.monotonic_ns()

        assert before <= result.completed_at <= after
        assert isinstance(result.completed_at, int | float)

    def test_query_result_multiple_roundtrips(self):
        """Test QueryResult survives multiple serialization roundtrips."""
        original = QueryResult(
            id="query-roundtrip",
            response_output=TextModelOutput(
                output=("Chunk A", "Chunk B"), reasoning=None
            ),
            metadata={"tokens": 42},
        )

        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=QueryResult)

        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=QueryResult)

        assert decoded2.id == original.id
        assert isinstance(decoded2.response_output, TextModelOutput)
        assert decoded2.response_output.output == original.response_output.output
        assert decoded2.metadata == original.metadata


class TestStreamChunkSerialization:
    """Test StreamChunk msgspec.msgpack serialization with various field combinations."""

    @pytest.mark.parametrize(
        "case_desc, chunk_kwargs",
        [
            ("minimal", {}),
            (
                "basic content",
                {
                    "id": "query-123",
                    "response_chunk": "Hello, this is a chunk of text.",
                },
            ),
            (
                "first chunk with metadata",
                {
                    "id": "query-456",
                    "response_chunk": "First token",
                    "is_complete": False,
                    "metadata": {"first_chunk": True, "latency_ns": 1234567},
                },
            ),
            (
                "final chunk",
                {
                    "id": "query-789",
                    "response_chunk": "Final text.",
                    "is_complete": True,
                },
            ),
            (
                "comprehensive metadata",
                {
                    "id": "query-meta",
                    "response_chunk": " next token",
                    "is_complete": False,
                    "metadata": {
                        "model": "llama-2-70b",
                        "chunk_index": 5,
                        "tokens_so_far": 50,
                        "timestamp_ns": 1700000000000000,
                        "first_chunk": False,
                    },
                },
            ),
            ("empty response", {"id": "query-empty", "response_chunk": ""}),
            (
                "special characters",
                {
                    "id": "query-unicode",
                    "response_chunk": "Hello 世界! 🚀 Special chars: \n\t\r",
                },
            ),
            (
                "all fields populated",
                {
                    "id": "query-full-chunk",
                    "response_chunk": "Complete chunk text",
                    "is_complete": True,
                    "metadata": {
                        "model": "gpt-4",
                        "finish_reason": "stop",
                        "total_tokens": 100,
                    },
                },
            ),
        ],
    )
    def test_stream_chunk_roundtrip(self, case_desc, chunk_kwargs):
        original = StreamChunk(**chunk_kwargs)
        encoded = msgspec.msgpack.encode(original)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.id == original.id
        assert decoded.response_chunk == original.response_chunk
        assert decoded.is_complete == original.is_complete
        assert decoded.metadata == original.metadata

    def test_stream_chunk_multiple_roundtrips(self):
        """Test StreamChunk survives multiple serialization roundtrips."""
        original = StreamChunk(
            id="query-roundtrip",
            response_chunk="Test chunk",
            is_complete=False,
            metadata={"index": 1},
        )

        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=StreamChunk)

        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=StreamChunk)

        assert decoded2.id == original.id
        assert decoded2.response_chunk == original.response_chunk
        assert decoded2.is_complete == original.is_complete
        assert decoded2.metadata == original.metadata


class TestQueryResultWorkerPatterns:
    """Test QueryResult serialization patterns used by worker.py (TextModelOutput)."""

    @pytest.mark.parametrize(
        "case_desc, response_output, expected_output, expected_reasoning",
        [
            (
                "reasoning chunks",
                TextModelOutput(
                    output="The answer is 42",
                    reasoning=["Let me think...", " step by step to solve this"],
                ),
                "The answer is 42",
                ("Let me think...", " step by step to solve this"),
            ),
            (
                "output only",
                TextModelOutput(output=["Hello", " world!"], reasoning=None),
                ("Hello", " world!"),
                None,
            ),
            (
                "no chunks",
                TextModelOutput(output=[], reasoning=None),
                (),
                None,
            ),
            (
                "single reasoning chunk",
                TextModelOutput(output="Answer", reasoning=["Quick thought"]),
                "Answer",
                ("Quick thought",),
            ),
            (
                "single output chunk",
                TextModelOutput(output=["SingleResponse"], reasoning=None),
                ("SingleResponse",),
                None,
            ),
        ],
    )
    def test_worker_pattern(
        self, case_desc, response_output, expected_output, expected_reasoning
    ):
        result = QueryResult(
            id=f"query-{case_desc}",
            response_output=response_output,
            metadata={"first_chunk": False, "final_chunk": True},
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == expected_output
        assert decoded.response_output.reasoning == expected_reasoning


class TestMixedTypeSerialization:
    """Test serialization of mixed type combinations and edge cases."""

    def test_serialize_list_of_queries(self):
        """Test serializing a list of Query objects."""
        queries = [
            Query(data={"prompt": "Query 1"}),
            Query(data={"prompt": "Query 2"}),
            Query(data={"prompt": "Query 3"}),
        ]

        encoded = msgspec.msgpack.encode(queries)
        decoded = msgspec.msgpack.decode(encoded, type=list[Query])

        assert len(decoded) == 3
        assert decoded[0].data["prompt"] == "Query 1"
        assert decoded[2].data["prompt"] == "Query 3"

    def test_serialize_list_of_results(self):
        """Test serializing a list of QueryResult objects."""
        results = [
            QueryResult(id="r1", response_output=TextModelOutput(output="Response 1")),
            QueryResult(id="r2", response_output=TextModelOutput(output="Response 2")),
            QueryResult(
                id="r3",
                error=ErrorData(
                    error_type="RuntimeError", error_message="Error in query 3"
                ),
            ),
        ]

        encoded = msgspec.msgpack.encode(results)
        decoded = msgspec.msgpack.decode(encoded, type=list[QueryResult])

        assert len(decoded) == 3
        assert decoded[0].response_output == TextModelOutput(output="Response 1")
        assert decoded[2].error is not None
        assert decoded[2].error.error_type == "RuntimeError"
        assert decoded[2].error.error_message == "Error in query 3"

    def test_serialize_list_of_chunks(self):
        """Test serializing a list of StreamChunk objects."""
        chunks = [
            StreamChunk(
                id="q1", response_chunk="First", metadata={"first_chunk": True}
            ),
            StreamChunk(id="q1", response_chunk=" second"),
            StreamChunk(id="q1", response_chunk=" final", is_complete=True),
        ]

        encoded = msgspec.msgpack.encode(chunks)
        decoded = msgspec.msgpack.decode(encoded, type=list[StreamChunk])

        assert len(decoded) == 3
        assert decoded[0].metadata.get("first_chunk") is True
        assert decoded[2].is_complete is True

    def test_nested_metadata(self):
        """Test QueryResult with deeply nested metadata and TextModelOutput."""
        result = QueryResult(
            id="query-nested",
            response_output=TextModelOutput(
                output=["First chunk", "remaining output"],
                reasoning=["Reasoning process"],
            ),
            metadata={
                "model_info": {
                    "name": "gpt-4",
                    "version": "2024-01",
                    "parameters": {"temperature": 0.7, "top_p": 0.9},
                },
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "breakdown": [5, 5, 10, 10],
                },
                "tags": ["production", "high-priority"],
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["model_info"]["parameters"]["temperature"] == 0.7
        assert decoded.metadata["usage"]["breakdown"] == [5, 5, 10, 10]
        assert "production" in decoded.metadata["tags"]
        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("First chunk", "remaining output")
        assert decoded.response_output.reasoning == ("Reasoning process",)

    def test_none_values_in_data(self):
        """Test Query with None values in data dict."""
        query = Query(
            data={"prompt": "Test", "optional_param": None, "another_param": None}
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.data["optional_param"] is None
        assert decoded.data["another_param"] is None

    def test_large_payload(self):
        """Test serialization of large payloads."""
        large_text = "A" * 10000

        query = Query(data={"prompt": large_text})
        encoded_query = msgspec.msgpack.encode(query)
        decoded_query = msgspec.msgpack.decode(encoded_query, type=Query)
        assert decoded_query.data["prompt"] == large_text

        result = QueryResult(
            id="large", response_output=TextModelOutput(output=large_text)
        )
        encoded_result = msgspec.msgpack.encode(result)
        decoded_result = msgspec.msgpack.decode(encoded_result, type=QueryResult)
        assert decoded_result.response_output == TextModelOutput(output=large_text)

    def test_numeric_types_in_metadata(self):
        """Test various numeric types in metadata."""
        result = QueryResult(
            id="numeric-test",
            response_output=TextModelOutput(output="Text"),
            metadata={
                "int_value": 42,
                "float_value": 3.14159,
                "large_int": 9999999999999999,
                "negative": -123.456,
                "zero": 0,
                "zero_float": 0.0,
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["int_value"] == 42
        assert abs(decoded.metadata["float_value"] - 3.14159) < 0.00001
        assert decoded.metadata["large_int"] == 9999999999999999
        assert decoded.metadata["negative"] == -123.456
        assert decoded.metadata["zero"] == 0
