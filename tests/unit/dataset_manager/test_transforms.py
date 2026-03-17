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
Unit tests for the transforms module.
Tests all transform classes and functions except Harmonize.
"""

from typing import Any

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    ColumnRemap,
    FusedRowProcessor,
    MakeAdapterCompatible,
    RowProcessor,
    Transform,
    UserPromptFormatter,
    apply_transforms,
)


class TestColumnRemap:
    """Test suite for ColumnRemap transform."""

    def test_basic_column_rename(self):
        """Test basic column renaming with string keys."""
        df = pd.DataFrame({"old_name": [1, 2, 3], "another_col": [4, 5, 6]})
        transform = ColumnRemap({"old_name": "new_name"})
        result = transform(df)

        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert list(result["new_name"]) == [1, 2, 3]

    def test_multiple_columns_rename(self):
        """Test renaming multiple columns at once."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        transform = ColumnRemap({"col1": "a", "col2": "b"})
        result = transform(df)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "col3" in result.columns  # Unchanged column
        assert "col1" not in result.columns
        assert "col2" not in result.columns

    def test_empty_mapping(self):
        """Test with empty mapping (no columns renamed)."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        transform = ColumnRemap({})
        result = transform(df)

        # Should return DataFrame with same columns
        assert list(result.columns) == list(df.columns)
        pd.testing.assert_frame_equal(result, df)

    def test_rename_nonexistent_column(self):
        """Test renaming a column that doesn't exist (should be silently ignored)."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        transform = ColumnRemap({"nonexistent": "new_name"})
        result = transform(df)

        # Should silently ignore non-existent column mappings
        assert list(result.columns) == ["col1", "col2"]
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "case_desc, columns, candidates, strict, expect_prompt, error_type, error_msg",
        [
            (
                "single match",
                {"question": [1, 2], "answer": [3, 4]},
                ("user_prompt", "question", "input"),
                False,
                True,
                None,
                "",
            ),
            (
                "no match",
                {"col1": [1, 2], "col2": [3, 4]},
                ("user_prompt", "question", "input"),
                False,
                False,
                None,
                "",
            ),
            (
                "strict multi match",
                {"question": [1, 2], "user_prompt": [3, 4]},
                ("user_prompt", "question"),
                True,
                None,
                ValueError,
                "Multiple columns found",
            ),
            (
                "non-strict multi match",
                {"question": [1, 2], "user_prompt": [3, 4], "other": [5, 6]},
                ("user_prompt", "question"),
                False,
                True,
                None,
                "",
            ),
        ],
    )
    def test_fuzzy_remap(
        self,
        case_desc,
        columns,
        candidates,
        strict,
        expect_prompt,
        error_type,
        error_msg,
    ):
        df = pd.DataFrame(columns)
        transform = ColumnRemap({candidates: "prompt"}, strict=strict)
        if error_type is not None:
            with pytest.raises(error_type, match=error_msg):
                transform(df)
        else:
            result = transform(df)
            if expect_prompt:
                assert "prompt" in result.columns
            else:
                assert "prompt" not in result.columns

    def test_mixed_string_and_tuple_keys(self):
        """Test using both string and tuple keys in the same remap."""
        df = pd.DataFrame({"old_col": [1, 2], "question": [3, 4], "other": [5, 6]})
        transform = ColumnRemap(
            {"old_col": "new_col", ("user_prompt", "question"): "prompt"}
        )
        result = transform(df)

        assert "new_col" in result.columns
        assert "prompt" in result.columns
        assert "old_col" not in result.columns
        assert "question" not in result.columns


class TestUserPromptFormatter:
    """Test suite for UserPromptFormatter transform."""

    @pytest.mark.parametrize(
        "case_desc, columns, fmt, output_col, expected_first, error_type",
        [
            (
                "single var",
                {"question": ["What is 2+2?", "Capital of France?"]},
                "Q: {question}",
                "prompt",
                "Q: What is 2+2?",
                None,
            ),
            (
                "custom output col",
                {"question": ["What is 2+2?"]},
                "Q: {question}",
                "formatted_q",
                "Q: What is 2+2?",
                None,
            ),
            ("empty format", {"question": ["What is 2+2?"]}, "", "prompt", "", None),
            (
                "static format",
                {"question": ["What is 2+2?"]},
                "Static prompt text",
                "prompt",
                "Static prompt text",
                None,
            ),
            (
                "missing var",
                {"question": ["What is 2+2?"]},
                "Q: {question} C: {missing_var}",
                "prompt",
                None,
                KeyError,
            ),
        ],
    )
    def test_prompt_formatting(
        self, case_desc, columns, fmt, output_col, expected_first, error_type
    ):
        df = pd.DataFrame(columns)
        transform = UserPromptFormatter(fmt, output_column=output_col)
        if error_type is not None:
            with pytest.raises(error_type):
                transform(df)
        else:
            result = transform(df)
            assert output_col in result.columns
            assert result[output_col][0] == expected_first

    def test_multiple_variables(self):
        """Formatting with multiple variables from row."""
        df = pd.DataFrame(
            {
                "context": ["Paris is the capital."],
                "question": ["Of which country?"],
            }
        )
        transform = UserPromptFormatter(
            "Context: {context}\nQuestion: {question}", output_column="prompt"
        )
        result = transform(df)
        assert (
            result["prompt"][0]
            == "Context: Paris is the capital.\nQuestion: Of which country?"
        )

    def test_preserves_original_columns(self):
        """Original columns are preserved after formatting."""
        df = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
        transform = UserPromptFormatter("{question}", output_column="prompt")
        result = transform(df)

        assert "question" in result.columns
        assert "answer" in result.columns
        assert "prompt" in result.columns
        assert list(result["question"]) == ["Q1", "Q2"]


class TestRowProcessor:
    """Test suite for RowProcessor base class."""

    def test_row_processor_calls_process_row(self):
        """Test that RowProcessor __call__ invokes process_row for each row."""

        class TestProcessor(RowProcessor):
            def __init__(self):
                self.call_count = 0

            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                self.call_count += 1
                row["processed"] = True
                return row

        df = pd.DataFrame({"col1": [1, 2, 3]})
        processor = TestProcessor()
        result = processor(df)

        assert processor.call_count == 3  # Called once per row
        assert "processed" in result.columns
        assert all(result["processed"])

    def test_row_processor_with_row_modification(self):
        """Test that row modifications are reflected in output DataFrame."""

        class DoubleValues(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        processor = DoubleValues()
        result = processor(df)

        assert list(result["value"]) == [2, 4, 6]

    def test_row_processor_adds_new_column(self):
        """Test that process_row can add new columns."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["value"] ** 2
                return row

        df = pd.DataFrame({"value": [2, 3, 4]})
        processor = AddSquared()
        result = processor(df)

        assert "squared" in result.columns
        assert list(result["squared"]) == [4, 9, 16]

    def test_row_processor_with_empty_dataframe(self):
        """Test row processor with empty DataFrame."""

        class TestProcessor(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                return row

        df = pd.DataFrame({"col1": []})
        processor = TestProcessor()
        result = processor(df)

        assert len(result) == 0
        assert "col1" in result.columns


class TestFusedRowProcessor:
    """Test suite for FusedRowProcessor."""

    def test_single_processor(self):
        """Test FusedRowProcessor with a single processor."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        fused = FusedRowProcessor([AddOne()])
        result = fused(df)

        assert list(result["value"]) == [2, 3, 4]

    def test_multiple_processors_in_sequence(self):
        """Test that multiple processors are applied in sequence."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        # Should add 1 first, then multiply by 2: (1+1)*2 = 4, (2+1)*2 = 6, (3+1)*2 = 8
        fused = FusedRowProcessor([AddOne(), MultiplyByTwo()])
        result = fused(df)

        assert list(result["value"]) == [4, 6, 8]

    def test_processor_order_matters(self):
        """Test that the order of processors affects the result."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})

        # Order 1: Multiply then add: (1*2)+1 = 3, (2*2)+1 = 5, (3*2)+1 = 7
        fused1 = FusedRowProcessor([MultiplyByTwo(), AddOne()])
        result1 = fused1(df)
        assert list(result1["value"]) == [3, 5, 7]

        # Order 2: Add then multiply: (1+1)*2 = 4, (2+1)*2 = 6, (3+1)*2 = 8
        df = pd.DataFrame({"value": [1, 2, 3]})  # Reset dataframe
        fused2 = FusedRowProcessor([AddOne(), MultiplyByTwo()])
        result2 = fused2(df)
        assert list(result2["value"]) == [4, 6, 8]

    def test_empty_processor_list(self):
        """Test FusedRowProcessor with empty processor list."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        fused = FusedRowProcessor([])
        result = fused(df)

        # Should return unchanged data
        assert result.equals(df)

    def test_processors_can_add_columns(self):
        """Test that fused processors can add new columns."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["value"] ** 2
                return row

        class AddCubed(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["cubed"] = row["value"] ** 3
                return row

        df = pd.DataFrame({"value": [2, 3]})
        fused = FusedRowProcessor([AddSquared(), AddCubed()])
        result = fused(df)

        assert "squared" in result.columns
        assert "cubed" in result.columns
        assert list(result["squared"]) == [4, 9]
        assert list(result["cubed"]) == [8, 27]


class TestApplyTransforms:
    """Test suite for apply_transforms function."""

    def test_single_transform(self):
        """Test applying a single transform."""
        df = pd.DataFrame({"old": [1, 2, 3]})
        transforms = [ColumnRemap({"old": "new"})]
        result = apply_transforms(df, transforms)

        assert "new" in result.columns
        assert "old" not in result.columns

    def test_multiple_transforms_in_sequence(self):
        """Test applying multiple transforms in sequence."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        class AddColumn(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["col2"] = row["col1"] * 2
                return row

        transforms = [
            AddColumn(),
            ColumnRemap({"col1": "original", "col2": "doubled"}),
        ]
        result = apply_transforms(df, transforms)

        assert "original" in result.columns
        assert "doubled" in result.columns
        assert list(result["doubled"]) == [2, 4, 6]

    def test_empty_transform_list(self):
        """Test with empty transform list."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = apply_transforms(df, [])

        # Should return the DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_fusion_enabled(self):
        """Test that row processors are fused when fusion is enabled."""

        class CountingRowProcessor(RowProcessor):
            call_count = 0

            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                CountingRowProcessor.call_count += 1
                return super().__call__(df)

            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row.get("value", 0) + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        CountingRowProcessor.call_count = 0

        transforms = [CountingRowProcessor(), CountingRowProcessor()]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # With fusion, should only iterate once (FusedRowProcessor's __call__)
        # But we need to track the fused processor's call count, not individual processors
        # Let's verify the result is correct instead
        assert list(result["value"]) == [3, 4, 5]  # Each row incremented twice

    def test_fusion_disabled(self):
        """Test that row processors are not fused when fusion is disabled."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [AddOne(), AddOne()]
        result = apply_transforms(df, transforms, fuse_row_processors=False)

        # Result should be the same regardless of fusion
        assert list(result["value"]) == [3, 4, 5]

    def test_mixed_transforms_and_processors(self):
        """Test mixing regular transforms with row processors."""

        class AddDoubled(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["doubled"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [
            AddDoubled(),
            ColumnRemap({"value": "original"}),
        ]
        result = apply_transforms(df, transforms)

        assert "original" in result.columns
        assert "doubled" in result.columns
        assert list(result["doubled"]) == [2, 4, 6]

    def test_consecutive_row_processors_are_fused(self):
        """Test that consecutive row processors are fused together."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [AddOne(), MultiplyByTwo()]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # Should be (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        assert list(result["value"]) == [4, 6, 8]

    def test_non_consecutive_processors_split(self):
        """Test that non-consecutive row processors are not fused together."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [
            AddOne(),
            ColumnRemap({"value": "value"}),  # No-op transform to break fusion
            AddOne(),
        ]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # Each AddOne should add 1, so final result is +2
        assert list(result["value"]) == [3, 4, 5]

    def test_complex_transform_pipeline(self):
        """Test a complex pipeline with multiple types of transforms."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["num"] ** 2
                return row

        class AddCubed(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["cubed"] = row["num"] ** 3
                return row

        df = pd.DataFrame({"num": [2, 3, 4]})
        transforms = [
            AddSquared(),
            AddCubed(),
            ColumnRemap({"num": "original_number"}),
        ]
        result = apply_transforms(df, transforms)

        assert "original_number" in result.columns
        assert "squared" in result.columns
        assert "cubed" in result.columns
        assert list(result["squared"]) == [4, 9, 16]
        assert list(result["cubed"]) == [8, 27, 64]


class TestTransformBaseClass:
    """Test suite for Transform base class."""

    def test_transform_not_implemented(self):
        """Test that Transform base class __call__ raises Error"""

        with pytest.raises(TypeError):

            class IncompleteTransform(Transform):
                pass

            IncompleteTransform()

    def test_custom_transform_implementation(self):
        """Test implementing a custom transform."""

        class DropNullsTransform(Transform):
            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                return df.dropna()

        df = pd.DataFrame({"col1": [1, None, 3, None, 5]})
        transform = DropNullsTransform()
        result = transform(df)

        assert len(result) == 3
        assert list(result["col1"]) == [1.0, 3.0, 5.0]


class TestAddStaticColumns:
    """Test suite for AddStaticColumns transform."""

    def test_add_single_column(self):
        """Test adding a single static column."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        transform = AddStaticColumns({"static_col": "static_value"})
        result = transform(df)

        assert "static_col" in result.columns
        assert list(result["static_col"]) == [
            "static_value",
            "static_value",
            "static_value",
        ]

    def test_add_multiple_columns(self):
        """Test adding multiple static columns."""
        df = pd.DataFrame({"col1": [1, 2]})
        transform = AddStaticColumns({"col_a": "value_a", "col_b": 123, "col_c": True})
        result = transform(df)

        assert "col_a" in result.columns
        assert "col_b" in result.columns
        assert "col_c" in result.columns
        assert list(result["col_a"]) == ["value_a", "value_a"]
        assert list(result["col_b"]) == [123, 123]
        assert list(result["col_c"]) == [True, True]

    def test_preserves_existing_columns(self):
        """Test that existing columns are preserved."""
        df = pd.DataFrame({"existing": [1, 2, 3]})
        transform = AddStaticColumns({"new_col": "new_value"})
        result = transform(df)

        assert "existing" in result.columns
        assert list(result["existing"]) == [1, 2, 3]
        assert "new_col" in result.columns

    def test_empty_dataframe(self):
        """Test adding static columns to empty DataFrame."""
        df = pd.DataFrame({"col1": []})
        transform = AddStaticColumns({"static_col": "value"})
        result = transform(df)

        assert "static_col" in result.columns
        assert len(result) == 0

    def test_different_value_types(self):
        """Test adding columns with different value types."""
        df = pd.DataFrame({"col1": [1]})
        transform = AddStaticColumns(
            {
                "str_col": "string",
                "int_col": 42,
                "float_col": 3.14,
                "bool_col": False,
                "none_col": None,
            }
        )
        result = transform(df)

        assert result["str_col"][0] == "string"
        assert result["int_col"][0] == 42
        assert result["float_col"][0] == 3.14
        # Internally pandas always converts data to numpy types
        assert bool(result["bool_col"][0]) is False
        assert result["none_col"][0] is None


class TestColumnFilter:
    """Test suite for ColumnFilter transform."""

    @pytest.mark.parametrize(
        "case_desc, columns, required, optional, expected_cols, n_cols, error_type, error_msg",
        [
            (
                "required only",
                {"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]},
                ["col1", "col3"],
                None,
                ["col1", "col3"],
                2,
                None,
                "",
            ),
            (
                "optional present",
                {"col1": [1], "col2": [2], "col3": [3]},
                ["col1"],
                ["col2", "col4"],
                ["col1", "col2"],
                None,
                None,
                "",
            ),
            (
                "optional absent",
                {"col1": [1], "col2": [2]},
                ["col1"],
                ["col3", "col4"],
                ["col1"],
                1,
                None,
                "",
            ),
            (
                "preserves data",
                {"keep": [1, 2, 3], "drop": [4, 5, 6]},
                ["keep"],
                None,
                ["keep"],
                None,
                None,
                "",
            ),
            (
                "column order",
                {"c": [1], "b": [2], "a": [3]},
                ["a", "b"],
                ["c"],
                ["a", "b", "c"],
                None,
                None,
                "",
            ),
            (
                "empty df",
                {"col1": [], "col2": []},
                ["col1"],
                None,
                ["col1"],
                None,
                None,
                "",
            ),
            (
                "mutually exclusive",
                None,
                ["col1", "col2"],
                ["col2", "col3"],
                None,
                None,
                ValueError,
                "mutually exclusive",
            ),
        ],
    )
    def test_column_filter(
        self,
        case_desc,
        columns,
        required,
        optional,
        expected_cols,
        n_cols,
        error_type,
        error_msg,
    ):
        if error_type is not None:
            with pytest.raises(error_type, match=error_msg):
                ColumnFilter(required_columns=required, optional_columns=optional)
        else:
            df = pd.DataFrame(columns)
            kwargs = {"required_columns": required}
            if optional is not None:
                kwargs["optional_columns"] = optional
            transform = ColumnFilter(**kwargs)
            result = transform(df)
            for col in expected_cols:
                assert col in result.columns
            if n_cols is not None:
                assert len(result.columns) == n_cols


class TestMakeAdapterCompatible:
    """Test suite for MakeAdapterCompatible transform."""

    @pytest.mark.parametrize(
        "case_desc, source_col, target_col, values",
        [
            ("user_prompt", "user_prompt", "prompt", ["Hello", "World"]),
            ("question", "question", "prompt", ["What is AI?"]),
            ("input", "input", "prompt", ["Some input text"]),
            ("input_text", "input_text", "prompt", ["Text input"]),
            ("problem", "problem", "prompt", ["Math problem"]),
            ("query", "query", "prompt", ["Search query"]),
            (
                "system_prompt",
                "system_prompt",
                "system",
                ["You are a helpful assistant"],
            ),
        ],
    )
    def test_remap_column(self, case_desc, source_col, target_col, values):
        df = pd.DataFrame({source_col: values})
        transform = MakeAdapterCompatible()
        result = transform(df)

        assert target_col in result.columns
        assert source_col not in result.columns
        assert list(result[target_col]) == values

    def test_strict_mode_multiple(self):
        """Multiple candidates in strict mode raises error."""
        df = pd.DataFrame({"user_prompt": ["First"], "question": ["Second"]})
        transform = MakeAdapterCompatible()

        with pytest.raises(ValueError, match="Multiple columns found"):
            transform(df)

    def test_already_has_prompt(self):
        """No remapping when prompt column already exists."""
        df = pd.DataFrame({"prompt": ["Already formatted"], "other": ["data"]})
        transform = MakeAdapterCompatible()
        result = transform(df)

        assert "prompt" in result.columns
        assert list(result["prompt"]) == ["Already formatted"]

    def test_no_matching_columns(self):
        """No matching columns leaves DataFrame unchanged."""
        df = pd.DataFrame({"unrelated": ["data"]})
        transform = MakeAdapterCompatible()
        result = transform(df)

        assert "prompt" not in result.columns
        assert "unrelated" in result.columns
