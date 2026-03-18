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

import pytest
from inference_endpoint.evaluation.extractor import Extractor, PythonCodeExtractor


class TestPythonCodeExtractor:
    """Test cases for PythonCodeExtractor."""

    @pytest.mark.parametrize(
        "case_desc, text, expected",
        [
            (
                "python block",
                "Here's the solution:\n```python\ndef foo():\n    pass\n```",
                "def foo():\n    pass",
            ),
            (
                "plain block",
                "Solution:\n```\nprint('hello')\n```",
                "print('hello')",
            ),
            (
                "plain with language tag",
                "```\npython\nprint('test')\n```",
                "print('test')",
            ),
            (
                "py tag",
                "```\npy\nprint('test')\n```",
                "print('test')",
            ),
            (
                "last python block wins",
                "First:\n```python\ndef wrong():\n    pass\n```\n\n"
                "Better:\n```python\ndef correct():\n    return True\n```",
                "def correct():\n    return True",
            ),
            (
                "multiline code",
                "```python\nclass Solution:\n    def solve(self, n: int) -> int:\n"
                "        result = 0\n        for i in range(n):\n"
                "            result += i\n        return result\n```",
                "class Solution:\n    def solve(self, n: int) -> int:\n"
                "        result = 0\n        for i in range(n):\n"
                "            result += i\n        return result",
            ),
            (
                "with markdown formatting",
                "**Solution:**\n\nHere's the code:\n\n```python\ndef foo():\n"
                "    return 42\n```\n\n*This works!*",
                "def foo():\n    return 42",
            ),
            (
                "python over plain priority",
                "Plain:\n```\ndef plain():\n    pass\n```\n\n"
                "Python:\n```python\ndef python():\n    return True\n```",
                "def python():\n    return True",
            ),
            (
                "whitespace handling",
                "  \n\n  ```python\n  def test():\n      pass\n  ```  \n\n  ",
                "def test():\n      pass",
            ),
        ],
    )
    def test_extract(self, case_desc, text, expected):
        assert PythonCodeExtractor.extract(text) == expected

    @pytest.mark.parametrize(
        "case_desc, text",
        [
            ("empty string", ""),
            ("no code block", "This is just plain text without any code blocks."),
            ("null input", None),
            ("non-string input", 123),
            ("inline code only", "Use `print()` to output. No code block here."),
        ],
    )
    def test_extract_returns_none(self, case_desc, text):
        assert PythonCodeExtractor.extract(text) is None

    def test_registered_in_extractor_registry(self):
        """Test that PythonCodeExtractor is registered."""
        assert "python_code_extractor" in Extractor.PREDEFINED
        assert Extractor.get("python_code_extractor") == PythonCodeExtractor

    def test_extractor_get_method(self):
        """Test that we can retrieve PythonCodeExtractor by name."""
        extractor_cls = Extractor.get("python_code_extractor")
        text = "```python\nprint('test')\n```"
        result = extractor_cls.extract(text)
        assert result == "print('test')"
