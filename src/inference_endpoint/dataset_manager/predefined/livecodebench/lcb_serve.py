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

"""LiveCodeBench (GitHub: https://github.com/LiveCodeBench/LiveCodeBench,
HuggingFace: https://huggingface.co/datasets/livecodebench/code_generation_lite)
is evaluated by running model-generated code against predefined test cases.

The LiveCodeBench authors note that running arbitrary code is dangerous, and
the existing lcb_runner scripts do not provide proper security and sandboxing.

As such, we provide a simple server that can be run inside containerized environments
such as Docker or enroot to run evaluation as a standalone service, which must be
started up manually before running Inference Endpoints.

This script is standalone, and does not require Inference Endpoints to be installed,
but can be invoked by running it as a module if it is.

It is assumed that:
1. LiveCodeBench is cloned in /opt/LiveCodeBench
2. LiveCodeBench has been installed to Python via pip or other means
3. A mechanism exists to transfer files from the Host to the Container LCB-Serve is
running in.
"""

import argparse
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def chdir(path: Path):
    """Context manager to change the current working directory to the given path."""
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


class _LCBWorker:
    def __init__(
        self,
        lcb_root: Path = Path("/opt/LiveCodeBench"),
        n_lcb_workers: int = 1,
        worker_timeout_sec: int = 60,
    ):
        if not lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {lcb_root} does not exist"
            )
        self.lcb_root = lcb_root
        self.n_lcb_workers = n_lcb_workers
        self.worker_timeout_sec = worker_timeout_sec

    def __call__(self, test_suites, extracted_code):
        # LiveCodeBench assumes that it is run from the root of its repository. As
        # such, we need to chdir() to it before any imports are done
        with chdir(self.lcb_root):
            from lcb_runner.evaluation import extract_instance_results
            from lcb_runner.runner.scenario_router import (
                get_metrics,
                sort_and_extract_save_results,
            )
            from lcb_runner.utils.scenarios import Scenario

            save_results = [
                suite.insert_output(output, output)
                for suite, output in zip(test_suites, extracted_code, strict=False)
            ]

            save_results, combined_results = sort_and_extract_save_results(
                Scenario.codegeneration, save_results
            )

            mock_args = argparse.Namespace(
                timeout=self.worker_timeout_sec,
                num_process_evaluate=self.n_lcb_workers,
            )
            _, instance_results, _ = get_metrics(
                Scenario.codegeneration,
                mock_args,
                sorted(test_suites, key=lambda x: x.question_id),
                combined_results,
            )
            graded = extract_instance_results(instance_results)

        # Currently, Endpoints scoring doesn't care about the reason for failed tests, just the
        # score itself. Also currently lcb_runner is hard-coded and only supports Pass@1 scoring.
        # In the future, if we want to log test failure reasons, it should be added here.
        return graded


class LCBServe:
    def __init__(
        self,
        version_tag: str = "release_v6",
        use_lite: bool = True,
        n_workers: int = 8,
        output_file_store: Path = Path("/mnt/lcb_outputs"),
        lcb_root: Path = Path("/opt/LiveCodeBench"),
    ):
        self.version_tag = version_tag
        self.use_lite = use_lite
        self.n_workers = n_workers
        self.output_file_store = Path(output_file_store)
        self.lcb_root = Path(lcb_root)

        if not self.output_file_store.exists():
            self.output_file_store.mkdir(parents=True)

        if not self.lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {self.lcb_root} does not exist"
            )

        self.test_suites = self.load_test_suites()

    def load_test_suites(self):
        with chdir(self.lcb_root):
            from lcb_runner.runner.scenario_router import build_prompt_benchmark
            from lcb_runner.utils.scenarios import Scenario

            mock_args = argparse.Namespace(
                scenario=Scenario.codegeneration,
                release_version=self.version_tag,
                start_date=None,
                end_date=None,
                not_fast=not self.use_lite,
            )
            test_suites, _ = build_prompt_benchmark(mock_args)
        return {suite.question_id: suite for suite in test_suites}

    def evaluate(
        self,
        question_ids: list[int],
        codes: list[list[str]],
        timeout_sec: int = 60,
        num_extract_fail: int = 0,
    ) -> tuple[float, int]:
        """Evaluates LiveCodeBench problems given question IDs and their corresponding code samples.

        Args:
            question_ids: List of question IDs to evaluate. Each question ID should exist in the loaded test suites.
            codes: List of lists of code strings. Each inner list contains code samples for the corresponding
                question_id. For example, codes[i] contains all code attempts for question_ids[i].
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)
            num_extract_fail: Number of samples that failed code extraction. This is added to the total sample
                count for pass@1 calculation. (Default: 0)

        Returns:
            tuple[float, int]: The pass@1 score and the number of samples that failed to extract code.

        Raises:
            KeyError: If any question_id is not found in the loaded test suites.
        """
        if len(question_ids) != len(codes):
            raise ValueError(
                f"Length mismatch: {len(question_ids)} question_ids but {len(codes)} code lists"
            )

        # Validate all question IDs exist in test suites
        invalid_ids = [qid for qid in question_ids if qid not in self.test_suites]
        if invalid_ids:
            raise KeyError(
                f"Question IDs not found in test suites: {invalid_ids[:10]}"
                + (
                    f" and {len(invalid_ids) - 10} more"
                    if len(invalid_ids) > 10
                    else ""
                )
            )

        # Prepare test suites and codes for evaluation
        test_suites_to_run = [self.test_suites[qid] for qid in question_ids]

        # Calculate total samples: all code attempts across all questions + extraction failures
        total_samples = sum(len(code_list) for code_list in codes) + num_extract_fail

        # In the eval code for GPT-OSS in MLPerf Inference v6.0, a ProcessPoolExecutor is used.
        # For now, we'll delegate the worker distribution to lcb_runner rather than handling it
        # ourselves.
        worker = _LCBWorker(
            lcb_root=self.lcb_root,
            n_lcb_workers=self.n_workers,
            worker_timeout_sec=timeout_sec,
        )
        graded = worker(test_suites_to_run, codes)
        pass_at_1 = sum([sum(results) for results in graded]) / total_samples
        return pass_at_1, num_extract_fail

    def eval_parquet(
        self, parquet_file: Path, timeout_sec: int = 60
    ) -> tuple[float, int]:
        """Evaluates all LiveCodeBench problems in a parquet file.

        Args:
            parquet_file: Path to the parquet file containing the outputs to evaluate.
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)

        Returns:
            tuple[float, int]: The pass@1 score and the number of samples that failed to extract code.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the extracted code column or question ID column is not found in the parquet file.
        """
        file_path = self.output_file_store / parquet_file
        if not file_path.exists():
            raise FileNotFoundError(f"Output file {file_path} does not exist")

        df = pd.read_parquet(file_path)
        if "extracted_code" not in df.columns:
            raise ValueError(f"Extracted code column not found in {file_path}")
        if "question_id" not in df.columns:
            raise ValueError(f"Question ID column not found in {file_path}")

        # Count extraction failures before dropping
        num_extract_fail = int(df["extracted_code"].isnull().sum())
        df = df.dropna().reset_index(drop=True)

        # Group codes by question ID
        test_inputs = defaultdict(list)
        for _, row in df.iterrows():
            test_inputs[row["question_id"]].append(row["extracted_code"])

        # Convert to lists for evaluate method
        question_ids = list(test_inputs.keys())
        codes = [test_inputs[qid] for qid in question_ids]

        # Delegate to evaluate method
        return self.evaluate(
            question_ids=question_ids,
            codes=codes,
            timeout_sec=timeout_sec,
            num_extract_fail=num_extract_fail,
        )


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate LiveCodeBench parquet file and output results as JSON"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the parquet file (relative to output_file_store)",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default="release_v6",
        help="LiveCodeBench version tag (default: release_v6)",
    )
    parser.add_argument(
        "--output-file-store",
        type=Path,
        default=Path("/mnt/lcb_outputs"),
        help="Directory where parquet file is stored (default: /mnt/lcb_outputs)",
    )
    parser.add_argument(
        "--lcb-root",
        type=Path,
        default=Path("/opt/LiveCodeBench"),
        help="Path to LiveCodeBench root directory (default: /opt/LiveCodeBench)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each test case (default: 60)",
    )

    args = parser.parse_args()

    lcb_serve = LCBServe(
        version_tag=args.version_tag,
        output_file_store=args.output_file_store,
        lcb_root=args.lcb_root,
    )

    pass_at_1, num_extract_fail = lcb_serve.eval_parquet(
        args.parquet_file, timeout_sec=args.timeout
    )

    result = {"pass_at_1": pass_at_1, "num_extract_fail": num_extract_fail}
    print(json.dumps(result))
