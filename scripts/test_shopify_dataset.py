#!/usr/bin/env python3
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

"""Sample script to load and test the Shopify dataset with q3vl preset transforms.

Run from project root:
    python scripts/test_shopify_dataset.py [--datasets-dir PATH] [--max-samples N]

Requires HuggingFace access. For gated datasets, run: huggingface-cli login
"""

import argparse
from pathlib import Path

from inference_endpoint.dataset_manager.predefined.shopify import Shopify
from inference_endpoint.dataset_manager.predefined.shopify import presets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and test Shopify dataset with q3vl preset transforms"
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Root directory for datasets (default: datasets)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Max samples to load (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train"],
        help="Splits to load (default: train)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate even if parquet exists",
    )
    args = parser.parse_args()

    print("Loading Shopify dataset with q3vl preset...")
    dataset = Shopify.get_dataloader(
        datasets_dir=args.datasets_dir,
        transforms=presets.q3vl(),
        max_samples=args.max_samples,
        split=args.split,
        force=args.force,
    )
    dataset.load()

    n = dataset.num_samples()
    print(f"Loaded {n} samples\n")

    for i in range(min(n, 2)):
        sample = dataset.load_sample(i)
        print(f"--- Sample {i} ---")
        print("Keys:", list(sample.keys()))
        print("System prompt (first 200 chars):", sample["system"][:200] + "...")
        print("Prompt content parts:", [p.get("type") for p in sample["prompt"]])
        for j, part in enumerate(sample["prompt"]):
            if part.get("type") == "text":
                print(f"  Text part {j} (first 100 chars):", part["text"][:100] + "...")
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                print(f"  Image part {j}: {url[:50]}... (len={len(url)})")
        print()

    # Verify schema in system prompt
    sample0 = dataset.load_sample(0)
    assert "ProductMetadata" in sample0["system"]
    assert "category" in sample0["system"] and "brand" in sample0["system"]
    assert "is_secondhand" in sample0["system"]
    assert "Json format for the expected responses from the VLM" in sample0["system"]
    print("All checks passed: schema, category, brand, is_secondhand present in system prompt")


if __name__ == "__main__":
    main()
