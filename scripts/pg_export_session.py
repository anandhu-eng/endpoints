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

"""Standalone utility to export a session's data from Postgres to local files.

Reads from the `events_{session_id}` table in Postgres and writes:
  - events.jsonl        — one JSON object per event row
  - result_summary.json — computed metrics summary

Usage:

    postgresql://neondb_owner:npg_GywCD8TukWI9@ep-withered-grass-akhya6bx-pooler.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require

    python scripts/pg_export_session.py \\
        --session-id <session_id> \\
        --connection-string "postgresql://user:pass@host:5432/dbname" \\
        --output-dir ./output

    # Or use DATABASE_URL env var:
    DATABASE_URL="postgresql://..." python scripts/pg_export_session.py \\
        --session-id <session_id>
"""

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Break the circular import:
#   reporter.py → load_generator.events (triggers __init__.py)
#   __init__.py → session.py → reporter.py  (circular!)
#
# Fix: pre-register a minimal load_generator package stub + load events.py
# directly so __init__.py is never executed.
# ---------------------------------------------------------------------------
_src = Path(__file__).resolve().parents[1] / "src"

_lg_pkg = types.ModuleType("inference_endpoint.load_generator")
_lg_pkg.__path__ = [str(_src / "inference_endpoint/load_generator")]
_lg_pkg.__package__ = "inference_endpoint.load_generator"
sys.modules.setdefault("inference_endpoint.load_generator", _lg_pkg)

_events_spec = importlib.util.spec_from_file_location(
    "inference_endpoint.load_generator.events",
    _src / "inference_endpoint/load_generator/events.py",
)
_events_mod = importlib.util.module_from_spec(_events_spec)
_events_mod.__package__ = "inference_endpoint.load_generator"
sys.modules.setdefault("inference_endpoint.load_generator.events", _events_mod)
_events_spec.loader.exec_module(_events_mod)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_src))
from inference_endpoint.metrics.reporter import MetricsReporter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a Postgres session to events.jsonl and result_summary.json"
        )
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID — determines table name (events_{session_id})",
    )
    parser.add_argument(
        "--connection-string",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres connection string (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output files into (default: .)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name/path for tokenizer (optional; enables output_sequence_lengths)",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=getattr(logging, args.log_level))

    tokenizer = None
    if args.model:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            logging.warning(f"Failed to load tokenizer for {args.model}: {e}")

    if not args.connection_string:
        parser.error(
            "Provide --connection-string or set DATABASE_URL environment variable"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_name = f"events_{args.session_id}"
    print(f"Connecting to Postgres, table: {table_name}")

    reporter = MetricsReporter(
        connection_name=args.connection_string,
        client_type="postgres",
        table_name=table_name,
    )

    with reporter:
        events_path = output_dir / "events.jsonl"
        print(f"Writing {events_path} ...")
        reporter.dump_to_json(events_path)
        print(f"  wrote {events_path}")

        summary_path = output_dir / "result_summary.json"
        print(f"Writing {summary_path} ...")
        report = reporter.create_report(tokenizer)
        report.to_json(save_to=summary_path)
        print(f"  wrote {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
