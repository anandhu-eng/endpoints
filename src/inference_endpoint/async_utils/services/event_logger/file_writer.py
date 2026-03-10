# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File-based writers for event records (JSONL, etc.)."""

from pathlib import Path

import msgspec
from inference_endpoint.core.record import EventRecord, EventType

from .writer import RecordWriter


class FileWriter(RecordWriter):
    """Writer for writing event records to a file."""

    def __init__(
        self,
        file_path: Path,
        mode: str = "w",
        flush_interval: int | None = None,
        **kwargs: object,
    ):
        super().__init__(flush_interval=flush_interval)
        self.file_path = Path(file_path)
        # No idea what the 'IO' type MyPy thinks this is, apparently even io.IOBase does not work, so just ignore.
        self.file_obj = self.file_path.open(mode=mode)  # type: ignore[assignment]

    def close(self) -> None:
        if self.file_obj is not None:
            try:
                self.flush()
                self.file_obj.close()
            except (OSError, FileNotFoundError):
                # File may already be closed or I/O error on close (e.g. disk full).
                pass
            finally:
                self.file_obj = None  # type: ignore[assignment]

    def record_to_line(self, record: EventRecord) -> str:
        """Convert an event record to a line of text."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _write_record(self, record: EventRecord) -> None:
        if self.file_obj is not None:
            self.file_obj.write(self.record_to_line(record) + "\n")

    def flush(self) -> None:
        if self.file_obj is not None:
            self.file_obj.flush()
        super().flush()


class JSONLWriter(FileWriter):
    """Writes to a JSONL file."""

    extension = ".jsonl"

    def __init__(self, file_path: Path, *args, **kwargs):
        super().__init__(file_path.with_suffix(self.extension), *args, **kwargs)

        # EventRecords are msgspec structs so we can use the built-in JSON encoder
        self.encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)

    def record_to_line(self, record: EventRecord) -> str:
        return self.encoder.encode(record).decode("utf-8")
