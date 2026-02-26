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

"""Unit tests for AsyncEventRecorder — context manager, start/stop, init defaults."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.metrics.async_recorder import AsyncEventRecorder


class TestInit:
    def test_defaults(self):
        r = AsyncEventRecorder("sess1", "ipc:///tmp/test")
        assert r.session_id == "sess1"
        assert r.publisher_address == "ipc:///tmp/test"
        assert r.txn_buffer_size == 1000
        assert r.sub_settle_s == 0.3
        assert r.start_timeout == 10.0
        assert r.stop_timeout == 10.0
        assert r.db_path == Path("/dev/shm/mlperf_testsession_sess1.db")
        assert r._process is None

    def test_custom_params(self):
        r = AsyncEventRecorder(
            "s2",
            "ipc:///tmp/x",
            txn_buffer_size=500,
            sub_settle_s=1.0,
            start_timeout=5.0,
            stop_timeout=3.0,
        )
        assert r.txn_buffer_size == 500
        assert r.sub_settle_s == 1.0
        assert r.start_timeout == 5.0
        assert r.stop_timeout == 3.0


class TestIsAlive:
    def test_not_alive_before_start(self):
        r = AsyncEventRecorder("s", "ipc:///tmp/x")
        assert not r.is_alive

    def test_alive_reflects_process(self):
        r = AsyncEventRecorder("s", "ipc:///tmp/x")
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        r._process = mock_proc
        assert r.is_alive

        mock_proc.is_alive.return_value = False
        assert not r.is_alive


class TestStartStop:
    @patch("inference_endpoint.metrics.async_recorder._subscriber_main")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Process")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Event")
    def test_start_uses_instance_defaults(
        self, mock_event_cls, mock_proc_cls, _mock_main
    ):
        mock_ready = MagicMock()
        mock_ready.wait.return_value = True
        mock_event_cls.return_value = mock_ready

        mock_proc = MagicMock()
        mock_proc_cls.return_value = mock_proc

        r = AsyncEventRecorder("s", "ipc:///x", sub_settle_s=0.7, start_timeout=5.0)
        r.start()

        mock_ready.wait.assert_called_once_with(timeout=5.0)
        mock_proc.start.assert_called_once()

    @patch("inference_endpoint.metrics.async_recorder._subscriber_main")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Process")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Event")
    def test_start_explicit_overrides(self, mock_event_cls, mock_proc_cls, _mock_main):
        mock_ready = MagicMock()
        mock_ready.wait.return_value = True
        mock_event_cls.return_value = mock_ready

        mock_proc = MagicMock()
        mock_proc_cls.return_value = mock_proc

        r = AsyncEventRecorder("s", "ipc:///x", start_timeout=5.0)
        r.start(timeout=2.0)

        mock_ready.wait.assert_called_once_with(timeout=2.0)

    @patch("inference_endpoint.metrics.async_recorder._subscriber_main")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Process")
    @patch("inference_endpoint.metrics.async_recorder.multiprocessing.Event")
    def test_start_timeout_kills_process(
        self, mock_event_cls, mock_proc_cls, _mock_main
    ):
        mock_ready = MagicMock()
        mock_ready.wait.return_value = False  # Simulate timeout
        mock_event_cls.return_value = mock_ready

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc_cls.return_value = mock_proc

        r = AsyncEventRecorder("s", "ipc:///x", start_timeout=0.1)
        with pytest.raises(TimeoutError, match="did not become ready"):
            r.start()

        mock_proc.kill.assert_called_once()

    def test_stop_noop_when_no_process(self):
        r = AsyncEventRecorder("s", "ipc:///x")
        r.stop()  # Should not raise

    def test_stop_kills_if_still_alive_after_join(self):
        r = AsyncEventRecorder("s", "ipc:///x", stop_timeout=1.0)
        mock_proc = MagicMock()
        # Still alive after first join → triggers kill
        mock_proc.is_alive.side_effect = [True, True, False]
        r._process = mock_proc

        r.stop()

        mock_proc.join.assert_any_call(timeout=1.0)
        mock_proc.kill.assert_called_once()

    def test_stop_uses_instance_default(self):
        r = AsyncEventRecorder("s", "ipc:///x", stop_timeout=7.0)
        mock_proc = MagicMock()
        mock_proc.is_alive.side_effect = [True, False]  # alive before join, dead after
        r._process = mock_proc

        r.stop()

        mock_proc.join.assert_called_once_with(timeout=7.0)


class TestContextManager:
    @patch.object(AsyncEventRecorder, "stop")
    @patch.object(AsyncEventRecorder, "start")
    def test_enter_calls_start(self, mock_start, mock_stop):
        r = AsyncEventRecorder("s", "ipc:///x")
        result = r.__enter__()
        assert result is r
        mock_start.assert_called_once()

    @patch.object(AsyncEventRecorder, "stop")
    @patch.object(AsyncEventRecorder, "start")
    def test_exit_normal_uses_full_timeout(self, mock_start, mock_stop):
        r = AsyncEventRecorder("s", "ipc:///x", stop_timeout=15.0)
        r.__exit__(None, None, None)
        mock_stop.assert_called_once_with(timeout=15.0)

    @patch.object(AsyncEventRecorder, "stop")
    @patch.object(AsyncEventRecorder, "start")
    def test_exit_on_error_uses_short_timeout(self, mock_start, mock_stop):
        r = AsyncEventRecorder("s", "ipc:///x", stop_timeout=15.0)
        r.__exit__(ValueError, ValueError("boom"), None)
        mock_stop.assert_called_once_with(timeout=2.0)

    @patch.object(AsyncEventRecorder, "stop")
    @patch.object(AsyncEventRecorder, "start")
    def test_with_statement(self, mock_start, mock_stop):
        with AsyncEventRecorder("s", "ipc:///x") as recorder:
            assert isinstance(recorder, AsyncEventRecorder)
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
