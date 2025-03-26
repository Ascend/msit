# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import sys
from unittest.mock import MagicMock, patch

import pytest

from ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.model_hookers import ModelRunnerExecuteHook, \
    ModelForwardHook


def test_execute_init():
    fake_model_runner = MagicMock()
    fake_model_runner.execute_model = MagicMock()

    vllm_work_model_runner = MagicMock(ModelRunner=fake_model_runner)

    sys.modules['vllm.worker.model_runner'] = vllm_work_model_runner

    hooker = ModelRunnerExecuteHook()

    with patch.object(hooker, 'do_hook') as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        hook_points, profiler_func_maker = mock_do_hook.call_args[0]
        assert hook_points == [fake_model_runner.execute_model]
        assert callable(profiler_func_maker)


def test_execute_model_maker():
    mock_ori_func = MagicMock(side_effect=["Result", "SecondResult"])

    mock_this = MagicMock()
    mock_model_input = MagicMock()
    mock_model_input.atten_metadata.prefill_metadata = True
    mock_model_input.request_ids_to_seq_ids.item.return_value = [("req1", "seq1")]
    mock_model_input.input_tokens.shape = [5, 0]

    hooker = ModelRunnerExecuteHook()
    with patch.object(hooker, 'do_hook') as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        _, execute_model_maker = mock_do_hook.call_args[0]
        execute_model = execute_model_maker(mock_ori_func)

        # 第一次执行，不进入profiler逻辑
        assert hooker.is_model_first_run is True
        result = execute_model(mock_this, mock_model_input, MagicMock())
        mock_ori_func.assert_called_once()
        assert result == "Result"

        # 第二次执行，进入profiler逻辑
        result = execute_model(mock_this, mock_model_input, MagicMock())
        assert result == "SecondResult"


def test_forward_init():
    fake_common_attention_state = MagicMock()
    fake_common_attention_state.begin_forward = MagicMock()

    vllm_attention_backends_utils = MagicMock(CommonAttentionState=fake_common_attention_state)

    sys.modules['vllm.attention.backends.utils'] = vllm_attention_backends_utils

    hooker = ModelForwardHook()

    with patch.object(hooker, 'do_hook') as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        hook_points, profiler_func_maker = mock_do_hook.call_args[0]
        assert hook_points == [fake_common_attention_state.begin_forward]
        assert callable(profiler_func_maker)


def test_begin_forward_maker():
    mock_ori_func = MagicMock(side_effect=["Result", "SecondResult"])

    mock_this = MagicMock()
    mock_model_input = MagicMock()
    mock_model_input.request_ids_to_seq_ids.item.return_value = [("req1", "seq1")]

    hooker = ModelForwardHook()
    with patch.object(hooker, 'do_hook') as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        _, begin_forward_maker = mock_do_hook.call_args[0]
        begin_forward = begin_forward_maker(mock_ori_func)

        # 第一次执行，不进入profiler逻辑
        assert hooker.is_forward_first_run is True
        result = begin_forward(mock_this, mock_model_input)
        mock_ori_func.assert_called_once()
        assert result == "Result"

        # 第二次执行，进入profiler逻辑
        result = begin_forward(mock_this, mock_model_input)
        assert result == "SecondResult"
