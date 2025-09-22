#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from functools import lru_cache
from pathlib import Path
from typing import List
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from msmodelslim.app import DeviceType
from msmodelslim.model import ModelFactory
from msmodelslim.model.qwen3 import Qwen3ModelAdapter
from msmodelslim.quant.processor.kv_smooth import KVSmoothFusedUnit, KVSmoothFusedType
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer


@lru_cache(maxsize=1)
def is_npu_available():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_cuda_available():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


@ModelFactory.register("fake_llama")
class FakeLlamaModelAdapter(Qwen3ModelAdapter):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        model, tokenizer = get_fake_llama_model_and_tokenizer()
        self.loaded_config = model.config
        self.loaded_model = model
        self.loaded_tokenizer = tokenizer
        Qwen3ModelAdapter.__init__(self, model_type, model_path, trust_remote_code)

    def _load_config(self, trust_remote_code=False) -> PretrainedConfig:
        return self.loaded_config

    def _load_model(self, device: DeviceType) -> nn.Module:
        return self.loaded_model

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return self.loaded_tokenizer

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_proj",
                fused_from_key_states_name="k_proj",
                fused_type=KVSmoothFusedType.StateViaRopeToLinear
            )
            for i in range(self.config.num_hidden_layers)
        ]


def invoke_test(config_name: str, model_save_path: str):
    """使用真正的CLI parser来模拟命令行参数并返回model_adapter"""
    import sys
    from msmodelslim.cli.__main__ import main as cli_main

    # 保存原始的sys.argv
    original_argv = sys.argv.copy()

    # 用于存储model_adapter的变量
    captured_model_adapter = None

    try:
        # 构建命令行参数
        config_path = os.path.join(os.path.dirname(__file__), "configs", config_name)
        sys.argv = [
            'msmodelslim',
            'quant',
            '--model_type', 'fake_llama',
            '--model_path', './',
            '--save_path', model_save_path,
            '--device', 'cpu',
            '--config_path', config_path,
            '--trust_remote_code', 'False'
        ]

        # 使用patch来模拟copy_files调用并拦截model_adapter
        with patch('msmodelslim.app.quant_service.modelslim_v1.save.ascendv1.copy_files') as mock_copy_files:
            # 获取原始的quantize方法
            from msmodelslim.app.quant_service.proxy import QuantServiceProxy
            original_quantize = QuantServiceProxy.quantize

            # 创建包装函数来捕获model_adapter但不影响原始流程
            def capture_model_adapter(self, quant_config, model_adapter, save_path=None, device=None):
                nonlocal captured_model_adapter
                captured_model_adapter = model_adapter
                # 调用原始方法，保持原始流程不变
                return original_quantize(self, quant_config, model_adapter, save_path, device)

            # 临时替换方法
            QuantServiceProxy.quantize = capture_model_adapter

            try:
                # 直接调用CLI main函数，它会解析sys.argv
                cli_main()
            finally:
                # 恢复原始方法
                QuantServiceProxy.quantize = original_quantize

    finally:
        # 恢复原始的sys.argv
        sys.argv = original_argv

    return captured_model_adapter
