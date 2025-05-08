#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import json
import unittest
from typing import Tuple
from unittest.mock import patch

import torch
import torch.nn as nn

from test.testing_utils.mock import mock_kia_library, mock_security_library

# 在导入任何可能使用这些函数的模块之前进行mock
def mock_init_weight_quant_normal(weight: torch.Tensor,
                                  bits: int = 8,
                                  is_sym=True,
                                  is_signed: bool = True,
                                  intergral_zp=True,
                                  admm=None,
                                  round_opt=False,
                                  mm_tensor=True,
                                  fake_quant=True,
                                  hqq=False,
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回原始值，不进行量化"""
    return weight, weight, torch.tensor(1.0), torch.tensor(0.0)


def mock_linear_quantization_params(bit: int, x_min: torch.Tensor, x_max: torch.Tensor, q_signed: bool, sym: bool) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(1.0), torch.tensor(0.0)


def mock_fake_quantize(tensor: torch.Tensor,
                       scale: torch.Tensor,
                       zero_point: torch.Tensor,
                       bits: int = 8,
                       is_signed: bool = True,
                       round_opt: bool = False,
                       is_fp=False,
                       dequant=True,
                       group_size=-1,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    return tensor, tensor


# 使用patch装饰器进行全局mock
patch('msmodelslim.quant.kia.utils.init_weight_quant_normal', mock_init_weight_quant_normal).start()
patch('msmodelslim.quant.kia.utils.linear_quantization_params', mock_linear_quantization_params).start()
patch('msmodelslim.quant.kia.utils.fake_quantize', mock_fake_quantize).start()

mock_kia_library()
mock_security_library()

from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig
from msmodelslim.quant.processor.quant.w8a8 import W8A8ProcessorConfig
from msmodelslim.quant.processor.save.saver_processor import SaverProcessorConfig
from msmodelslim.quant.session.session import quant_model, SessionConfig
from msmodelslim.quant.quantizer.linear.fake import W8A8LinearFakeQuantizer


class TestW8A8Quantization(unittest.TestCase):
    """测试W8A8量化功能的单元测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的模型
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm([1]),
            nn.Linear(20, 5)
        )

        # 创建W8A8量化配置
        self.w8a8_config = W8A8QuantConfig(
            w_cfg=dict(
                bits=8,
                symmetric=True,
                per_channel=True
            ),
            a_cfg=dict(
                bits=8,
                symmetric=True,
                per_channel=False
            )
        )

        W8A8QuantConfig.model_validate(self.w8a8_config)

        # 创建会话配置
        self.session_config = SessionConfig(
            processor_cfg_map={
                "w8a8": W8A8ProcessorConfig(
                    cfg_map={
                        "*": self.w8a8_config
                    }
                ),
                "save": SaverProcessorConfig(
                    save_output_path=".",
                    safetensors_name="w8a8_model.safetensors",
                    json_name="w8a8_config.json",
                    save_type="safe_tensor"
                )
            },
            calib_data=[
                torch.ones([10])
            ]
        )

        self.session_config.model_validate(self.session_config)

    def test_w8a8_quantization_basic(self):
        """测试基本的W8A8量化功能"""
        # 记录原始模型的权重
        original_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights.append((name, module.weight.data.clone()))

        # 执行量化
        quant_model(self.model, self.session_config)

        # 检查伪量化成功部署
        for name, _ in original_weights:
            self.assertIsInstance(self.model.get_submodule(name), W8A8LinearFakeQuantizer)

        # 验证模型文件是否被保存
        import os
        self.assertTrue(os.path.exists("./w8a8_model.safetensors"))
        self.assertTrue(os.path.exists("./w8a8_config.json"))

        # 检查json内容
        with open("./w8a8_config.json", "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "model_quant_type": "W8A8",
            "0.weight": "W8A8",
            "0.input_scale": "W8A8",
            "0.input_offset": "W8A8",
            "0.deq_scale": "W8A8",
            "0.quant_bias": "W8A8",
            "1.weight": "FLOAT",
            "1.bias": "FLOAT",
            "2.weight": "W8A8",
            "2.input_scale": "W8A8",
            "2.input_offset": "W8A8",
            "2.deq_scale": "W8A8",
            "2.quant_bias": "W8A8"
        }

        self.assertEqual(config_data.keys(), expected_config_data.keys())

        for key, value in expected_config_data.items():
            self.assertEqual(value, config_data[key])

        os.remove("./w8a8_model.safetensors")
        os.remove("./w8a8_config.json")


if __name__ == '__main__':
    unittest.main()
