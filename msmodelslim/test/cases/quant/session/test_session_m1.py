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
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from msmodelslim.quant.processor.anti_outlier.m1 import M1ProcessorConfig
from msmodelslim.quant.processor.anti_outlier.m1 import smooth_ln_fcs
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session.session import quant_model, SessionConfig


class SimpleModel(nn.Module):
    """简单的测试模型，包含LayerNorm和Linear层"""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class TestM1Processor(unittest.TestCase):
    """测试M1处理器通过quant_model函数的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.batch_size = 4
        self.hidden_size = 64
        self.model = SimpleModel(self.hidden_size)
        self.calib_data = [torch.randn(self.batch_size, self.hidden_size) for _ in range(2)]
        self.temp_dir = tempfile.mkdtemp()
        self.save_cfg = SaverProcessorConfig(
            save_output_path=self.temp_dir,
            safetensors_name="m1_model.safetensors",
            json_name="m1_config.json",
            save_type="safe_tensor"
        )

    @patch('msmodelslim.quant.processor.anti_outlier.m1.smooth_ln_fcs', wraps=smooth_ln_fcs)
    def test_quant_model_with_m1(self, mock_smooth_ln_fcs):
        """测试使用quant_model函数应用M1处理器"""
        # 创建会话配置
        session_cfg = SessionConfig(
            processor_cfg_map={
                "m1": M1ProcessorConfig(),
                "save": self.save_cfg,
            },
            calib_data=self.calib_data
        )

        # 应用量化处理
        quant_model(self.model, session_cfg)

        # 验证模型是否被正确处理
        # 这里我们主要验证模型结构是否保持不变，以及是否可以正常前向传播
        test_input = torch.randn(self.batch_size, self.hidden_size)
        output = self.model(test_input)

        # 验证输出形状是否正确
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

        # 验证smooth_ln_fcs是否被调用
        self.assertTrue(hasattr(mock_smooth_ln_fcs, 'call_args'))
        smooth_args = mock_smooth_ln_fcs.call_args[0]
        self.assertEqual(smooth_args[0].__class__.__name__, "NormBias")
        self.assertEqual(smooth_args[1], [self.model.fc1])

        with open(os.path.join(self.temp_dir, "m1_config.json"), "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "model_quant_type": "W8A8",
            "ln.weight": "FLOAT",
            "ln.bias": "FLOAT",
            "fc1.weight": "FLOAT",
            "fc1.bias": "FLOAT",
            "fc2.weight": "FLOAT",
            "fc2.bias": "FLOAT",
            "ln2.weight": "FLOAT",
            "ln2.bias": "FLOAT"
        }

        self.assertEqual(config_data.keys(), expected_config_data.keys())

        for key, value in expected_config_data.items():
            self.assertEqual(value, config_data[key])

        dtype_check = {
            "ln.weight": torch.float32,
            "ln.bias": torch.float32,
            "fc1.weight": torch.float32,
            "fc1.bias": torch.float32,
            "fc2.weight": torch.float32,
            "fc2.bias": torch.float32,
            "ln2.weight": torch.float32,
            "ln2.bias": torch.float32
        }

        shape_check = {
            "ln.weight": (64,),
            "ln.bias": (64,),
            "fc1.weight": (64, 64),
            "fc1.bias": (64,),
            "fc2.weight": (64, 64),
            "fc2.bias": (64,),
            "ln2.weight": (64,),
            "ln2.bias": (64,)
        }

        # 检查safetensor中的tensor数据类型
        with safe_open(os.path.join(self.temp_dir, "m1_model-00001-of-00001.safetensors"), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # 根据key的模式匹配对应的dtype
                matched_dtype = None
                for pattern, expected_dtype in dtype_check.items():
                    if key.endswith(pattern.replace("*", "")):
                        matched_dtype = expected_dtype
                        break

                if matched_dtype is not None:
                    self.assertEqual(tensor.dtype, matched_dtype,
                                     f"Tensor {key} has incorrect dtype. Expected {matched_dtype}, got {tensor.dtype}")

                # 根据key的模式匹配对应的shape
                matched_shape = None
                for pattern, expected_shape in shape_check.items():
                    if key.endswith(pattern.replace("*", "")):
                        matched_shape = expected_shape
                        break

                if matched_shape is not None:
                    self.assertEqual(tensor.shape, matched_shape,
                                     f"Tensor {key} has incorrect shape. Expected {matched_shape}, got {tensor.shape}")


if __name__ == "__main__":
    unittest.main()
