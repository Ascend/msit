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

import torch
import torch.nn as nn
from safetensors import safe_open

from msmodelslim.quant.processor.quant.w8a8 import W8A8ProcessorConfig, W8A8LinearFakeQuantizer
from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session.session import quant_model, SessionConfig


class TestW8A8Quantization(unittest.TestCase):
    """测试W8A8量化功能的单元测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的模型
        self.model = nn.Sequential(
            nn.Linear(10, 20, dtype=torch.float32),
            nn.LayerNorm([20]),
            nn.Linear(20, 5, dtype=torch.float32)
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

        for data in self.session_config.calib_data:
            self.model(data)

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

        # 检查伪量化推理
        for data in self.session_config.calib_data:
            self.model(data)

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

        dtype_check = {
            "[02].weight": torch.int8,
            "[02].input_scale": torch.float32,
            "[02].input_offset": torch.float32,
            "[02].deq_scale": torch.float32,
            "[02].quant_bias": torch.int32
        }
        
        shape_check = {
            "0.weight": (20, 10),
            "0.input_scale": (1,),
            "0.input_offset": (1,),
            "0.deq_scale": (20,),
            "0.quant_bias": (20,),
            "2.weight": (5, 20),
            "2.input_scale": (1,),
            "2.input_offset": (1,),
            "2.deq_scale": (5,),
            "2.quant_bias": (5,)
        }

        # 检查safetensor中的tensor数据类型
        with safe_open("./w8a8_model.safetensors", framework="pt") as f:
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
                    

        os.remove("./w8a8_model.safetensors")
        os.remove("./w8a8_config.json")


if __name__ == '__main__':
    unittest.main()
