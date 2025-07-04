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

import torch
import torch.nn as nn
from safetensors import safe_open

from msmodelslim.quant.processor.quant.w4a8_dynamic import W4A8DynamicProcessorConfig, W4A8DynamicLinearFakeQuantizer
from msmodelslim.quant.processor.quant.w4a8_dynamic import W4A8DynamicQuantConfig
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session.session import quant_model, SessionConfig


class TestW4A8DynamicQuantization(unittest.TestCase):
    """测试W4A8动态量化功能的单元测试类"""

    def setUp(self):
        """测试前的准备工作"""

        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.realpath(self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

        # 创建一个简单的模型
        self.model = nn.Sequential(
            nn.Linear(1024, 2048, dtype=torch.float32),
            nn.LayerNorm([2048]),
            nn.Linear(2048, 5, dtype=torch.float32, bias=True)
        )

        # 创建W4A8动态量化配置
        self.w4a8_dynamic_config = W4A8DynamicQuantConfig()

        W4A8DynamicQuantConfig.model_validate(self.w4a8_dynamic_config)

        # 创建会话配置
        self.session_config = SessionConfig(
            processor_cfg_map={
                "w4a8_dynamic": W4A8DynamicProcessorConfig(
                    cfg_map={
                        "*": self.w4a8_dynamic_config
                    }
                ),
                "save": SaverProcessorConfig(
                    save_output_path=self.temp_dir,
                    safetensors_name="w4a8_dynamic_model.safetensors",
                    json_name="w4a8_dynamic_config.json",
                    save_type="safe_tensor",
                    quant_type="w4a8_dynamic"
                )
            },
            calib_data=[
                torch.ones([1024])
            ]
        )

        self.session_config.model_validate(self.session_config)

        for data in self.session_config.calib_data:
            self.model(data)

    def tearDown(self):
        pass
        # shutil.rmtree(self.temp_dir)

    def test_w4a8_dynamic_quantization_basic(self):
        """测试基本的W4A8动态量化功能"""
        # 记录原始模型的权重
        original_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights.append((name, module.weight.data.clone()))

        # 执行量化
        quant_model(self.model, self.session_config)

        # 检查伪量化成功部署
        for name, _ in original_weights:
            self.assertIsInstance(self.model.get_submodule(name), W4A8DynamicLinearFakeQuantizer)

        # 检查伪量化推理
        with self.assertRaises(NotImplementedError):
            for data in self.session_config.calib_data:
                self.model(data)

        # 验证模型文件是否被保存
        import os
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "w4a8_dynamic_model-00001-of-00001.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "w4a8_dynamic_config.json")))

        # 检查json内容
        with open(os.path.join(self.temp_dir, "w4a8_dynamic_config.json"), "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "model_quant_type": "W4A8_DYNAMIC",
            "0.weight_scale": "W4A8_DYNAMIC",
            "0.weight_offset": "W4A8_DYNAMIC",
            "0.weight_scale_second": "W4A8_DYNAMIC",
            "0.weight_offset_second": "W4A8_DYNAMIC",
            "0.weight": "W4A8_DYNAMIC",
            "0.bias": "W4A8_DYNAMIC",
            "1.weight": "FLOAT",
            "1.bias": "FLOAT",
            "2.weight_scale": "W4A8_DYNAMIC",
            "2.weight_offset": "W4A8_DYNAMIC",
            "2.weight_scale_second": "W4A8_DYNAMIC",
            "2.weight_offset_second": "W4A8_DYNAMIC",
            "2.weight": "W4A8_DYNAMIC",
            "2.bias": "W4A8_DYNAMIC"
        }

        self.assertEqual(config_data.keys(), expected_config_data.keys())

        for key, value in expected_config_data.items():
            self.assertEqual(value, config_data[key])

        dtype_check = {
            "[02].weight": torch.int8,
            "[02].weight_scale": torch.float32,
            "[02].weight_offset": torch.float32,
            "[02].weight_scale_second": torch.float32,
            "[02].weight_offset_second": torch.float32,
        }

        shape_check = {
            "0.weight": (2048, 1024),
            "0.weight_scale": (2048, 1),
            "0.weight_offset": (2048, 1),
            "0.weight_scale_second": (8192, 1),
            "0.weight_offset_second": (8192, 1),
            "1.bias": (2048,),
            "2.weight": (5, 2048),
            "2.weight_scale": (5, 1),
            "2.weight_offset": (5, 1),
            "2.weight_scale_second": (40, 1),
            "2.weight_offset_second": (40, 1),
            "2.bias": (5,)
        }

        # 检查safetensor中的tensor数据类型
        with safe_open(os.path.join(self.temp_dir, "w4a8_dynamic_model-00001-of-00001.safetensors"),
                       framework="pt") as f:
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


if __name__ == '__main__':
    unittest.main()
