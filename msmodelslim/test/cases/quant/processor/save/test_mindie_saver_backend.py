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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from testing_utils.mock import mock_kia_library, mock_security_library

mock_kia_library()
mock_security_library()

from msmodelslim.quant.processor.quant.w8a8 import W8A8LinearFakeQuantizer, W8A8QuantConfig
from msmodelslim.quant.processor.save.backend.mindie import MindIESaverBackend
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.quant_linear = W8A8LinearFakeQuantizer(cfg=W8A8QuantConfig(),
                                                    input_scale=torch.tensor([1.0]),
                                                    input_offset=torch.tensor([0.0]),
                                                    deq_scale=torch.tensor([1.0]),
                                                    quant_bias=torch.tensor([0.0]),
                                                    weight=torch.tensor([1.0]))


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def run_distributed_test(rank, world_size, temp_dir):
    setup_distributed(rank, world_size)

    # 创建模型
    model = SimpleModel()

    # 配置保存参数
    save_cfg = SaverProcessorConfig(
        save_output_path=temp_dir,
        safetensors_name="model.safetensors",
        json_name="model.json",
        save_type="safe_tensor",
        part_file_size=1024
    )

    # 创建saver backend
    saver = MindIESaverBackend(model, save_cfg)

    # 执行保存
    saver.pre_process()
    saver.save("", model)
    saver.post_process()

    cleanup_distributed()


class TestMindIESaverBackend(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.realpath(self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def tearDown(self):
        pass
        # shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(sys.platform == "win32", reason="分布式测试在Windows上不支持")
    def test_distributed_save(self):
        world_size = 2
        mp.spawn(
            run_distributed_test,
            args=(world_size, self.temp_dir),
            nprocs=world_size,
            join=True
        )

        # 验证文件是否被正确保存
        safetensors_path_1 = Path(self.temp_dir) / "model-00001-of-00002.safetensors"
        safetensors_path_2 = Path(self.temp_dir) / "model-00002-of-00002.safetensors"
        index_json_path = Path(self.temp_dir) / "model.safetensors.index.json"
        json_path = Path(self.temp_dir) / "model.json"

        self.assertTrue(safetensors_path_1.exists())
        self.assertTrue(safetensors_path_2.exists())
        self.assertTrue(index_json_path.exists())
        self.assertTrue(json_path.exists())

        with open(json_path, "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "model_quant_type": "W8A8",
            "linear1.weight": "FLOAT",
            "linear1.bias": "FLOAT",
            "linear2.weight": "FLOAT",
            "linear2.bias": "FLOAT",
            "quant_linear.weight": "W8A8",
            "quant_linear.input_scale": "W8A8",
            "quant_linear.input_offset": "W8A8",
            "quant_linear.deq_scale": "W8A8",
            "quant_linear.quant_bias": "W8A8"
        }

        self.assertEqual(config_data.keys(), expected_config_data.keys())

    def test_single_process_save(self):
        # 创建模型
        model = SimpleModel()

        # 配置保存参数
        save_cfg = SaverProcessorConfig(
            save_output_path=self.temp_dir,
            safetensors_name="model.safetensors",
            json_name="model.json",
            save_type="safe_tensor",
            part_file_size=4
        )

        # 创建saver backend
        saver = MindIESaverBackend(model, save_cfg)

        # 执行保存
        saver.pre_process()
        saver.save("", model)
        saver.post_process()

        # 验证文件是否被正确保存
        safetensors_path = Path(self.temp_dir) / "model-00001-of-00001.safetensors"
        index_json_path = Path(self.temp_dir) / "model.safetensors.index.json"
        json_path = Path(self.temp_dir) / "model.json"

        self.assertTrue(safetensors_path.exists())
        self.assertTrue(index_json_path.exists())
        self.assertTrue(json_path.exists())

        with open(json_path, "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "model_quant_type": "W8A8",
            "linear1.weight": "FLOAT",
            "linear1.bias": "FLOAT",
            "linear2.weight": "FLOAT",
            "linear2.bias": "FLOAT",
            "quant_linear.weight": "W8A8",
            "quant_linear.input_scale": "W8A8",
            "quant_linear.input_offset": "W8A8",
            "quant_linear.deq_scale": "W8A8",
            "quant_linear.quant_bias": "W8A8"
        }

        self.assertEqual(config_data.keys(), expected_config_data.keys())


if __name__ == '__main__':
    unittest.main()
