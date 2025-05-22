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

import pytest
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.quant.w8a8 import (
    W8A8LinearFakeQuantizer,
    W8A8LinearQuantizer,
    W8A8Processor,
    W8A8ProcessorConfig,
    W8A8QuantConfig
)
from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig
from msmodelslim.quant.quantizer.base.const import WeightQuantMethod, ActivationQuantMethod
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig, WeightQuantConfig


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


class TestW8A8LinearFakeQuantizer:
    @pytest.fixture
    def fake_quantizer(self):
        cfg = LinearQuantConfig(
            w_cfg=WeightQuantConfig(bits=8, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )
        input_scale = torch.tensor([1.0])
        input_offset = torch.tensor([0.0])
        deq_scale = torch.tensor([1.0])
        quant_bias = torch.tensor([0], dtype=torch.int32)
        weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)

        return W8A8LinearFakeQuantizer(
            cfg=cfg,
            input_scale=input_scale,
            input_offset=input_offset,
            deq_scale=deq_scale,
            quant_bias=quant_bias,
            weight=weight
        )

    def test_forward(self, fake_quantizer):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        output = fake_quantizer(x)
        assert output.dtype == torch.float32
        assert output.shape == (2, 2)


class TestW8A8LinearQuantizer:
    @pytest.fixture
    def linear_module(self):
        return nn.Linear(2, 2)

    @pytest.fixture
    def quant_config(self):
        return LinearQuantConfig(
            w_cfg=WeightQuantConfig(bits=8, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )

    def test_match(self, linear_module, quant_config):
        # 测试匹配条件
        assert W8A8LinearQuantizer.match(linear_module, quant_config)

        # 测试不匹配的情况
        wrong_config = LinearQuantConfig(
            w_cfg=WeightQuantConfig(bits=4, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )
        assert not W8A8LinearQuantizer.match(linear_module, wrong_config)

    def test_deploy(self, linear_module, quant_config):
        quantizer = W8A8LinearQuantizer(linear_module, quant_config)
        quantizer(torch.ones([2]))
        fake_quantizer = quantizer.deploy()
        assert isinstance(fake_quantizer, W8A8LinearFakeQuantizer)
        assert fake_quantizer.weight.shape == linear_module.weight.shape


class TestW8A8Processor:
    @pytest.fixture
    def model(self):
        return nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 2)
        )

    @pytest.fixture
    def processor_config(self):
        quant_config = W8A8QuantConfig(
            w_cfg=WeightQuantConfig(bits=8, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )
        return W8A8ProcessorConfig(
            cfg_map={"0": quant_config, "1": quant_config}
        )

    def test_processor_initialization(self, model, processor_config):
        processor = W8A8Processor(model, processor_config)
        assert not processor.is_data_free()
        assert len(processor.cfg_manager) == 2

    def test_processor_quant_type(self, model, processor_config):
        processor = W8A8Processor(model, processor_config)
        quant_config = processor.cfg_manager.get("0")
        assert quant_config.quant_type == QuantType.W8A8


class TestW8A8QuantConfig:
    def test_quant_type(self):
        config = W8A8QuantConfig(
            w_cfg=WeightQuantConfig(bits=8, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )
        assert config.quant_type == QuantType.W8A8


class TestW8A8ProcessorConfig:
    @pytest.fixture
    def quant_config(self):
        return W8A8QuantConfig(
            w_cfg=WeightQuantConfig(bits=8, method=WeightQuantMethod.MINMAX),
            a_cfg=ActivationQuantConfig(bits=8, method=ActivationQuantMethod.MINMAX)
        )

    def test_processor_config_initialization(self, quant_config):
        # 测试正常初始化
        config = W8A8ProcessorConfig(cfg_map={"0": quant_config})
        assert "0" in config.cfg_map
        assert isinstance(config.cfg_map["0"], W8A8QuantConfig)

    def test_processor_config_validation(self, quant_config):
        # 测试配置验证
        config = W8A8ProcessorConfig(cfg_map={"0": quant_config})
        config.model_validate(config.model_dump())

    def test_processor_config_empty_map(self):
        # 测试空配置映射
        config = W8A8ProcessorConfig(cfg_map={})
        assert len(config.cfg_map) == 0

    def test_processor_config_multiple_entries(self, quant_config):
        # 测试多个配置条目
        config = W8A8ProcessorConfig(cfg_map={
            "0": quant_config,
            "1": quant_config,
            "2": quant_config
        })
        assert len(config.cfg_map) == 3
        assert all(isinstance(cfg, W8A8QuantConfig) for cfg in config.cfg_map.values())

    def test_processor_config_json_serialization(self, quant_config):
        # 测试 JSON 序列化
        config = W8A8ProcessorConfig(cfg_map={"0": quant_config})
        json_str = config.model_dump_json()
        config_dict = json.loads(json_str)

        # 验证 JSON 结构
        assert "cfg_map" in config_dict
        assert "0" in config_dict["cfg_map"]
        assert "w_cfg" in config_dict["cfg_map"]["0"]
        assert "a_cfg" in config_dict["cfg_map"]["0"]

        # 验证配置值
        w_cfg = config_dict["cfg_map"]["0"]["w_cfg"]
        a_cfg = config_dict["cfg_map"]["0"]["a_cfg"]
        assert w_cfg["bits"] == 8
        assert w_cfg["method"] == WeightQuantMethod.MINMAX.value
        assert a_cfg["bits"] == 8
        assert a_cfg["method"] == WeightQuantMethod.MINMAX.value

    def test_processor_config_json_deserialization(self, quant_config):
        # 准备测试数据
        json_str = '''{
            "cfg_map": {
                "0": {
                    "w_cfg": {
                        "bits": 8,
                        "method": "minmax"
                    },
                    "a_cfg": {
                        "bits": 8,
                        "method": "minmax"
                    }
                }
            },
            "disable_names": [
                "abc"
            ]
        }'''

        # 测试 JSON 反序列化
        config = W8A8ProcessorConfig.model_validate_json(json_str)
        assert "0" in config.cfg_map
        assert isinstance(config.cfg_map["0"], W8A8QuantConfig)
        assert config.cfg_map["0"].w_cfg.bits == 8
        assert config.cfg_map["0"].w_cfg.method == WeightQuantMethod.MINMAX
        assert config.cfg_map["0"].a_cfg.bits == 8
        assert config.cfg_map["0"].a_cfg.method == ActivationQuantMethod.MINMAX
        assert config.disable_names == ["abc"]
