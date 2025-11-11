# -*- coding: utf-8 -*-

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

from functools import partial
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.flex_smooth_quant import (
    FlexSmoothQuantProcessor,
    FlexSmoothQuantProcessorConfig
)
from msmodelslim.quant.processor.anti_outlier.smooth_base import StatKey
from msmodelslim.quant.processor.anti_outlier.smooth_interface import FlexSmoothQuantInterface
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig, FusionConfig



class MockModel(nn.Module):
    """模拟模型用于测试"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.norm_layer = nn.LayerNorm(10)
        
    def get_submodule(self, name):
        """模拟get_submodule方法"""
        if name == "layer1":
            return self.layer1
        elif name == "layer2":
            return self.layer2
        elif name == "model.layers.0.input_layernorm":
            return self.norm_layer
        return None
    
    def set_submodule(self, name, module):
        """模拟set_submodule方法"""
        if name == "model.layers.0.input_layernorm":
            self.norm_layer = module
    
    def named_modules(self):
        """模拟named_modules方法"""
        modules = [
            ('layer1', self.layer1),
            ('layer2', self.layer2), 
            ('norm_layer', self.norm_layer)
        ]
        for name, module in modules:
            yield name, module


class MockFlexSmoothQuantInterface(FlexSmoothQuantInterface):
    """模拟实现FlexSmoothQuantInterface的适配器"""
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="model.layers.0.input_layernorm", targets=["layer1"])
            )
        ]



class MockDistHelper:
    """模拟分布式助手"""

    def __init__(self, model):
        self.model = model
        self.shared_modules = {}

    @staticmethod
    def gather_variable_shapes(tensor: torch.Tensor):
        """模拟收集变量形状"""
        return [tensor.clone(), tensor.clone()]

    def is_shared(self, name: str) -> bool:
        return name in self.shared_modules


class TestFlexSmoothQuantProcessor:
    """FlexSmoothQuantProcessor 测试类"""

    model: MockModel = None
    adapter: MockFlexSmoothQuantInterface = None
    default_config: FlexSmoothQuantProcessorConfig = None

    @staticmethod
    def test_config_validation():
        """测试配置验证"""
        # 测试 alpha 值验证
        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(alpha=-1.0)

        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(alpha=2.0)

        # 测试 beta 值验证
        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(beta=-0.5)

        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(beta=1.5)

    def setup_method(self):
        """测试前的设置"""
        self.model = MockModel()
        self.adapter = MockFlexSmoothQuantInterface()
        self.default_config = FlexSmoothQuantProcessorConfig(
            alpha=0.5,
            beta=0.8,
            enable_subgraph_type=["norm-linear", "linear-linear"]
        )

    def test_init_with_valid_adapter(self):
        """测试使用有效适配器初始化"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert processor.model == self.model
        assert processor.config == self.default_config
        assert processor.adapter == self.adapter
        assert isinstance(processor.act_stats, dict)
        assert processor.config.alpha == 0.5
        assert processor.config.beta == 0.8
        assert processor.config.enable_subgraph_type == ["norm-linear", "linear-linear"]

    def test_init_with_invalid_adapter(self):
        """测试使用无效适配器初始化"""
        invalid_adapter = Mock()

        with pytest.raises(UnsupportedError) as exc_info:
            FlexSmoothQuantProcessor(
                model=self.model,
                config=self.default_config,
                adapter=invalid_adapter
            )

        assert "does not support FlexSmooth" in str(exc_info.value)
        assert "Please provide a valid model adapter" in str(exc_info.value)

    def test_config_default_values(self):
        """测试配置默认值"""
        config = FlexSmoothQuantProcessorConfig()
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=config,
            adapter=self.adapter
        )

        assert processor.config.alpha is None
        assert processor.config.beta is None
        assert processor.config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    def test_support_distributed(self):
        """测试分布式支持"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert processor.support_distributed() is True

    def test_preprocess(self):
        """测试预处理方法"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        processor.global_adapter_config = self.adapter.get_adapter_config_for_subgraph()

        request = BatchProcessRequest(
            name="model.layers.0",
            module=self.model.layer1
        )

        # 预处理方法应该正常执行，不会抛出异常
        processor.preprocess(request)

    @patch('torch.distributed.is_initialized')
    def test_init_with_distributed(self, mock_dist):
        """测试分布式环境初始化"""
        mock_dist.return_value = True

        with patch('msmodelslim.quant.processor.anti_outlier.flex_smooth_quant.DistHelper') as mock_dist_helper:
            mock_dist_helper.return_value = MockDistHelper(self.model)

            processor = FlexSmoothQuantProcessor(
                model=self.model,
                config=self.default_config,
                adapter=self.adapter
            )

            assert processor.dist_helper is not None
            assert isinstance(processor.dist_helper, MockDistHelper)

    @patch('torch.distributed.is_initialized')
    def test_init_without_distributed(self, mock_dist):
        """测试非分布式环境初始化"""
        mock_dist.return_value = False

        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert processor.dist_helper is None

    def test_get_stats_hook(self):
        """测试统计钩子生成"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        # 创建钩子函数
        hook = processor._get_stats_hook("test_module")
        assert callable(hook)
        assert isinstance(hook, partial)

        # 测试钩子函数执行
        mock_module = nn.Linear(10, 20)
        input_tensor = (torch.randn(5, 10),)
        output = torch.randn(5, 20)

        hook(mock_module, input_tensor, output)

        # 验证统计数据已更新
        assert "test_module" in processor.act_stats
        stats_dict = processor.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    def test_get_stats_hook_with_multiple_calls(self):
        """测试钩子函数多次调用"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor._get_stats_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # 第一次调用
        input_tensor1 = (torch.randn(3, 10),)
        hook(mock_module, input_tensor1, torch.randn(3, 20))

        # 第二次调用
        input_tensor2 = (torch.randn(4, 10),)
        hook(mock_module, input_tensor2, torch.randn(4, 20))

        # 验证多次调用后的统计结果
        stats_dict = processor.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert len(stats_dict[StatKey.TENSOR]) == 2  # 应该有两个张量
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    def test_get_stats_hook_with_different_shapes(self):
        """测试不同形状输入的处理"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor._get_stats_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # 测试不同批次大小
        for batch_size in [1, 5, 10]:
            input_tensor = (torch.randn(batch_size, 10),)
            output = torch.randn(batch_size, 20)

            hook(mock_module, input_tensor, output)

        # 验证统计结果
        stats_dict = processor.act_stats["test_module"]
        assert len(stats_dict[StatKey.TENSOR]) == 3  # 应该有3个张量

    @patch('msmodelslim.quant.processor.anti_outlier.flex_smooth_quant.DistHelper')
    def test_get_stats_hook_distributed(self, mock_dist_helper_class):
        """测试分布式环境下的钩子函数"""
        mock_dist_helper = MockDistHelper(self.model)
        mock_dist_helper_class.return_value = mock_dist_helper

        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor._get_stats_hook("test_module")
        mock_module = nn.Linear(10, 20)
        input_tensor = (torch.randn(5, 10),)
        output = torch.randn(5, 20)

        # 模拟共享模块的情况
        mock_dist_helper.is_shared = Mock(return_value=True)

        hook(mock_module, input_tensor, output)

        # 验证统计结果
        assert "test_module" in processor.act_stats
        stats_dict = processor.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    @patch('msmodelslim.quant.processor.anti_outlier.flex_smooth_quant.flex_smooth_quant')
    @patch('msmodelslim.quant.processor.anti_outlier.flex_smooth_quant.get_logger')
    def test_flex_smooth_quant_processor_apply_smooth(self, mock_logger, mock_flex_smooth_quant):
        """验证_apply_smooth_to_subgraph方法是否能正确应用平滑配置并处理子图"""
        # 前置操作：创建一个子图对象和线性模块列表，调用_apply_smooth_to_subgraph方法
        model = MockModel()
        config = FlexSmoothQuantProcessorConfig()
        adapter = MockFlexSmoothQuantInterface()
        
        processor = FlexSmoothQuantProcessor(model, config, adapter)
        
        # 创建模拟的子图对象和线性模块列表
        subgraph_obj = Mock()
        linear_modules = [model.layer1, model.layer2]
        
        # 模拟_build_smooth_context方法
        mock_smooth_context = Mock()
        processor._build_smooth_context = Mock(return_value=mock_smooth_context)
        
        # 调用_apply_smooth_to_subgraph方法
        processor._apply_smooth_to_subgraph(subgraph_obj, linear_modules)
        
        # 验证_build_smooth_context被调用
        processor._build_smooth_context.assert_called_once_with(linear_modules)
        
        # 验证flex_smooth_quant被调用，参数正确
        mock_flex_smooth_quant.assert_called_once()
        call_args = mock_flex_smooth_quant.call_args
        
        # 验证参数
        assert call_args[0][0] == subgraph_obj  # subgraph_obj
        assert call_args[0][2] == mock_smooth_context  # smooth_context
        
        # 验证FlexSmoothQuantConfig参数
        smooth_config = call_args[0][1]
        assert smooth_config.alpha == config.alpha
        assert smooth_config.beta == config.beta

    def test_act_stats_initialization(self):
        """测试激活统计初始化"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert isinstance(processor.act_stats, dict)
        assert len(processor.act_stats) == 0

    def test_hook_handles_initialization(self):
        """测试钩子句柄初始化"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert isinstance(processor.hook_handles, dict)
        assert len(processor.hook_handles) == 0

    def test_gradient_tracking_in_tensor(self):
        """测试张量梯度跟踪"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor._get_stats_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # 创建需要梯度的张量
        input_tensor = (torch.randn(5, 10).requires_grad_(True),)
        output = torch.randn(5, 20)

        hook(mock_module, input_tensor, output)

        # 验证统计结果正确获取
        stats_dict = processor.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

        # 验证张量形状
        tensor_list = stats_dict[StatKey.TENSOR]
        assert len(tensor_list) == 1
        assert tensor_list[0].shape[-1] == 10  # hidden_dim

    def test_channel_max_calculation(self):
        """测试通道最大值计算"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor._get_stats_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # 创建已知的张量值来测试通道最大值计算
        tensor_values = torch.tensor([[1.0, -2.0, 3.0],
                                      [-1.5, 2.5, -3.5]])
        input_tensor = (tensor_values,)
        output = torch.randn(2, 20)

        hook(mock_module, input_tensor, output)

        # 验证通道最大值
        stats_dict = processor.act_stats["test_module"]
        channel_max = stats_dict[StatKey.STAT_KEY_SMOOTH_SCALE]
        expected_max = torch.tensor([1.5, 2.5, 3.5])  # 每列的绝对值最大值

        assert torch.allclose(channel_max, expected_max)

    def test_multiple_modules_stats_accumulation(self):
        """测试多个模块的统计累积"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        mock_module1 = nn.Linear(10, 20)
        mock_module2 = nn.Linear(10, 20)

        hook1 = processor._get_stats_hook("module1")
        hook2 = processor._get_stats_hook("module2")

        # 为两个模块记录统计信息
        input_tensor1 = (torch.randn(3, 10),)
        output1 = torch.randn(3, 20)
        hook1(mock_module1, input_tensor1, output1)

        input_tensor2 = (torch.randn(4, 10),)
        output2 = torch.randn(4, 20)
        hook2(mock_module2, input_tensor2, output2)

        # 验证两个模块的统计信息都被正确记录
        assert "module1" in processor.act_stats
        assert "module2" in processor.act_stats

        assert len(processor.act_stats["module1"][StatKey.TENSOR]) == 1
        assert len(processor.act_stats["module2"][StatKey.TENSOR]) == 1



class TestFlexSmoothQuantProcessorConfig:
    """FlexSmoothQuantProcessorConfig 配置测试类"""

    @staticmethod
    def test_default_config():
        """测试默认配置"""
        config = FlexSmoothQuantProcessorConfig()

        assert config.type == "flex_smooth_quant"
        assert config.alpha is None
        assert config.beta is None
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    @staticmethod
    def test_custom_config():
        """测试自定义配置"""
        config = FlexSmoothQuantProcessorConfig(
            alpha=0.7,
            beta=0.9,
            enable_subgraph_type=["norm-linear", "linear-linear"]
        )

        assert config.type == "flex_smooth_quant"
        assert config.alpha == 0.7
        assert config.beta == 0.9
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear"]

    @staticmethod
    def test_config_type_literal():
        """测试配置类型字面量"""
        config = FlexSmoothQuantProcessorConfig()

        # 测试 Literal 类型的默认值
        assert isinstance(config.type, str)
        assert config.type == "flex_smooth_quant"

    @staticmethod
    def test_alpha_validation():
        """测试 alpha 值验证"""
        valid_config = FlexSmoothQuantProcessorConfig(alpha=0.5)
        assert valid_config.alpha == 0.5

    @staticmethod
    def test_beta_validation():
        """测试 beta 值验证"""      
        valid_config = FlexSmoothQuantProcessorConfig(beta=0.8)
        assert valid_config.beta == 0.8

    @staticmethod
    def test_enable_subgraph_type_validation():
        """测试子图类型验证"""
        # 有效列表
        valid_config = FlexSmoothQuantProcessorConfig(
            enable_subgraph_type=["norm-linear"]
        )
        assert valid_config.enable_subgraph_type == ["norm-linear"]

        # 多子图类型
        valid_config = FlexSmoothQuantProcessorConfig(
            enable_subgraph_type=["norm-linear", "linear-linear", "ov"]
        )
        assert valid_config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov"]
