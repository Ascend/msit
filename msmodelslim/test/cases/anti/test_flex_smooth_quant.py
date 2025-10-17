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

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional

from msmodelslim.core.KIA.impl.flex_smooth_quant import (
    quant_int8sym,
    quant_int8tasym,
    scale_descale,
    search_alpha_beta,
    compute_smooth_scale,
    apply_smooth_scale_shift,
    prepare_mqga_parameters,
    reduce_scales_for_mqga,
    flex_smooth_impl_OV,
    flex_smooth_impl_UpDown,
    flex_smooth_impl_LinearLinear,
    flex_smooth_impl_NormLinear,
)
from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
    SmoothContext,
    FlexSmoothQuantConfig,
)
from msmodelslim.utils.exception import MisbehaviorError, UnexpectedError


class TestQuantizationFunctions:
    """测试量化相关的基础函数"""

    @staticmethod
    def test_quant_int8sym_basic():
        """测试对称int8量化的基本功能"""
        # 测试正常情况
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = quant_int8sym(x)
        
        # 验证输出形状
        assert result.shape == x.shape
        # 验证数据类型
        assert result.dtype == x.dtype
        # 验证值在合理范围内
        assert torch.all(result >= -127)
        assert torch.all(result <= 127)

    @staticmethod
    def test_quant_int8tasym_basic():
        """测试非对称int8量化的基本功能"""
        # 测试正常情况
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = quant_int8tasym(x)
        
        # 验证输出形状
        assert result.shape == x.shape
        # 验证数据类型
        assert result.dtype == x.dtype

    @staticmethod
    def test_quant_int8tasym_edge_cases():
        """测试非对称int8量化的边界情况"""
        # 测试零张量
        x_zero = torch.zeros(2, 3)
        result_zero = quant_int8tasym(x_zero)
        assert torch.allclose(result_zero, x_zero)
        
        # 测试负值
        x_neg = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        result_neg = quant_int8tasym(x_neg)
        assert result_neg.shape == x_neg.shape
        
        # 测试单值张量
        x_single = torch.tensor([5.0])
        result_single = quant_int8tasym(x_single)
        assert result_single.shape == x_single.shape

    @staticmethod
    def test_scale_descale_basic():
        """测试尺度缩放和反缩放的基本功能"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        alpha = 0.5
        beta = 0.5
        
        result = scale_descale(act, fc_weights, alpha, beta)
        
        # 验证返回的是标量
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1
        assert result >= 0  # MSE应该非负

    @staticmethod
    def test_scale_descale_with_asym():
        """测试非对称激活的尺度缩放"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        alpha = 0.3
        beta = 0.7
        
        # 测试对称激活
        result_sym = scale_descale(act, fc_weights, alpha, beta, act_sym=True)
        # 测试非对称激活
        result_asym = scale_descale(act, fc_weights, alpha, beta, act_sym=False)
        
        assert isinstance(result_sym, torch.Tensor)
        assert isinstance(result_asym, torch.Tensor)
        assert result_sym.numel() == 1
        assert result_asym.numel() == 1

    @staticmethod
    def test_search_alpha_beta_basic():
        """测试alpha和beta搜索的基本功能"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        
        best_p, best_mse = search_alpha_beta(act, fc_weights)
        
        # 验证返回值
        assert isinstance(best_p, float)
        assert isinstance(best_mse, torch.Tensor)
        assert 0.0 <= best_p <= 1.0
        assert best_mse >= 0

    @staticmethod
    def test_search_alpha_beta_with_best_alpha():
        """测试给定最佳alpha时的beta搜索"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        best_alpha = 0.5
        
        best_p, best_mse = search_alpha_beta(act, fc_weights, best_alpha=best_alpha)
        
        assert isinstance(best_p, float)
        assert isinstance(best_mse, torch.Tensor)
        assert 0.0 <= best_p <= 1.0
        assert best_mse >= 0

    @staticmethod
    def test_compute_smooth_scale_basic():
        """测试平滑尺度计算的基本功能"""
        a_scale = torch.tensor([1.0, 2.0, 3.0])
        w_scale = torch.tensor([0.5, 1.5, 2.5])
        alpha = 0.5
        beta = 0.5
        
        result = compute_smooth_scale(a_scale, w_scale, alpha, beta)
        
        assert result.shape == a_scale.shape
        assert torch.all(result > 0)  # 尺度应该为正
        assert result.dtype == a_scale.dtype

    @staticmethod
    def test_compute_smooth_scale_edge_cases():
        """测试平滑尺度计算的边界情况"""
        # 测试零值
        a_scale = torch.tensor([0.0, 1.0, 2.0])
        w_scale = torch.tensor([1.0, 0.0, 1.0])
        alpha = 0.5
        beta = 0.5
        
        result = compute_smooth_scale(a_scale, w_scale, alpha, beta)
        assert torch.all(result >= 1e-5)  # 应该应用最小阈值

    @staticmethod
    def test_apply_smooth_scale_shift():
        """测试平滑尺度应用"""
        # 创建模拟层
        layer = Mock()
        layer.weight = torch.randn(8, 4)
        original_weight = layer.weight.clone()
        
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        apply_smooth_scale_shift(layer, scales)
        
        # 验证权重被修改
        assert not torch.allclose(layer.weight, original_weight)

    @staticmethod
    def test_prepare_mqga_parameters():
        """测试MQGA参数准备"""
        num_attention_heads = 8
        num_key_value_heads = 2
        
        ratio, pad_size = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        
        assert ratio == 4  # 8 // 2
        assert pad_size == 0

    @staticmethod
    def test_reduce_scales_for_mqga():
        """测试MQGA尺度缩减"""
        scales = torch.randn(16)  # 假设16个尺度
        shape_ratio = 4
        num_attention_heads = 8
        
        updated_scales, reduced_scales = reduce_scales_for_mqga(scales, shape_ratio, num_attention_heads)
        
        assert updated_scales.shape == scales.shape
        assert reduced_scales.numel() == scales.numel() // shape_ratio


class TestFlexSmoothImplOV:
    """测试OV子图的平滑实现"""

    @staticmethod
    def create_mock_ov_subgraph():
        """创建模拟的OV子图"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = Mock()
        subgraph.o_proj = Mock()
        subgraph.num_attention_heads = 8
        subgraph.key_value_heads = 2
        
        # 设置权重
        subgraph.o_proj.weight = torch.randn(8, 16)
        subgraph.v_proj.weight = torch.randn(4, 8)
        
        # 设置参数 - 返回迭代器而不是列表
        subgraph.v_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_ov_basic():
        """测试OV平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        # 应该正常执行而不抛出异常
        flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_with_provided_params():
        """测试使用提供的alpha和beta参数"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = TestFlexSmoothImplOV.create_mock_context()
        config = TestFlexSmoothImplOV.create_mock_config(alpha=0.5, beta=0.5)
        
        flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_missing_v_proj():
        """测试缺少v_proj时的错误处理"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = None
        subgraph.o_proj = Mock()
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_missing_o_proj():
        """测试缺少o_proj时的错误处理"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = Mock()
        subgraph.o_proj = None
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_no_tensors():
        """测试没有张量时的处理"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = Mock(spec=SmoothContext)
        context.tensors = None
        context.a_smooth_scale = torch.randn(16)
        
        config = TestFlexSmoothImplOV.create_mock_config()
        
        # 应该正常返回而不抛出异常
        flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_no_valid_tensors():
        """测试没有有效张量时的处理"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = Mock(spec=SmoothContext)
        context.tensors = [None, "invalid"]
        context.a_smooth_scale = torch.randn(16)
        
        config = TestFlexSmoothImplOV.create_mock_config()
        
        # 应该正常返回而不抛出异常
        flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_no_a_scale():
        """测试缺少激活尺度时的错误处理"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = None
        
        config = TestFlexSmoothImplOV.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_OV(subgraph, config, context)


class TestFlexSmoothImplUpDown:
    """测试Up-Down子图的平滑实现"""

    @staticmethod
    def create_mock_updown_subgraph():
        """创建模拟的Up-Down子图"""
        subgraph = Mock(spec=UpDownSubgraph)
        subgraph.up_proj = Mock()
        subgraph.down_proj = Mock()
        subgraph.gate_proj = None  # 可选
        
        # 设置权重
        subgraph.down_proj.weight = torch.randn(8, 16)
        subgraph.up_proj.weight = torch.randn(16, 8)
        
        # 设置参数
        subgraph.up_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_updown_basic():
        """测试Up-Down平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplUpDown.create_mock_updown_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_UpDown(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_updown_with_gate_proj():
        """测试包含gate_proj的Up-Down平滑实现"""
        subgraph = TestFlexSmoothImplUpDown.create_mock_updown_subgraph()
        subgraph.gate_proj = Mock()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_UpDown(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_updown_missing_up_proj():
        """测试缺少up_proj时的错误处理"""
        subgraph = Mock(spec=UpDownSubgraph)
        subgraph.up_proj = None
        subgraph.down_proj = Mock()
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_UpDown(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_updown_missing_down_proj():
        """测试缺少down_proj时的错误处理"""
        subgraph = Mock(spec=UpDownSubgraph)
        subgraph.up_proj = Mock()
        subgraph.down_proj = None
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_UpDown(subgraph, config, context)


class TestFlexSmoothImplLinearLinear:
    """测试Linear-Linear子图的平滑实现"""

    @staticmethod
    def create_mock_linearlinear_subgraph():
        """创建模拟的Linear-Linear子图"""
        subgraph = Mock(spec=LinearLinearSubgraph)
        subgraph.linear1 = Mock()
        subgraph.linear2 = Mock()
        
        # 设置权重
        subgraph.linear2.weight = torch.randn(8, 16)
        subgraph.linear1.weight = torch.randn(16, 8)
        
        # 设置参数
        subgraph.linear1.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_linearlinear_basic():
        """测试Linear-Linear平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplLinearLinear.create_mock_linearlinear_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_LinearLinear(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_linearlinear_missing_linear1():
        """测试缺少linear1时的错误处理"""
        subgraph = Mock(spec=LinearLinearSubgraph)
        subgraph.linear1 = None
        subgraph.linear2 = Mock()
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_LinearLinear(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_linearlinear_missing_linear2():
        """测试缺少linear2时的错误处理"""
        subgraph = Mock(spec=LinearLinearSubgraph)
        subgraph.linear1 = Mock()
        subgraph.linear2 = None
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_LinearLinear(subgraph, config, context)


class TestFlexSmoothImplNormLinear:
    """测试Norm-Linear子图的平滑实现"""

    @staticmethod
    def create_mock_normlinear_subgraph():
        """创建模拟的Norm-Linear子图"""
        subgraph = Mock(spec=NormLinearSubgraph)
        subgraph.norm = Mock()
        subgraph.linears = [Mock(), Mock()]  # 多个线性层
        
        # 设置权重
        for linear in subgraph.linears:
            linear.weight = torch.randn(8, 16)
        
        # 设置参数
        subgraph.norm.parameters.return_value = iter([torch.randn(16)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_normlinear_basic():
        """测试Norm-Linear平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplNormLinear.create_mock_normlinear_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_NormLinear(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_normlinear_missing_norm():
        """测试缺少norm时的错误处理"""
        subgraph = Mock(spec=NormLinearSubgraph)
        subgraph.norm = None
        subgraph.linears = [Mock()]
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_NormLinear(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_normlinear_missing_linears():
        """测试缺少linears时的错误处理"""
        subgraph = Mock(spec=NormLinearSubgraph)
        subgraph.norm = Mock()
        subgraph.linears = None
        
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        with pytest.raises(MisbehaviorError):
            flex_smooth_impl_NormLinear(subgraph, config, context)


class TestErrorHandling:
    """测试错误处理"""

    @staticmethod
    def test_unexpected_error_in_alpha_beta_search():
        """测试alpha/beta搜索中的意外错误"""
        # 创建会导致搜索函数出错的输入
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        
        # 模拟search_alpha_beta函数抛出异常
        with patch('msmodelslim.core.KIA.impl.flex_smooth_quant.search_alpha_beta') as mock_search:
            mock_search.side_effect = Exception("Test error")
            
            subgraph = Mock(spec=OVSubgraph)
            subgraph.v_proj = Mock()
            subgraph.o_proj = Mock()
            subgraph.num_attention_heads = 8
            subgraph.key_value_heads = 2
            subgraph.o_proj.weight = fc_weights
            subgraph.v_proj.parameters.return_value = iter([torch.randn(16, 8)])
            
            context = Mock(spec=SmoothContext)
            context.tensors = [act]
            context.a_smooth_scale = torch.randn(8)
            
            config = Mock(spec=FlexSmoothQuantConfig)
            config.alpha = None
            config.beta = None
            
            with pytest.raises(UnexpectedError):
                flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_unexpected_error_in_scale_computation():
        """测试尺度计算中的意外错误"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = Mock()
        subgraph.o_proj = Mock()
        subgraph.num_attention_heads = 8
        subgraph.key_value_heads = 2
        subgraph.o_proj.weight = torch.randn(8, 16)
        subgraph.v_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = 0.5
        config.beta = 0.5
        
        # 模拟compute_smooth_scale函数抛出异常
        with patch('msmodelslim.core.KIA.impl.flex_smooth_quant.compute_smooth_scale') as mock_compute:
            mock_compute.side_effect = Exception("Scale computation error")
            
            with pytest.raises(UnexpectedError):
                flex_smooth_impl_OV(subgraph, config, context)

    @staticmethod
    def test_unexpected_error_in_scale_application():
        """测试尺度应用中的意外错误"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = Mock()
        subgraph.o_proj = Mock()
        subgraph.num_attention_heads = 8
        subgraph.key_value_heads = 2
        subgraph.o_proj.weight = torch.randn(8, 16)
        subgraph.v_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = 0.5
        config.beta = 0.5
        
        # 模拟apply_smooth_scale_shift函数抛出异常
        with patch('msmodelslim.core.KIA.impl.flex_smooth_quant.apply_smooth_scale_shift') as mock_apply:
            mock_apply.side_effect = Exception("Scale application error")
            
            with pytest.raises(UnexpectedError):
                flex_smooth_impl_OV(subgraph, config, context)

