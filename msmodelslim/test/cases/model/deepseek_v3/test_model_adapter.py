# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from typing import List, Tuple

import torch
import torch.nn as nn

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.deepseek_v3.model_adapter import DeepSeekV3ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.qk_nope_head_dim = 64
        self.v_head_dim = 64


class DummyDecoderLayer(nn.Module):
    """模拟DecoderLayer"""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128)
        self.self_attn = type('SelfAttn', (), {
            'q_a_proj': nn.Linear(128, 128),
            'kv_a_proj_with_mqa': nn.Linear(128, 128),
            'q_b_proj': nn.Linear(128, 128),
            'kv_b_proj': nn.Linear(128, 128),
            'o_proj': nn.Linear(128, 128),
            'q_a_layernorm': nn.LayerNorm(128),
        })()

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class DummyModelInner(nn.Module):
    """模拟模型的内部model对象"""

    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(128)

    def forward(self, *args, **kwargs):
        return torch.randn(1, 10, 128)


class DummyModel(nn.Module):
    """模拟模型"""

    def __init__(self, num_layers=2):
        super().__init__()
        self.model = DummyModelInner(num_layers)
        self.lm_head = nn.Linear(128, 1000)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return torch.randn(1, 10, 1000)


class TestDeepSeekV3ModelAdapter(unittest.TestCase):

    def setUp(self):
        self.model_path = Path('.')
        self.model_type = 'DeepSeek-V3'

    def test_get_model_type_when_initialized_then_return_model_type(self):
        """测试get_model_type方法：初始化后应返回模型类型"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.model_type = self.model_type
            self.assertEqual(adapter.get_model_type(), self.model_type)

    def test_get_model_pedigree_when_called_then_return_deepseek_v3(self):
        """测试get_model_pedigree方法：调用时应返回deepseek_v3"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            self.assertEqual(adapter.get_model_pedigree(), 'deepseek_v3')

    @patch('msmodelslim.model.deepseek_v3.model_adapter.auto_convert_model_fp8_to_bf16')
    @patch('msmodelslim.model.deepseek_v3.model_adapter.SafeGenerator.get_model_from_pretrained')
    @patch('msmodelslim.model.deepseek_v3.model_adapter.warp_mtp_model')
    def test_init_model_when_called_then_increment_layers_and_wrap_model(
            self, mock_warp_mtp, mock_get_model, mock_auto_convert
    ):
        """测试init_model方法：调用时应增加层数并包装模型"""
        mock_model = DummyModel()
        mock_get_model.return_value = mock_model
        mock_warp_mtp.return_value = mock_model
        mock_auto_convert.return_value = None  # mock auto_convert_model_fp8_to_bf16

        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter._model_path = self.model_path
            adapter.config = DummyConfig()
            adapter._trust_remote_code = False

            original_layers = adapter.config.num_hidden_layers
            result = adapter.init_model(device=DeviceType.NPU)

            self.assertEqual(adapter.config.num_hidden_layers, original_layers + 1)
            self.assertEqual(adapter.config.num_hidden_layers, 3)
            self.assertIsNotNone(result)
            mock_warp_mtp.assert_called_once()
            mock_auto_convert.assert_called_once()

    def test_handle_dataset_when_called_then_return_tokenized_data(self):
        """测试handle_dataset方法：调用时应返回tokenized数据"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter._get_tokenized_data = MagicMock(return_value=['data1', 'data2'])

            dataset = ['test_data']
            result = adapter.handle_dataset(dataset, device=DeviceType.NPU)

            self.assertEqual(result, ['data1', 'data2'])
            adapter._get_tokenized_data.assert_called_once_with(dataset, DeviceType.NPU)

    def test_enable_kv_cache_when_called_then_register_hook(self):
        """测试enable_kv_cache方法：调用时应注册hook"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            model = DummyModel()

            # 测试启用kv_cache
            adapter.enable_kv_cache(model, True)

    def test_get_adapter_config_for_subgraph_when_called_then_return_fusion_configs(self):
        """测试get_adapter_config_for_subgraph方法：调用时应返回融合配置"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()

            result = adapter.get_adapter_config_for_subgraph()

            self.assertIsInstance(result, list)
            expected_configs = adapter.config.num_hidden_layers * 3
            self.assertEqual(len(result), expected_configs)
            self.assertIsInstance(result[0], AdapterConfig)

            subgraph_types = [config.subgraph_type for config in result]
            self.assertIn('ov', subgraph_types)
            self.assertIn('norm-linear', subgraph_types)

    def test_get_adapter_config_for_subgraph_structure_when_called_then_have_correct_structure(self):
        """测试get_adapter_config_for_subgraph方法：调用时应有正确的配置结构"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()

            result = adapter.get_adapter_config_for_subgraph()

            layer_0_configs = result[0:3]

            okv_config = layer_0_configs[0]
            self.assertEqual(okv_config.subgraph_type, 'ov')
            self.assertEqual(okv_config.mapping.source, 'model.layers.0.self_attn.kv_b_proj')
            self.assertIn('model.layers.0.self_attn.o_proj', okv_config.mapping.targets)

            norm_linear_config1 = layer_0_configs[1]
            self.assertEqual(norm_linear_config1.subgraph_type, 'norm-linear')
            self.assertEqual(norm_linear_config1.mapping.source, 'model.layers.0.input_layernorm')

            norm_linear_config2 = layer_0_configs[2]
            self.assertEqual(norm_linear_config2.subgraph_type, 'norm-linear')
            self.assertEqual(
                norm_linear_config2.mapping.source,
                'model.layers.0.self_attn.q_a_layernorm'
            )

    def test_enable_kv_cache_hook_functionality_when_called_then_register_pre_hooks(self):
        """测试enable_kv_cache方法：调用时应注册forward_pre_hooks"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            model = DummyModel()

            adapter.enable_kv_cache(model, True)

            # 验证hook被正确注册（检查_forward_pre_hooks是否不为空）
            self.assertGreater(len(model.model._forward_pre_hooks), 0)

            # 测试禁用kv_cache
            model2 = DummyModel()
            adapter.enable_kv_cache(model2, False)
            self.assertGreater(len(model2.model._forward_pre_hooks), 0)

    def test_get_adapter_config_for_subgraph_fusion_config_when_called_then_have_kv_fusion(self):
        """测试get_adapter_config_for_subgraph方法：调用时应包含KV融合配置"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()

            result = adapter.get_adapter_config_for_subgraph()

            # 验证第一个配置（OKV_b融合）有FusionConfig
            okv_config = result[0]
            self.assertIsNotNone(okv_config.fusion)
            self.assertEqual(okv_config.fusion.fusion_type, 'kv')
            self.assertEqual(okv_config.fusion.num_attention_heads, adapter.config.num_attention_heads)
            self.assertEqual(okv_config.fusion.num_key_value_heads, adapter.config.num_key_value_heads)

            # 验证custom_config
            self.assertIn('qk_nope_head_dim', okv_config.fusion.custom_config)
            self.assertIn('v_head_dim', okv_config.fusion.custom_config)
            self.assertEqual(okv_config.fusion.custom_config['qk_nope_head_dim'], adapter.config.qk_nope_head_dim)
            self.assertEqual(okv_config.fusion.custom_config['v_head_dim'], adapter.config.v_head_dim)

    def test_get_adapter_config_for_subgraph_when_zero_layers_then_return_empty_list(self):
        """测试get_adapter_config_for_subgraph方法：0层时应返回空列表"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 0

            result = adapter.get_adapter_config_for_subgraph()

            # 验证返回空列表
            self.assertEqual(len(result), 0)
            self.assertIsInstance(result, list)

    def test_get_adapter_config_for_subgraph_when_multiple_layers_then_return_all_configs(self):
        """测试get_adapter_config_for_subgraph方法：多层时应返回所有配置"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 5

            result = adapter.get_adapter_config_for_subgraph()

            # 验证返回正确数量的配置
            expected_count = 5 * 3  # 5层 * 3个配置
            self.assertEqual(len(result), expected_count)

            # 验证第二层的配置
            layer_1_configs = result[3:6]
            self.assertEqual(layer_1_configs[0].mapping.source, 'model.layers.1.self_attn.kv_b_proj')
            self.assertEqual(layer_1_configs[1].mapping.source, 'model.layers.1.input_layernorm')
            self.assertEqual(layer_1_configs[2].mapping.source, 'model.layers.1.self_attn.q_a_layernorm')

    def test_generate_model_forward_when_exception_in_forward_then_reraise(self):
        """测试generate_model_forward方法：前向传播异常时应重新抛出"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()

            # 创建一个会抛出异常的模型
            class ErrorModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = type('Model', (), {'layers': []})()

                def forward(self, *args, **kwargs):
                    raise RuntimeError("Forward error")

                def named_modules(self):
                    # 返回一个decoder layer以避免IndexError
                    dummy_layer = nn.Module()
                    dummy_layer.__class__.__name__ = 'DecoderLayer'
                    return [('', self), ('layer0', dummy_layer)]

            model = ErrorModel()
            inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}

            gen = adapter.generate_model_forward(model, inputs)

            # 验证会抛出RuntimeError
            with self.assertRaises(RuntimeError):
                list(gen)

    def test_generate_model_forward_when_first_block_input_none_then_raise_invalid_model_error(self):
        """测试generate_model_forward方法：无法获取第一个block输入时应抛出InvalidModelError"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()

            # 创建一个模型，但hook不会被触发（first_block_input保持None）
            class NoHookModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = type('Model', (), {'layers': [nn.Module()]})()
                    # 给layer设置正确的类名
                    self.model.layers[0].__class__.__name__ = 'DecoderLayer'

                def forward(self, *args, **kwargs):
                    # 正常返回，不触发TransformersForwardBreak
                    return None

                def named_modules(self):
                    return [('', self), ('layer0', self.model.layers[0])]

            model = NoHookModel()
            inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}

            gen = adapter.generate_model_forward(model, inputs)

            # 验证会抛出InvalidModelError
            with self.assertRaises(InvalidModelError) as context:
                list(gen)

            self.assertIn("Can't get first block input", str(context.exception))
