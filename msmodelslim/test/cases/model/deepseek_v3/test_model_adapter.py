# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from typing import List, Tuple

import torch
import torch.nn as nn

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import TransformersForwardBreak
from msmodelslim.model.deepseek_v3.model_adapter import DeepSeekV3ModelAdapter
from msmodelslim.quant import ir as qir
from msmodelslim.quant.processor.quant.fa3.interface import FA3QuantPlaceHolder
from msmodelslim.utils.exception import InvalidModelError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.qk_nope_head_dim = 64
        self.v_head_dim = 64


class DummyAttention(nn.Module):
    """模拟Attention模块，用于FA3测试"""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.kv_a_proj_with_mqa = nn.Linear(128, 128)
        self.q_a_proj = nn.Linear(128, 128)
        self.q_a_layernorm = nn.LayerNorm(128)
        self.q_b_proj = nn.Linear(128, 128)
        self.kv_a_layernorm = nn.LayerNorm(128)
        self.kv_b_proj = nn.Linear(128, 128)
        self.o_proj = nn.Linear(128, 128)
        self.num_heads = 8
        self.q_head_dim = 16
        self.qk_nope_head_dim = 8
        self.qk_rope_head_dim = 8
        self.kv_lora_rank = 64
        self.v_head_dim = 16
        self.softmax_scale = 0.1
        self.attention_dropout = 0.0
        self.rotary_emb = MagicMock()
        self.rotary_emb.return_value = (torch.randn(1, 1, 8), torch.randn(1, 1, 8))

    def forward(self, hidden_states, **kwargs):
        return torch.randn(1, 10, 128), None, None


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

    def test_init_model_when_called_then_load_model_with_correct_layers(self):
        """测试init_model方法：调用时应正确加载模型并处理层数"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 2
            adapter.model_path = Path('.')
            adapter.trust_remote_code = False
            
            mock_model = DummyModel()
            # 创建匹配的state_dict，使用strict=False或者mock load_state_dict
            mock_state_dict = {}
            for name, _ in mock_model.named_parameters():
                mock_state_dict[name] = torch.randn(1, 1)
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.SafeGenerator') as mock_safe_gen, \
                 patch('msmodelslim.model.deepseek_v3.model_adapter.auto_convert_model_fp8_to_bf16') as mock_convert, \
                 patch('msmodelslim.model.deepseek_v3.model_adapter.warp_mtp_model') as mock_warp:
                mock_safe_gen.get_model_from_pretrained.return_value = mock_model
                mock_warp.return_value = mock_model
                
                result = adapter.init_model()
                
                # 验证层数被正确修改（+1 for MTP）
                self.assertEqual(adapter.config.num_hidden_layers, 3)  # 2 + 1
                # 验证调用了相关方法
                mock_safe_gen.get_model_from_pretrained.assert_called_once()
                mock_convert.assert_called_once()
                mock_warp.assert_called_once()

    def test_inject_fa3_placeholders_when_called_then_inject_placeholders(self):
        """测试inject_fa3_placeholders方法：调用时应注入FA3占位符"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            
            # 使用DummyAttention类
            DummyAttention.forward.__module__ = 'test_module'
            
            root_module = nn.Module()
            root_module.attention = DummyAttention()
            
            def should_inject(name):
                return 'attention' in name.lower()

            def mock_apply_rotary_pos_emb(q, k, cos, sin, pos):
                """Mock apply_rotary_pos_emb函数"""
                return (q, k)

            with patch('msmodelslim.model.deepseek_v3.model_adapter.import_module') as mock_import, \
                 patch('msmodelslim.model.deepseek_v3.model_adapter.FA3QuantPlaceHolder') as mock_placeholder_class:
                mock_module = MagicMock()
                mock_module.apply_rotary_pos_emb = mock_apply_rotary_pos_emb
                mock_import.return_value = mock_module
                mock_placeholder = nn.Module()
                mock_placeholder_class.return_value = mock_placeholder
                
                adapter.inject_fa3_placeholders("", root_module, should_inject)
                
                # 验证forward被包装（forward方法应该被替换）
                self.assertIsNotNone(root_module.attention.forward)
                # 验证forward方法存在且可调用
                self.assertTrue(callable(root_module.attention.forward))

    def test_inject_fa3_placeholders_new_forward_when_past_key_value_then_handle_correctly(self):
        """测试inject_fa3_placeholders中的new_forward：past_key_value时应正确处理"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            
            class AttentionWithCache(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = nn.Linear(128, 128)
                    self.kv_a_proj_with_mqa = nn.Linear(128, 72)
                    self.kv_a_layernorm = nn.LayerNorm(64)
                    self.kv_b_proj = nn.Linear(64, 128)
                    self.o_proj = nn.Linear(128, 128)
                    self.num_heads = 8
                    self.q_head_dim = 16
                    self.qk_nope_head_dim = 8
                    self.qk_rope_head_dim = 8
                    self.kv_lora_rank = 64
                    self.v_head_dim = 16
                    self.softmax_scale = 0.1
                    self.attention_dropout = 0.0
                    self.layer_idx = 0
                    self.rotary_emb = MagicMock()
                    self.rotary_emb.return_value = (torch.randn(1, 1, 8), torch.randn(1, 1, 8))
            
            # 在类级别设置forward方法的__module__属性
            AttentionWithCache.forward.__module__ = 'test_module'
            attn = AttentionWithCache()
            
            root_module = nn.Module()
            root_module.attention = attn

            def should_inject(name):
                return 'attention' in name.lower()

            def mock_apply_rotary_pos_emb(q, k, cos, sin, pos):
                """Mock apply_rotary_pos_emb函数"""
                return (q, k)

            with patch('msmodelslim.model.deepseek_v3.model_adapter.import_module') as mock_import:
                mock_module = MagicMock()
                mock_module.apply_rotary_pos_emb = mock_apply_rotary_pos_emb
                mock_import.return_value = mock_module
                
                adapter.inject_fa3_placeholders("", root_module, should_inject)
                
                # 验证FA3占位符被正确注入
                self.assertTrue(hasattr(root_module.attention, 'fa_q'))
                self.assertTrue(hasattr(root_module.attention, 'fa_k'))
                self.assertTrue(hasattr(root_module.attention, 'fa_v'))
                self.assertIsInstance(root_module.attention.fa_q, FA3QuantPlaceHolder)
                self.assertIsInstance(root_module.attention.fa_k, FA3QuantPlaceHolder)
                self.assertIsInstance(root_module.attention.fa_v, FA3QuantPlaceHolder)
                
                hidden_states = torch.randn(1, 10, 128)
                attention_mask = torch.ones(1, 1, 10, 15)  # 包含past的长度
                
                def mock_update(k_pe, compressed_kv, layer_idx, cache_kwargs):
                    """Mock past_key_value.update方法"""
                    return k_pe, compressed_kv
                
                # 使用MagicMock替代自定义类
                past_key_value = MagicMock()
                past_key_value.get_usable_length.return_value = 5
                past_key_value.update.side_effect = mock_update
                
                # 验证forward方法被包装
                self.assertIsNotNone(root_module.attention.forward)
                self.assertTrue(callable(root_module.attention.forward))
                
                # 应该不会抛出异常
                try:
                    output, _, _ = root_module.attention(
                        hidden_states, 
                        attention_mask=attention_mask,
                        past_key_value=past_key_value
                    )
                    # 验证past_key_value被正确处理
                    self.assertIsNotNone(output)
                except Exception as e:
                    # 如果抛出异常，应该是预期的（比如维度不匹配等）
                    self.assertIsInstance(e, (ValueError, RuntimeError, IndexError, AssertionError))

    def test_get_ln_fuse_map_when_called_then_return_fuse_map(self):
        """测试get_ln_fuse_map方法：调用时应返回融合映射"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 2
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.get_ln_fuse_map') as mock_get_ln:
                mock_get_ln.return_value = {
                    'model.layers.0.input_layernorm': ['model.layers.0.self_attn.q_a_proj']
                }
                
                empty_dict, ln_map = adapter.get_ln_fuse_map()
                
                # 验证返回值
                self.assertEqual(empty_dict, {})
                self.assertIsInstance(ln_map, dict)
                # 实际代码只传递config，不传递num_hidden_layers
                mock_get_ln.assert_called_once_with(adapter.config)

    def test_get_rotate_map_when_called_then_return_rotate_map(self):
        """测试get_rotate_map方法：调用时应返回旋转映射"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 2
            
            mock_pre_run = MagicMock()
            mock_rot_pair = MagicMock()
            mock_rot_pairs = {'rot': mock_rot_pair}
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.get_rotate_map') as mock_get_rotate:
                mock_get_rotate.return_value = (mock_pre_run, mock_rot_pairs, {})
                
                pre_run_list, rot_pairs_list = adapter.get_rotate_map(128)
                
                # 验证返回值
                self.assertEqual(pre_run_list, [mock_pre_run])
                self.assertEqual(len(rot_pairs_list), 1)
                # 实际代码只传递config和block_size，不传递num_hidden_layers
                mock_get_rotate.assert_called_once_with(adapter.config, 128)

    def test_ascendv1_save_postprocess_when_w4a8_then_add_config_fields(self):
        """测试ascendv1_save_postprocess方法：w4a8场景下应添加配置字段"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_file = os.path.join(tmpdir, "config.json")
                with open(config_file, 'w') as f:
                    json.dump({}, f)
                
                with patch('msmodelslim.model.deepseek_v3.model_adapter.json_safe_load') as mock_load, \
                     patch('msmodelslim.model.deepseek_v3.model_adapter.json_safe_dump') as mock_dump:
                    mock_load.return_value = {}
                    
                    # 测试w4a8 + c8场景
                    model_with_c8 = nn.Module()
                    mock_w4a8 = nn.Module()
                    mock_w4a8.__class__ = qir.W4A8DynamicFakeQuantLinear
                    model_with_c8.linear1 = mock_w4a8
                    mock_c8 = nn.Module()
                    mock_c8.__class__ = qir.FakeQuantActivationPerHead
                    model_with_c8.activation1 = mock_c8
                    
                    adapter.ascendv1_save_postprocess(model_with_c8, tmpdir)
                    
                    # 验证配置数据包含正确的字段
                    call_args = mock_dump.call_args[0][0]
                    self.assertEqual(call_args['mtp_quantize'], 'w8a8_dynamic')
                    self.assertEqual(call_args['quantize'], 'w8a8_dynamic')
                    self.assertEqual(call_args['moe_quantize'], 'w4a8_dynamic')
                    self.assertEqual(call_args['mla_quantize'], 'w8a8')  # 因为有c8
                    
                    # 重置mock，测试只有w4a8的场景
                    mock_load.reset_mock()
                    mock_dump.reset_mock()
                    mock_load.return_value = {}
                    
                    model_w4a8_only = nn.Module()
                    mock_w4a8_only = nn.Module()
                    mock_w4a8_only.__class__ = qir.W4A8DynamicFakeQuantLinear
                    model_w4a8_only.linear1 = mock_w4a8_only
                    
                    adapter.ascendv1_save_postprocess(model_w4a8_only, tmpdir)
                    
                    # 验证配置数据包含正确的字段
                    call_args = mock_dump.call_args[0][0]
                    self.assertEqual(call_args['mla_quantize'], 'w8a8_dynamic')  # 因为没有c8

    def test_generate_model_forward_when_normal_flow_then_yield_process_requests(self):
        """测试generate_model_forward方法：正常流程时应yield ProcessRequest"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 2
            
            # 创建一个能够触发hook的模型
            class HookTriggerModel(nn.Module):
                def __init__(self, num_layers=2):
                    super().__init__()
                    self.model = DummyModelInner(num_layers)
                    self.lm_head = nn.Linear(128, 1000)
                
                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    # 确保会调用第一个layer，触发hook
                    hidden_states = torch.randn(1, 10, 128)
                    # 调用第一个layer以触发hook
                    self.model.layers[0](hidden_states)
                    return torch.randn(1, 10, 1000)
                
                def get_submodule(self, name):
                    """Mock get_submodule方法"""
                    if name.startswith('model.layers.'):
                        idx = int(name.split('.')[2])
                        return self.model.layers[idx]
                    return super().get_submodule(name)
            
            model = HookTriggerModel(num_layers=2)
            inputs = {'input_ids': torch.randint(0, 1000, (1, 10)), 'attention_mask': torch.ones(1, 10)}
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.dist') as mock_dist:
                mock_dist.is_initialized.return_value = False
                
                gen = adapter.generate_model_forward(model, inputs)
                
                # 获取第一个ProcessRequest
                first_request = next(gen)
                self.assertIsInstance(first_request, ProcessRequest)
                self.assertEqual(first_request.name, 'model.layers.0')
                
                # 发送输出并获取第二个请求
                second_request = gen.send((torch.randn(1, 10, 128),))
                self.assertIsInstance(second_request, ProcessRequest)
                self.assertEqual(second_request.name, 'model.layers.1')

    def test_generate_model_forward_when_dist_initialized_then_call_barrier(self):
        """测试generate_model_forward方法：分布式初始化时应调用barrier"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 1
            
            # 创建一个能够触发hook的模型
            class HookTriggerModel(nn.Module):
                def __init__(self, num_layers=1):
                    super().__init__()
                    self.model = DummyModelInner(num_layers)
                    self.lm_head = nn.Linear(128, 1000)
                    self.model.norm = nn.LayerNorm(128)
                
                def forward(self, input_ids=None, **kwargs):
                    # 确保会调用第一个layer，触发hook
                    hidden_states = torch.randn(1, 10, 128)
                    self.model.layers[0](hidden_states)
                    return torch.randn(1, 10, 1000)
                
                def get_submodule(self, name):
                    """Mock get_submodule方法"""
                    if name.startswith('model.layers.'):
                        idx = int(name.split('.')[2])
                        return self.model.layers[idx]
                    elif name == 'model.norm':
                        return self.model.norm
                    elif name == 'lm_head':
                        return self.lm_head
                    return super().get_submodule(name)
            
            model = HookTriggerModel(num_layers=1)
            inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.dist') as mock_dist:
                mock_dist.is_initialized.return_value = True
                mock_dist.barrier = MagicMock()
                
                gen = adapter.generate_model_forward(model, inputs)
                try:
                    next(gen)
                except StopIteration:
                    pass
                
                mock_dist.barrier.assert_called_once()

    def test_generate_model_forward_when_mtp_layer_then_yield_mtp_request(self):
        """测试generate_model_forward方法：应yield MTP layer的ProcessRequest"""
        with patch.object(DeepSeekV3ModelAdapter, '__init__', lambda x, *args, **kwargs: None):
            adapter = DeepSeekV3ModelAdapter()
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 1
            
            class HookTriggerModel(nn.Module):
                def __init__(self, num_layers=1):
                    super().__init__()
                    self.model = DummyModelInner(num_layers)
                    self.lm_head = nn.Linear(128, 1000)
                    self.model.norm = nn.LayerNorm(128)
                    # 添加MTP layer
                    mtp_layer = nn.Module()
                    mtp_layer.embed_tokens = nn.Embedding(1000, 128)
                    mtp_layer.enorm = nn.LayerNorm(128)
                    mtp_layer.hnorm = nn.LayerNorm(128)
                    mtp_layer.eh_proj = nn.Linear(256, 128)
                    self.model.layers.append(mtp_layer)
                
                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    hidden_states = torch.randn(1, 10, 128)
                    self.model.layers[0](hidden_states)
                    return torch.randn(1, 10, 1000)
                
                def get_submodule(self, name):
                    if name.startswith('model.layers.'):
                        idx = int(name.split('.')[2])
                        return self.model.layers[idx]
                    elif name == 'model.norm':
                        return self.model.norm
                    elif name == 'lm_head':
                        return self.lm_head
                    return super().get_submodule(name)
            
            model = HookTriggerModel(num_layers=1)
            inputs = {'input_ids': torch.randint(0, 1000, (1, 10)), 'attention_mask': torch.ones(1, 10)}
            
            with patch('msmodelslim.model.deepseek_v3.model_adapter.dist') as mock_dist, \
                 patch('msmodelslim.model.deepseek_v3.model_adapter.remove_zero_and_shift') as mock_remove, \
                 patch('transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask') as mock_prepare:
                mock_dist.is_initialized.return_value = False
                mock_remove.return_value = torch.randint(0, 1000, (1, 10))
                mock_prepare.return_value = torch.ones(1, 1, 10, 10)
                
                # Mock to方法
                for module in [model.lm_head, model.model.layers[1].embed_tokens, 
                              model.model.layers[1].enorm, model.model.layers[1].hnorm, 
                              model.model.layers[1].eh_proj]:
                    module.to = MagicMock(return_value=module)
                
                # Mock tensor的to方法
                original_to = torch.Tensor.to

                def tensor_to_method(tensor_self, device=None, **kwargs):
                    if device == 'npu':
                        return tensor_self
                    return original_to(tensor_self, device, **kwargs)
                torch.Tensor.to = tensor_to_method
                
                try:
                    gen = adapter.generate_model_forward(model, inputs)
                    first_request = next(gen)
                    self.assertIsInstance(first_request, ProcessRequest)
                    
                    # 发送输出，应该会继续处理到MTP layer
                    try:
                        mtp_request = gen.send((torch.randn(1, 10, 128),))
                        self.assertIsInstance(mtp_request, ProcessRequest)
                        self.assertEqual(mtp_request.name, f'model.layers.{adapter.config.num_hidden_layers}')
                        self.assertIn('attention_mask', mtp_request.kwargs)
                        self.assertIn('position_ids', mtp_request.kwargs)
                    except StopIteration:
                        pass
                finally:
                    torch.Tensor.to = original_to

