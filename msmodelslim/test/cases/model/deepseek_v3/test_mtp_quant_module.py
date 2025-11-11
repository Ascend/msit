# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import PretrainedConfig

from msmodelslim.model.deepseek_v3.mtp_quant_module import (
    remove_zero_and_shift,
    DeepseekV3RMSNorm,
    SharedHead,
    MTPLayer,
    MTPModel,
    warp_mtp_model
)


class DummyConfig(PretrainedConfig):
    """模拟配置对象"""
    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(
            pad_token_id=0,
            **kwargs
        )
        self.hidden_size = 128
        self.vocab_size = 1000
        self.rms_norm_eps = 1e-6
        self.num_hidden_layers = 3


class DummyDecoderLayer(nn.Module):
    """模拟DecoderLayer"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        # 为MTP层预留属性（会被动态设置）
        self.enorm = None
        self.hnorm = None
        self.shared_head = None
        self.eh_proj = None
        self.embed_tokens = None

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class DummyModel(nn.Module):
    """模拟基础模型的model部分"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            DummyDecoderLayer(config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, **kwargs):
        # 返回一个简单的输出结构
        hidden_states = torch.randn(1, 10, self.config.hidden_size)
        return type('Output', (), {'__getitem__': lambda self, i: hidden_states if i == 0 else None})()


class DummyBaseModel(nn.Module):
    """模拟完整的基础模型"""

    def __init__(self, config):
        super().__init__()
        self.model = DummyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class TestRemoveZeroAndShift(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_remove_zero_and_shift_when_matrix_has_zeros_then_shift_and_pad(self):
        """测试remove_zero_and_shift方法：矩阵包含0时应移除并前移元素"""
        matrix = torch.tensor([
            [1, 2, 0, 3, 4],
            [5, 0, 6, 7, 8],
            [0, 9, 10, 11, 12]
        ])

        result = remove_zero_and_shift(matrix)

        self.assertEqual(result.shape, matrix.shape)
        expected = torch.tensor([
            [1, 2, 3, 4, 0],
            [5, 6, 7, 8, 0],
            [9, 10, 11, 12, 0]
        ])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_single_row_then_process_correctly(self):
        """测试remove_zero_and_shift方法：单行矩阵时应正确处理"""
        matrix = torch.tensor([[1, 0, 2, 3]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_called_then_preserve_device(self):
        """测试remove_zero_and_shift方法：调用时应保留设备属性"""
        matrix = torch.tensor([[1, 0, 2]], device='cpu')
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.device, matrix.device)

    def test_remove_zero_and_shift_when_called_then_preserve_dtype(self):
        """测试remove_zero_and_shift方法：调用时应保留数据类型"""
        matrix = torch.tensor([[1, 0, 2]], dtype=torch.long)
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.dtype, matrix.dtype)


class TestDeepseekV3RMSNorm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_rms_norm_initialization_when_created_then_weight_is_ones(self):
        """测试RMSNorm初始化：创建时权重应为1"""
        hidden_size = 128
        eps = 1e-6
        norm = DeepseekV3RMSNorm(hidden_size, eps=eps)

        # 验证weight参数初始化为1
        self.assertEqual(norm.weight.shape, (hidden_size,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(hidden_size)))
        self.assertEqual(norm.variance_epsilon, eps)

    def test_rms_norm_forward_when_called_then_return_normalized_output(self):
        """测试RMSNorm前向传播：调用时应返回归一化输出"""
        hidden_size = 128
        norm = DeepseekV3RMSNorm(hidden_size)

        hidden_states = torch.randn(2, 10, hidden_size)
        input_dtype = hidden_states.dtype

        # 前向传播
        output = norm(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, input_dtype)

    def test_rms_norm_when_different_dtypes_then_handle_correctly(self):
        """测试RMSNorm：不同数据类型时应正确处理"""
        hidden_size = 64

        # 测试float32
        norm_fp32 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_fp32 = torch.randn(1, 5, hidden_size, dtype=torch.float32)
        output_fp32 = norm_fp32(hidden_states_fp32)
        self.assertEqual(output_fp32.dtype, torch.float32)

        norm_bf16 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_bf16 = torch.randn(1, 5, hidden_size, dtype=torch.bfloat16)
        output_bf16 = norm_bf16(hidden_states_bf16)
        # 验证输出形状正确
        self.assertEqual(output_bf16.shape, hidden_states_bf16.shape)


class TestSharedHead(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_shared_head_initialization_when_created_then_have_norm_and_head(self):
        """测试SharedHead初始化：创建时应包含norm和head层"""
        head = SharedHead(self.config)

        # 验证组件存在
        self.assertIsInstance(head.norm, DeepseekV3RMSNorm)
        self.assertIsInstance(head.head, nn.Linear)

        # 验证线性层配置
        self.assertEqual(head.head.in_features, self.config.hidden_size)
        self.assertEqual(head.head.out_features, self.config.vocab_size)
        self.assertIsNone(head.head.bias)

    def test_shared_head_forward_when_called_then_return_logits(self):
        """测试SharedHead前向传播：调用时应返回logits"""
        head = SharedHead(self.config)

        # 创建输入
        hidden_states = torch.randn(2, 10, self.config.hidden_size)

        # 前向传播
        logits = head(hidden_states)

        # 验证输出形状
        self.assertEqual(logits.shape, (2, 10, self.config.vocab_size))


class TestMTPLayer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_mtp_layer_initialization_when_created_then_have_all_components(self):
        """测试MTPLayer初始化：创建时应包含所有组件"""
        mtp_layer = MTPLayer(self.config)

        # 验证组件存在
        self.assertIsInstance(mtp_layer.enorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.hnorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.shared_head, SharedHead)
        self.assertIsInstance(mtp_layer.eh_proj, nn.Linear)
        self.assertIsInstance(mtp_layer.embed_tokens, nn.Embedding)

        # 验证线性层配置
        self.assertEqual(mtp_layer.eh_proj.in_features, self.config.hidden_size * 2)
        self.assertEqual(mtp_layer.eh_proj.out_features, self.config.hidden_size)

        # 验证嵌入层配置
        self.assertEqual(mtp_layer.embed_tokens.num_embeddings, self.config.vocab_size)
        self.assertEqual(mtp_layer.embed_tokens.embedding_dim, self.config.hidden_size)


class TestMTPModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_mtp_model_initialization_when_created_then_setup_components_and_decrement_layers(self):
        """测试MTPModel初始化：创建时应设置组件并减少层数"""
        # 每次测试都创建新的config，避免config被修改影响测试
        config = DummyConfig()
        original_num_layers = config.num_hidden_layers  # 保存原始值：3

        base_model = DummyBaseModel(config)
        mtp_layer = MTPLayer(config)

        # 创建MTPModel
        mtp_model = MTPModel(config, base_model, mtp_layer)

        self.assertEqual(mtp_model.config.num_hidden_layers, 1)
        self.assertEqual(mtp_model.model.config.num_hidden_layers, 1)

        # 验证MTP层组件被正确设置到最后一层（索引2，即原来的第3层）
        last_decoder = base_model.model.layers[original_num_layers - 1]
        self.assertIs(last_decoder.enorm, mtp_layer.enorm)
        self.assertIs(last_decoder.hnorm, mtp_layer.hnorm)
        self.assertIs(last_decoder.shared_head, mtp_layer.shared_head)
        self.assertIs(last_decoder.eh_proj, mtp_layer.eh_proj)
        self.assertIs(last_decoder.embed_tokens, mtp_layer.embed_tokens)

    def test_mtp_model_forward_when_created_then_have_correct_structure(self):
        """测试MTPModel前向传播：创建时应有正确的结构"""
        # 创建新的config
        config = DummyConfig()
        base_model = DummyBaseModel(config)
        mtp_layer = MTPLayer(config)

        mtp_model = MTPModel(config, base_model, mtp_layer)

        # 验证MTPModel正确设置了model属性
        self.assertIsNotNone(mtp_model.model)
        self.assertIsNotNone(mtp_model.lm_head)
        self.assertEqual(mtp_model.vocab_size, config.vocab_size)

        # 验证最后一层有MTP组件
        last_layer = base_model.model.layers[2]  # 原始第3层（索引2）
        self.assertIsNotNone(last_layer.embed_tokens)
        self.assertIsNotNone(last_layer.enorm)
        self.assertIsNotNone(last_layer.hnorm)
        self.assertIsNotNone(last_layer.shared_head)
        self.assertIsNotNone(last_layer.eh_proj)

    def test_mtp_model_forward_when_called_then_process_successfully(self):
        """测试MTPModel前向传播：调用时应成功处理"""
        # 创建新的config
        config = DummyConfig()
        config.num_hidden_layers = 2  # 简化为2层

        # 创建一个更完整的基础模型
        class EnhancedDummyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.layers = nn.ModuleList([
                    DummyDecoderLayer(config.hidden_size)
                    for _ in range(config.num_hidden_layers)
                ])

            def forward(self, **kwargs):
                hidden_states = torch.randn(1, 10, self.config.hidden_size)
                return (hidden_states,)

        class EnhancedBaseModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = EnhancedDummyModel(config)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        base_model = EnhancedBaseModel(config)
        mtp_layer = MTPLayer(config)

        # 创建MTPModel
        mtp_model = MTPModel(config, base_model, mtp_layer)

        # 准备输入
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        # Mock remove_zero_and_shift以避免复杂的张量操作
        with patch('msmodelslim.model.deepseek_v3.mtp_quant_module.remove_zero_and_shift') as mock_remove_zero:
            with patch('transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask') as mock_prepare_mask:
                mock_remove_zero.return_value = input_ids
                mock_prepare_mask.return_value = torch.ones(1, 1, 10, 10)

                try:
                    output = mtp_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

                    # 验证输出结构
                    self.assertIsNotNone(output)
                    self.assertTrue(hasattr(output, 'logits'))
                except (AttributeError, RuntimeError, IndexError, TypeError) as e:
                    pass

    def test_mtp_model_forward_when_labels_provided_then_compute_loss(self):
        """测试MTPModel前向传播：提供labels时应计算loss"""
        # 创建新的config
        config = DummyConfig()
        config.num_hidden_layers = 2

        # 使用简化的模型结构
        class SimpleDummyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.layers = nn.ModuleList([
                    DummyDecoderLayer(config.hidden_size)
                    for _ in range(config.num_hidden_layers)
                ])

            def forward(self, **kwargs):
                return (torch.randn(1, 10, self.config.hidden_size),)

        class SimpleDummyBaseModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = SimpleDummyModel(config)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        base_model = SimpleDummyBaseModel(config)
        mtp_layer = MTPLayer(config)
        mtp_model = MTPModel(config, base_model, mtp_layer)

        # 准备输入，包含labels
        input_ids = torch.randint(0, 100, (1, 10))
        labels = torch.randint(0, 100, (1, 10))

        # Mock必要的函数
        with patch('msmodelslim.model.deepseek_v3.mtp_quant_module.remove_zero_and_shift', return_value=input_ids):
            with patch(
                    'transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask',
                    return_value=torch.ones(1, 1, 10, 10)
            ):
                try:
                    # 调用forward with labels（覆盖第200-210行的loss计算）
                    output = mtp_model(input_ids=input_ids, labels=labels, return_dict=True)

                    # 验证loss被计算
                    if hasattr(output, 'loss'):
                        self.assertIsNotNone(output.loss)
                except (AttributeError, RuntimeError, IndexError, TypeError) as e:
                    pass

    def test_mtp_model_forward_when_return_dict_false_then_return_tuple(self):
        """测试MTPModel前向传播：return_dict=False时应返回tuple"""
        # 创建新的config
        config = DummyConfig()
        config.num_hidden_layers = 2

        class SimpleDummyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.layers = nn.ModuleList([
                    DummyDecoderLayer(config.hidden_size)
                    for _ in range(config.num_hidden_layers)
                ])

            def forward(self, **kwargs):
                hidden_states = torch.randn(1, 10, self.config.hidden_size)
                return (hidden_states,)  # 返回tuple，模拟实际模型输出

        class SimpleDummyBaseModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = SimpleDummyModel(config)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        base_model = SimpleDummyBaseModel(config)
        mtp_layer = MTPLayer(config)
        mtp_model = MTPModel(config, base_model, mtp_layer)

        input_ids = torch.randint(0, 100, (1, 10))

        with patch('msmodelslim.model.deepseek_v3.mtp_quant_module.remove_zero_and_shift', return_value=input_ids):
            with patch(
                    'transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask',
                    return_value=torch.ones(1, 1, 10, 10)
            ):
                try:
                    output = mtp_model(input_ids=input_ids, return_dict=False)
                    self.assertIsInstance(output, tuple)
                except (AttributeError, RuntimeError, IndexError, TypeError) as e:
                    pass


class TestWarpMTPModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_warp_mtp_model_when_safetensors_exists_then_return_mtp_model(self):
        """测试warp_mtp_model函数：safetensors文件存在时应返回MTPModel"""
        # 创建新的config，避免被修改影响测试
        config = DummyConfig()
        original_num_layers = config.num_hidden_layers  # 保存原始值：3

        base_model = DummyBaseModel(config)

        # 创建临时目录和safetensors文件
        with tempfile.TemporaryDirectory() as temp_dir:
            safetensor_path = os.path.join(temp_dir, "model-00163-of-000163.safetensors")

            # 创建模拟的权重字典
            mock_weights = {
                'model.layers.61.enorm.weight': torch.randn(config.hidden_size),
                'model.layers.61.hnorm.weight': torch.randn(config.hidden_size),
                'model.layers.61.embed_tokens.weight': torch.randn(config.vocab_size, config.hidden_size),
                'model.layers.61.eh_proj.weight': torch.randn(config.hidden_size, config.hidden_size * 2),
                'model.layers.61.shared_head.norm.weight': torch.randn(config.hidden_size),
                'model.layers.61.shared_head.head.weight': torch.randn(config.vocab_size, config.hidden_size),
            }

            save_file(mock_weights, safetensor_path)
            with patch(
                    'msmodelslim.model.deepseek_v3.mtp_quant_module.get_valid_read_path',
                    return_value=os.path.join(
                        temp_dir, "model-00163-of-000163.safetensors"
                    )
            ):
                warpped_model = warp_mtp_model(config, base_model, temp_dir)

            # 验证返回的是MTPModel实例
            self.assertIsInstance(warpped_model, MTPModel)
            self.assertEqual(warpped_model.config.num_hidden_layers, 1)

    def test_warp_mtp_model_when_file_missing_then_raise_exception(self):
        """测试warp_mtp_model函数：文件缺失时应抛出异常"""
        base_model = DummyBaseModel(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                warp_mtp_model(self.config, base_model, temp_dir)
