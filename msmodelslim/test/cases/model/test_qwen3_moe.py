# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.app import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.qwen3_moe import Qwen3MoeModelAdapter


class DummyConfig:
    """模拟配置对象"""
    def __init__(self):
        self.num_hidden_layers = 3


class TestQwen3MoeModelAdapter(unittest.TestCase):
    """测试Qwen3MoeModelAdapter的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'Qwen3-30B'
        self.model_path = Path('.')
        self.trust_remote_code = False

    def test_get_model_type_when_initialized_then_return_model_type(self):
        """测试get_model_type方法：初始化后应返回正确的模型类型"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code
            )
            adapter.model_type = self.model_type
            
            result = adapter.get_model_type()
            
            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree_when_called_then_return_qwen3_moe(self):
        """测试get_model_pedigree方法：应返回'qwen3_moe'"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            result = adapter.get_model_pedigree()
            
            self.assertEqual(result, 'qwen3_moe')

    def test_load_model_when_called_then_delegate_to_load_model(self):
        """测试load_model方法：应委托给_load_model方法"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)
            
            result = adapter.load_model(device=DeviceType.NPU)
            
            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_handle_dataset_when_called_then_return_tokenized_data(self):
        """测试handle_dataset方法：应返回tokenized数据"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)
            
            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)
            
            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_by_batch_when_called_then_return_batch_tokenized_data(self):
        """测试handle_dataset_by_batch方法：应返回批量tokenized数据"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_batch_dataset = [['batch1'], ['batch2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)
            
            result = adapter.handle_dataset_by_batch(
                dataset='test_data',
                batch_size=2,
                device=DeviceType.CPU
            )
            
            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list='test_data',
                batch_size=2,
                device=DeviceType.CPU
            )

    def test_init_model_when_called_then_delegate_to_load_model(self):
        """测试init_model方法：应委托给_load_model方法"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)
            
            result = adapter.init_model(device=DeviceType.NPU)
            
            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache_when_called_then_delegate_to_enable_kv_cache(self):
        """测试enable_kv_cache方法：应委托给_enable_kv_cache方法"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)
            
            result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)
            
            # 验证_enable_kv_cache被调用
            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_with_false_then_disable_cache(self):
        """测试enable_kv_cache方法：传入False时应禁用缓存"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)
            
            adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)
            
            # 验证参数正确传递
            adapter._enable_kv_cache.assert_called_once_with(mock_model, False)

    def test_get_adapter_config_for_subgraph_when_called_then_return_adapter_configs(self):
        """测试get_adapter_config_for_subgraph方法：应返回适配器配置列表"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.config = DummyConfig()
            
            result = adapter.get_adapter_config_for_subgraph()
            
            # 验证返回列表
            self.assertIsInstance(result, list)
            
            # 每层应该有2个配置（Norm-Linear融合 + OV融合）
            expected_configs = adapter.config.num_hidden_layers * 2
            self.assertEqual(len(result), expected_configs)
            
            # 验证第一个配置的类型
            self.assertIsInstance(result[0], AdapterConfig)

    def test_get_adapter_config_for_subgraph_when_zero_layers_then_return_empty_list(self):
        """测试get_adapter_config_for_subgraph方法：0层时应返回空列表"""
        with patch('msmodelslim.model.qwen3_moe.TransformersModel.__init__', return_value=None):
            adapter = Qwen3MoeModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 0
            
            result = adapter.get_adapter_config_for_subgraph()
            
            # 验证返回空列表
            self.assertEqual(len(result), 0)
            self.assertIsInstance(result, list)
