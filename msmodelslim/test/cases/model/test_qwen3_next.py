# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.app import DeviceType
from msmodelslim.model.qwen3_next import Qwen3NextModelAdapter
from msmodelslim.core.base.protocol import ProcessRequest


class TestQwen3NextModelAdapter(unittest.TestCase):

    def setUp(self):
        self.model_type = 'Qwen3-Next-80B-A3B-Instruct'
        self.model_path = Path('.')

    def test_get_model_type(self):
        """测试get_model_type方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.model_type = self.model_type
            
            result = adapter.get_model_type()
            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree(self):
        """测试get_model_pedigree方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            result = adapter.get_model_pedigree()
            self.assertEqual(result, 'qwen3_next')

    def test_load_model(self):
        """测试load_model方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)
            
            result = adapter.load_model(device=DeviceType.NPU)
            
            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_handle_dataset(self):
        """测试handle_dataset方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)
            
            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)
            
            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_by_batch(self):
        """测试handle_dataset_by_batch方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
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

    def test_init_model(self):
        """测试init_model方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)
            
            result = adapter.init_model(device=DeviceType.NPU)
            
            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_generate_model_visit(self):
        """测试generate_model_visit方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            mock_transformer_blocks = [('block1', nn.Linear(5, 5)), ('block2', nn.Linear(5, 5))]
            
            with patch('msmodelslim.model.qwen3_next.generated_decoder_layer_visit_func') as mock_visit_func:
                mock_visit_func.return_value = iter([MagicMock(spec=ProcessRequest)])
                
                result = list(adapter.generate_model_visit( \
                    model=mock_model, transformer_blocks=mock_transformer_blocks))
                
                mock_visit_func.assert_called_once_with(mock_model, mock_transformer_blocks)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertIsInstance(result[0], ProcessRequest)

    def test_generate_model_visit_with_none_transformer_blocks(self):
        """测试generate_model_visit方法当transformer_blocks为None时"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            
            with patch('msmodelslim.model.qwen3_next.generated_decoder_layer_visit_func') as mock_visit_func:
                mock_visit_func.return_value = iter([MagicMock(spec=ProcessRequest)])
                
                result = list(adapter.generate_model_visit(model=mock_model))
                
                mock_visit_func.assert_called_once_with(mock_model, None)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)

    def test_generate_model_forward(self):
        """测试generate_model_forward方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            mock_inputs = {'input_ids': [1, 2, 3]}
            
            with patch('msmodelslim.model.qwen3_next.transformers_generated_forward_func') as mock_forward_func:
                mock_forward_func.return_value = iter([MagicMock(spec=ProcessRequest)])
                
                result = list(adapter.generate_model_forward(model=mock_model, inputs=mock_inputs))
                
                mock_forward_func.assert_called_once_with(mock_model, mock_inputs)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertIsInstance(result[0], ProcessRequest)

    def test_enable_kv_cache(self):
        """测试enable_kv_cache方法"""
        with patch('msmodelslim.model.qwen3_next.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            
            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)
            
            result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)
            
            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)
            self.assertIsNone(result)