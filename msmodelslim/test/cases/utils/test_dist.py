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


"""
msmodelslim.utils.dist 模块的单元测试
"""

import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from msmodelslim.utils.dist import DistHelper


class TestDistHelper(unittest.TestCase):

    def setUp(self):

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

        self.test_model = TestModel()
        self.test_model.child_module = nn.Linear(5, 2)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_mocked_distributed(self, mock_all_gather_object, mock_get_world_size):
        """测试初始化方法"""
        mock_get_world_size.return_value = 2
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertEqual(helper._model, self.test_model)
        expected_local_modules = {
            '', 'linear1', 'relu', 'dropout', 'child_module'
        }
        self.assertEqual(helper._local_modules, expected_local_modules)
        self.assertEqual(helper._shared_modules, expected_local_modules)
        self.assertEqual(helper._all_modules, expected_local_modules)
        self.assertEqual(helper._local_only_modules, set())

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_local_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试本地模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        local_modules = list(helper.local_modules())

        self.assertEqual(len(local_modules), 5)
        self.assertIn(self.test_model, local_modules)
        self.assertIn(self.test_model.linear1, local_modules)
        self.assertIn(self.test_model.relu, local_modules)
        self.assertIn(self.test_model.dropout, local_modules)
        self.assertIn(self.test_model.child_module, local_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_shared_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试共享模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        shared_modules = list(helper.shared_modules())

        self.assertEqual(len(shared_modules), 5)
        self.assertIn(self.test_model, shared_modules)
        self.assertIn(self.test_model.linear1, shared_modules)
        self.assertIn(self.test_model.relu, shared_modules)
        self.assertIn(self.test_model.dropout, shared_modules)
        self.assertIn(self.test_model.child_module, shared_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_all_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试所有模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        all_modules = list(helper.all_modules())

        self.assertEqual(len(all_modules), 5)
        self.assertIn(self.test_model, all_modules)
        self.assertIn(self.test_model.linear1, all_modules)
        self.assertIn(self.test_model.relu, all_modules)
        self.assertIn(self.test_model.dropout, all_modules)
        self.assertIn(self.test_model.child_module, all_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_local_only_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试仅本地模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        local_only_modules = list(helper.local_only_modules())

        self.assertEqual(len(local_only_modules), 0)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_local_only_modules_generator_with_different_modules(self, mock_get_rank, mock_all_gather_object,
                                                                 mock_get_world_size):
        """测试仅本地模块生成器在不同模块配置下的行为"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = {'', 'module_a', 'module_b'}
            gathered_modules[1] = {'', 'module_a', 'module_c'}
            return None

        mock_all_gather_object.side_effect = side_effect

        class LocalOnlyTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_a = nn.Linear(10, 5)
                self.module_b = nn.Linear(5, 2)

        local_only_model = LocalOnlyTestModel()
        helper = DistHelper(local_only_model)

        local_only_modules = list(helper.local_only_modules())

        self.assertEqual(len(local_only_modules), 1)
        self.assertIn(local_only_model.module_b, local_only_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_local_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_local方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_local('linear1'))
        self.assertTrue(helper.is_local(''))

        self.assertFalse(helper.is_local('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_local_only_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_local_only方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertFalse(helper.is_local_only('linear1'))
        self.assertFalse(helper.is_local_only(''))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_shared_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_shared方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_shared('linear1'))
        self.assertTrue(helper.is_shared(''))

        self.assertFalse(helper.is_shared('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_all_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_all方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_all('linear1'))
        self.assertTrue(helper.is_all(''))

        self.assertFalse(helper.is_all('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_shared_modules_slice(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试get_shared_modules_slice方法"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        result = helper.get_shared_modules_slice()
        expected = sorted(['', 'child_module', 'dropout', 'linear1', 'relu'])[0::2]
        self.assertEqual(result, expected)

        result_with_prefix = helper.get_shared_modules_slice(prefix="model")
        expected_with_prefix = sorted([
            f"model.{name}" for name in ['', 'child_module', 'dropout', 'linear1', 'relu']
        ])[0::2]
        self.assertEqual(result_with_prefix, expected_with_prefix)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_shared_modules_slice_different_rank(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试不同rank下的get_shared_modules_slice方法"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        result = helper.get_shared_modules_slice()
        expected = sorted(['', 'child_module', 'dropout', 'linear1', 'relu'])[1::2]
        self.assertEqual(result, expected)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_rank_method(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试get_rank方法"""
        mock_get_world_size.return_value = 1
        mock_get_rank.return_value = 42
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertEqual(helper.get_rank(), 42)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather')
    def test_gather_variable_shapes_static_method(self, mock_all_gather, mock_get_world_size):
        """测试gather_variable_shapes静态方法"""
        mock_get_world_size.return_value = 2
        mock_all_gather.return_value = None

        local_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

        def all_gather_side_effect(tensor_list, tensor):
            tensor_list[:] = [x.clone() for x in [tensor] * len(tensor_list)]
            return None

        mock_all_gather.side_effect = all_gather_side_effect

        result = DistHelper.gather_variable_shapes(local_tensor)

        self.assertEqual(len(result), 2)
        for tensor in result:
            self.assertTrue(torch.equal(tensor, local_tensor))
            self.assertEqual(tensor.dtype, local_tensor.dtype)
            self.assertEqual(tensor.shape, local_tensor.shape)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather')
    def test_gather_variable_shapes_with_different_shapes(self, mock_all_gather, mock_get_world_size):
        """测试gather_variable_shapes静态方法处理不同形状的张量"""
        mock_get_world_size.return_value = 2
        mock_all_gather.return_value = None

        local_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

        def all_gather_side_effect(tensor_list, tensor):
            tensor_list[:] = [x.clone() for x in [tensor] * len(tensor_list)]
            return None

        mock_all_gather.side_effect = all_gather_side_effect

        result = DistHelper.gather_variable_shapes(local_tensor)

        self.assertEqual(len(result), 2)
        for tensor in result:
            self.assertTrue(torch.equal(tensor, local_tensor))
            self.assertEqual(tensor.dtype, local_tensor.dtype)
            self.assertEqual(tensor.shape, local_tensor.shape)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_empty_model(self, mock_all_gather_object, mock_get_world_size):
        """测试使用空模型初始化"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        empty_model = nn.Module()
        helper = DistHelper(empty_model)

        self.assertEqual(helper._local_modules, {''})
        self.assertEqual(helper._shared_modules, {''})
        self.assertEqual(helper._all_modules, {''})
        self.assertEqual(helper._local_only_modules, set())

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_nested_modules(self, mock_all_gather_object, mock_get_world_size):
        """测试带有嵌套模块的模型"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            return None

        mock_all_gather_object.side_effect = side_effect

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU()
                )
                self.layer2 = nn.Linear(5, 1)

        nested_model = NestedModel()
        helper = DistHelper(nested_model)

        expected_modules = {
            '', 'layer1', 'layer1.0', 'layer1.1', 'layer2'
        }
        self.assertEqual(helper._local_modules, expected_modules)


if __name__ == '__main__':
    unittest.main()