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
msmodelslim.utils.dag_utils.torch_dag_adapter 模块的单元测试
"""

import unittest
from collections import defaultdict
from unittest.mock import Mock, patch
import torch
import torch.nn as nn

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode
from msmodelslim.utils.dag_utils.dag_torch_hook import DagTorchHook
from msmodelslim.utils.dag_utils.model_infos import ModuleType
from msmodelslim.utils.dag_utils.model_structure_process import StructureProcess
from msmodelslim.utils.dag_utils.torch_dag_adapter import DagNodeInfo, TorchDAGAdapter


class TestDagNodeInfo(unittest.TestCase):
    """测试DagNodeInfo类"""

    def setUp(self):
        """测试前准备"""
        self.dag_node_info = DagNodeInfo(
            name="test_node",
            class_type="Linear",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output2"]
        )

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.dag_node_info.name, "test_node")
        self.assertEqual(self.dag_node_info.class_type, "Linear")
        self.assertEqual(self.dag_node_info.input_nodes, ["input1", "input2"])
        self.assertEqual(self.dag_node_info.output_nodes, ["output1", "output2"])

    def test_eq_true(self):
        """测试相等比较"""
        other = DagNodeInfo(
            name="test_node",
            class_type="Linear",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output2"]
        )
        self.assertEqual(self.dag_node_info, other)

    def test_eq_false_different_name(self):
        """测试名称不同"""
        other = DagNodeInfo(
            name="other_node",
            class_type="Linear",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output2"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_class_type(self):
        """测试类型不同"""
        other = DagNodeInfo(
            name="test_node",
            class_type="Conv2d",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output2"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_input_nodes(self):
        """测试输入节点不同"""
        other = DagNodeInfo(
            name="test_node",
            class_type="Linear",
            input_nodes=["input1", "input3"],
            output_nodes=["output1", "output2"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_output_nodes(self):
        """测试输出节点不同"""
        other = DagNodeInfo(
            name="test_node",
            class_type="Linear",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output3"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_none(self):
        """测试与None比较"""
        self.assertIsNotNone(self.dag_node_info)


class MockDagNode:
    def __init__(self, name, op_type, input_nodes=None, output_nodes=None, inputs=None, outputs=None):
        self.name_in_network = name
        self.op_type = op_type
        self.input_nodes = input_nodes or []
        self.output_nodes = output_nodes or []
        self.inputs = inputs or []
        self.outputs = outputs or []


class TestTorchDAGAdapter(unittest.TestCase):
    """测试TorchDAGAdapter类"""

    def setUp(self):
        """测试前准备"""
        self.model = nn.Linear(224 * 224 * 3, 10)

        self.dag_nodes = [
            MockDagNode("layer1", "Linear"),
            MockDagNode("relu", "ReLU"),
            MockDagNode("layer2", "Linear"),
            MockDagNode("add_op", "__add__"),
            MockDagNode("norm", "LayerNorm")
        ]

        self.dag_nodes[0].output_nodes = [self.dag_nodes[1]]
        self.dag_nodes[1].input_nodes = [self.dag_nodes[0]]
        self.dag_nodes[1].output_nodes = [self.dag_nodes[2]]
        self.dag_nodes[2].input_nodes = [self.dag_nodes[1]]

        self.mock_dag_hook = Mock()
        self.mock_dag_hook.dag_node_list = self.dag_nodes

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=self.mock_dag_hook):
            self.adapter = TorchDAGAdapter(
                model=self.model,
                dummy_input=torch.randn(1, 224 * 224 * 3),
                hook_nodes=[nn.LayerNorm]
            )

    def test_init_with_default_dummy_input(self):
        """测试使用默认dummy_input初始化"""
        default_model = nn.Linear(224 * 224 * 3, 10)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(default_model, dummy_input=torch.randn(1, 224 * 224 * 3))
            self.assertIsNotNone(adapter._dummy_input)
            self.assertEqual(adapter._dummy_input.shape, torch.Size([1, 224 * 224 * 3]))

    def test_init_with_tensor_dummy_input(self):
        """测试使用tensor dummy_input初始化"""
        dummy_input = torch.randn(2, 10)
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=dummy_input)
            self.assertTrue(torch.equal(adapter._dummy_input, dummy_input))

    def test_init_with_hook_nodes(self):
        """测试使用hook_nodes初始化"""
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=torch.randn(1, 10), hook_nodes=[nn.LayerNorm, nn.BatchNorm2d])
            self.assertIn("layernorm", adapter.norm_nodes)
            self.assertIn("batchnorm2d", adapter.norm_nodes)

    def test_init_with_none_hook_nodes(self):
        """测试使用None hook_nodes初始化"""
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=torch.randn(1, 10), hook_nodes=None)
            self.assertEqual(adapter.norm_nodes, [])

    def test_get_node_name_in_order(self):
        """测试获取节点名称"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            names = self.adapter.get_node_name_in_order()
            expected = [node.name_in_network for node in self.dag_nodes]
            self.assertEqual(names, expected)

    def test_get_name_type_dict_in_order(self):
        """测试获取名称类型字典"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            result = self.adapter.get_name_type_dict_in_order()
            expected = {node.name_in_network: node.op_type for node in self.dag_nodes}
            self.assertEqual(result, expected)

    def test_get_network_topology(self):
        """测试获取网络拓扑"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            topology = self.adapter.get_network_topology()
            self.assertEqual(len(topology), len(self.dag_nodes))

            first_node = topology[0]
            self.assertEqual(first_node.name, self.dag_nodes[0].name_in_network)
            self.assertEqual(first_node.class_type, self.dag_nodes[0].op_type)
            self.assertEqual(first_node.input_nodes,
                             [item.name_in_network if item else None for item in self.dag_nodes[0].input_nodes])
            self.assertEqual(first_node.output_nodes,
                             [item.name_in_network if item else None for item in self.dag_nodes[0].output_nodes])

    def test_get_mhsa_pattern_with_valid_conditions(self):
        """测试获取MHSA模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch, MockDagNode("linear1", "Linear"),
                      MockDagNode("linear2", "Linear")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process') \
                        as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试MHSA模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process') \
                        as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试MHSA模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_add_node.input_nodes = [mock_node]

        mock_other_branch = MockDagNode("other", "Linear")
        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process') \
                        as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_no_other_branch(self):
        """测试MHSA模式中满足continue条件：没有other_branch"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        mock_node.output_nodes = [mock_add_node]

        test_nodes = [mock_node, mock_add_node, MockDagNode("other_input", "Linear")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process') \
                        as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_ffn_pattern_with_valid_conditions(self):
        """测试获取FFN模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [mock_node, mock_add_node, mock_other_branch, MockDagNode("linear1", "Linear"),
                      MockDagNode("linear2", "Linear")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul',
                       return_value=True):
                ffn_pattern = self.adapter.get_ffn_pattern()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试FFN模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2, mock_other_branch]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul') \
                        as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_other_branch_not_1(self):
        """测试FFN模式中满足continue条件：other_branch数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch1 = MockDagNode("other1", "Linear")
        mock_other_branch2 = MockDagNode("other2", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch1, mock_other_branch2]

        test_nodes = [mock_node, mock_add_node, mock_other_branch1, mock_other_branch2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul') \
                        as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试FFN模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul') \
                        as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_mhsa_ln_pattern(self):
        """测试获取MHSA LN模式"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [mock_node, mock_add_node, mock_other_branch, MockDagNode("linear1", "Linear"),
                      MockDagNode("linear2", "Linear"), MockDagNode("norm", "LayerNorm")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_ln_process'):
                qkv_list, proj_list, ln_list = self.adapter.get_mhsa_ln_pattern()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)
                self.assertIsInstance(ln_list, list)

    def test_get_ffn_ln_pattern(self):
        """测试获取FFN LN模式"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [mock_node, mock_add_node, mock_other_branch, MockDagNode("linear1", "Linear"),
                      MockDagNode("linear2", "Linear"), MockDagNode("norm", "LayerNorm")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'):
                ffn_pattern, ffn_ln_list = self.adapter.get_ffn_ln_pattern()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_mhsa_ln_pattern(self):
        """测试获取LLAMA MHSA LN模式"""
        with patch.object(self.adapter, 'get_mhsa_ln_pattern') as mock_get:
            self.adapter.get_llama_mhsa_ln_pattern()
            mock_get.assert_called_once_with(ln_type='Llamarmsnormbias')

    def test_get_llama_ffn_ln_pattern_with_valid_conditions(self):
        """测试获取LLAMA FFN LN模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [mock_node, mock_add_node, mock_other_branch, MockDagNode("linear1", "Linear"),
                      MockDagNode("linear2", "Linear"), MockDagNode("norm", "LayerNorm")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'):
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试LLAMA FFN LN模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2, mock_other_branch]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list') \
                        as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_other_branch_not_1(self):
        """测试LLAMA FFN LN模式中满足continue条件：other_branch数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch1 = MockDagNode("other1", "Linear")
        mock_other_branch2 = MockDagNode("other2", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch1, mock_other_branch2]

        test_nodes = [mock_node, mock_add_node, mock_other_branch1, mock_other_branch2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list') \
                        as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试LLAMA FFN LN模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                    'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list') \
                        as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_norm_linear_subgraph(self):
        """测试获取norm-linear子图"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                result = self.adapter.get_norm_linear_subgraph()
                self.assertIsInstance(result, defaultdict)
                self.assertIn(norm_node.name_in_network, result)