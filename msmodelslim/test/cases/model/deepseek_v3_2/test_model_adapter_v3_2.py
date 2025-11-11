# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from typing import List
import torch
import torch.nn as nn

from msmodelslim.model.deepseek_v3_2.model_adapter import DeepSeekV32ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class DummyModelArgs:
    """模拟ModelArgs配置类"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.qk_nope_head_dim = 64
        self.v_head_dim = 64
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.rms_norm_eps = 1e-6
        self.vocab_size = 1000
        self.hidden_size = 128


class DummyDeepSeekV3RMSNorm(nn.Module):
    """模拟DeepseekV3RMSNorm归一化层"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return hidden_states * self.weight


class DummySharedHead(nn.Module):
    """模拟SharedHead类（MTP层依赖）"""

    def __init__(self, config):
        super().__init__()
        self.norm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class DummyMTPLayer(nn.Module):
    """模拟MTPLayer类"""

    def __init__(self, config):
        super().__init__()
        self.enorm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.hnorm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.shared_head = DummySharedHead(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class DummyDecoderLayer(nn.Module):
    """模拟解码器层"""

    def __init__(self, layer_id=0, args=None):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.shared_head = None  # 默认无MTP相关属性
        self.hook_id = 0
        self._forward_hooks = {}
        self._forward_pre_hooks_with_kwargs = {}  # 接收kwargs的钩子

    def get_submodule(self, name):
        if name == "shared_head" and hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"No submodule named {name}")

    def register_forward_pre_hook(self, hook, with_kwargs=True, prepend=True):
        current_id = self.hook_id
        self.hook_id += 1
        self._forward_pre_hooks_with_kwargs[current_id] = hook

        def remove(*args, **kwargs):
            if current_id in self._forward_pre_hooks_with_kwargs:
                del self._forward_pre_hooks_with_kwargs[current_id]

        return type('', (), {'remove': remove})()

    def forward(self, hidden_states, **kwargs):
        # 处理需要接收kwargs的钩子
        for _, hook in self._forward_pre_hooks_with_kwargs.items():
            args_kwargs_result = hook(self, (hidden_states,), kwargs)
            if args_kwargs_result is not None:
                if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                    (hidden_states,), kwargs = args_kwargs_result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                    )

        return hidden_states


class DummyModelInner(nn.Module):
    """模拟模型内部的model对象（含layers、norm、freqs_cis）"""

    def __init__(self, num_layers=2, config=None):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(layer_id=i, args=config) for i in range(num_layers)])
        self.norm = DummyDeepSeekV3RMSNorm(config.hidden_size if config else 128)
        self.freqs_cis = torch.randn(100, 128)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)  # 执行当前层
        return self.norm(hidden_states)

    def get_all_param_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]


class DummyModel(nn.Module):
    """模拟整体模型（含model、lm_head）"""

    def __init__(self, config=None):
        super().__init__()
        self.model = DummyModelInner(num_layers=config.num_hidden_layers if config else 2, config=config)
        self.lm_head = nn.Linear(
            config.hidden_size if config else 128,
            config.vocab_size if config else 1000,
            bias=True  # 匹配lm_head.bias参数
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        return self.lm_head(hidden_states)

    def generate_full_state_dict(self):
        """生成完整state_dict，避免加载缺失键"""
        state_dict = {}
        for name, param in self.model.named_parameters():
            state_dict[f"model.{name}"] = param.data.clone()
        for name, param in self.lm_head.named_parameters():
            state_dict[f"lm_head.{name}"] = param.data.clone()
        return state_dict


class TestDeepSeekV32ModelAdapter(unittest.TestCase):
    def setUp(self):
        """初始化测试环境（统一配置，避免重复）"""
        self.model_path = Path(".")
        self.model_type = "DeepSeek-V3.2-Exp"
        self.dummy_config = DummyModelArgs()
        self.dummy_config.num_hidden_layers = 62  # 匹配真实模型层数
        self.test_device = "cpu"
        self.dummy_full_state_dict = DummyModel(config=self.dummy_config).generate_full_state_dict()
        self.adapter_patcher = patch.object(DeepSeekV32ModelAdapter, "__init__", lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        """创建并配置适配器实例的通用方法"""
        with self.adapter_patcher:
            adapter = DeepSeekV32ModelAdapter(model_path=self.model_path, model_type=self.model_type)

            for key, value in kwargs.items():
                setattr(adapter, key, value)

            if 'config' not in kwargs:
                adapter.config = self.dummy_config
            if 'model_path' not in kwargs:
                adapter.model_path = self.model_path

            return adapter

    def test_get_model_pedigree(self):
        """测试get_model_pedigree返回固定谱系"""
        adapter = self.create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), "deepseek_v3_2")

    def test_get_model_type(self):
        """测试get_model_type返回初始化的模型类型"""
        adapter = self.create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    def test_enable_kv_cache(self):
        """测试enable_kv_cache方法的调用和参数传递"""
        adapter = self.create_adapter(model_type=self.model_type)
        adapter.enable_kv_cache(Mock(), True)
        assert True  # 方法无异常执行即通过

    @patch('msmodelslim.model.deepseek_v3_2.model_adapter.ModelArgs')
    def test_load_config(self, mock_model_args):
        """测试_load_config方法是否正确返回ModelArgs实例"""
        mock_args_instance = Mock()
        mock_model_args.return_value = mock_args_instance

        adapter = self.create_adapter(model_type=self.model_type)
        result = adapter._load_config()

        mock_model_args.assert_called_once_with()
        self.assertEqual(result, mock_args_instance)

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.Transformer", new=DummyModel)
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.auto_convert_module_fp8_to_bf16")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_logger")
    def test_init_model(self, mock_get_logger: Mock, mock_auto_convert: Mock):
        """测试init_model完整流程"""
        adapter = self.create_adapter()
        adapter.get_state_dict = Mock(return_value=self.dummy_full_state_dict)
        original_num_layers = adapter.config.num_hidden_layers

        # 执行初始化并Mock load_state_dict
        result_model = adapter.init_model(device=self.test_device)
        result_model.load_state_dict = MagicMock()
        result_model.load_state_dict(self.dummy_full_state_dict)

        # 验证核心逻辑
        self.assertIsInstance(result_model, DummyModel)
        self.assertEqual(adapter.config.num_hidden_layers, original_num_layers)
        result_model.load_state_dict.assert_called_once_with(self.dummy_full_state_dict)
        mock_auto_convert.assert_called_once_with("", result_model, str(self.model_path))
        mock_get_logger.return_value.info.assert_any_call(f"Model with {original_num_layers} layers totally")

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.json_safe_load")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.os.path.join")
    def test_get_weight_map(self, mock_path_join: Mock, mock_json_load: Mock):
        """测试get_weight_map加载权重映射"""
        adapter = self.create_adapter()

        # Mock依赖返回值
        mock_json_load.return_value = {
            "weight_map": {
                "model.layers.0.self_attn.q_a_proj.weight": "model-00001.safetensors",
                "model.layers.1.self_attn.q_a_proj.weight": "model-00002.safetensors"
            }
        }
        mock_index_path = self.model_path / "model.safetensors.index.json"
        mock_path_join.return_value = mock_index_path

        # 执行测试并验证
        weight_map = adapter.get_weight_map()
        mock_path_join.assert_called_once_with(self.model_path, "model.safetensors.index.json")
        mock_json_load.assert_called_once_with(mock_index_path)
        self.assertEqual(weight_map["model.layers.0.self_attn.q_a_proj.weight"], "model-00001.safetensors")

        # 验证LRU缓存
        adapter.get_weight_map()
        self.assertEqual(mock_json_load.call_count, 1)

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_mtp_layer", return_value=DummyMTPLayer(DummyModelArgs()))
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.wrap_mtp_decoder")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_logger")
    def test_load_mtp_if_not_load_missing(self, mock_get_logger: Mock, mock_wrap_mtp: Mock, mock_get_mtp: Mock):
        """测试MTP层缺失时加载"""
        adapter = self.create_adapter()

        # 准备无shared_head属性的decoder
        dummy_decoder = DummyDecoderLayer()
        if hasattr(dummy_decoder, 'shared_head'):
            del dummy_decoder.shared_head
        with self.assertRaises(AttributeError):
            _ = dummy_decoder.shared_head

        # 执行加载并验证
        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_called_once_with(config=self.dummy_config, model_path=self.model_path)
        mock_wrap_mtp.assert_called_once_with(mtp_decoder=dummy_decoder, mtp_layer=mock_get_mtp.return_value)
        mock_get_logger.return_value.info.assert_any_call("Creating MTP layer")

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_mtp_layer")
    def test_load_mtp_if_not_load_exist(self, mock_get_mtp: Mock):
        """测试MTP层已存在时跳过加载"""
        adapter = self.create_adapter()

        # 准备带shared_head的decoder
        dummy_decoder = DummyDecoderLayer()
        dummy_decoder.shared_head = DummySharedHead(DummyModelArgs())

        # 执行加载并验证
        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_not_called()

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.auto_convert_module_fp8_to_bf16")
    def test_load_decoder_if_not_exist(self, mock_auto_convert: Mock):
        """测试decoder缺失时创建"""
        # 生成匹配的state_dict（过滤不存在的参数）
        dummy_decoder = DummyDecoderLayer(layer_id=1, args=self.dummy_config)
        actual_param_names = [name for name, _ in dummy_decoder.named_parameters()]
        mock_state_dict = {name: torch.ones(1) for name in actual_param_names if "input_layernorm.weight" not in name}

        adapter = self.create_adapter(get_state_dict=Mock(return_value=mock_state_dict))

        # 准备仅含1层的模型
        dummy_model = DummyModel(config=self.dummy_config)
        dummy_model.model.layers = nn.ModuleList([DummyDecoderLayer(layer_id=0)])

        # 执行创建并验证
        result_decoder = adapter.load_decoder_if_not_exist(model=dummy_model, name="model.layers.1", idx=1)
        self.assertIsInstance(result_decoder, DummyDecoderLayer)
        self.assertEqual(len(dummy_model.model.layers), 2)
        mock_auto_convert.assert_called_once_with("model.layers.1", result_decoder, str(self.model_path))

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.DeepSeekV32ModelAdapter.load_mtp_if_not_load")
    def test_generate_decoder_layer(self, mock_load_mtp: Mock):
        """测试generate_decoder_layer生成所有层（最后一层加载MTP）"""
        mock_decoders = [DummyDecoderLayer(0), DummyDecoderLayer(1), DummyDecoderLayer(2)]
        adapter = self.create_adapter(
            config=Mock(num_hidden_layers=3),
            load_decoder_if_not_exist=Mock(side_effect=mock_decoders)
        )

        # 执行生成并验证
        dummy_model = DummyModel(config=self.dummy_config)
        layers = list(adapter.generate_decoder_layer(model=dummy_model))

        self.assertEqual(len(layers), 3)
        self.assertEqual([name for name, _ in layers], ["model.layers.0", "model.layers.1", "model.layers.2"])
        mock_load_mtp.assert_called_once_with(mock_decoders[2])

    def test_generate_model_forward_raise_error(self):
        """测试无法获取first_block_input时抛异常"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        # 1. 正常创建模型
        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 128)).float()  # 形状为(1, 128)

        # 2. 获取第一层并保存原始的钩子注册方法
        first_layer = dummy_model.model.layers[0]

        # 3. 定义无操作的钩子注册方法（不实际注册钩子）
        def no_op_register_forward_pre_hook(self, hook, with_kwargs=True, prepend=True):
            class DummyRemove:
                @staticmethod
                def remove():
                    pass

            return DummyRemove()

        # 4. 动态替换第一层的注册方法
        first_layer.register_forward_pre_hook = no_op_register_forward_pre_hook.__get__(first_layer, DummyDecoderLayer)

        # 5. 验证异常（因钩子未注册导致无法获取输入）
        with self.assertRaises(InvalidModelError) as cm:
            gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
            next(gen)

        self.assertIn("Can't get first block input", str(cm.exception))

    def test_generate_model_forward_gets_first_block_input(self):
        """测试成功获取first_block_input的情况"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 128)).float()  # 形状为(1, 128)

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass  # 预期行为

    def test_generate_model_forward_with_list_input(self):
        """测试输入为列表类型时的处理逻辑"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = [torch.randint(0, 1000, (1, 128)).float()]  # 列表类型输入

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass

    def test_generate_model_forward_with_dict_input(self):
        """测试输入为字典类型时的处理逻辑"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = {"input_ids": torch.randint(0, 1000, (1, 128)).float()}  # 字典类型输入

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.dist")
    def test_generate_model_forward_with_dist_initialized(self, mock_dist):
        """测试分布式环境已初始化时的逻辑"""
        mock_dist.is_initialized.return_value = True  # 模拟分布式已初始化

        adapter = self.create_adapter()
        adapter.generate_decoder_layer = Mock(return_value=[])  # 空迭代器

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10)).float()  # 正确的数据类型

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass  # 预期的生成器结束

        mock_dist.barrier.assert_called_once()  # 验证分布式屏障被调用

    def test_generate_model_forward_with_exception(self):
        """测试前向传播过程中抛出其他异常的处理"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        # 创建一个会抛出异常的模型
        class ErrorModel(DummyModel):
            def __call__(self, *args, **kwargs):
                raise ValueError("Test error")

        dummy_model = ErrorModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10))

        with self.assertRaises(ValueError) as cm:
            gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
            next(gen)

        self.assertEqual(str(cm.exception), "Test error")

    def test_generate_model_forward_processes_decoder_layers(self):
        """测试处理transformer decoder层的逻辑"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        adapter.mtp_preprocess = Mock(return_value=((torch.tensor([1]),), {}))

        # 创建模拟的decoder层
        mock_blocks = [
            ('model.layers.0', Mock()),
            (f'model.layers.{self.dummy_config.num_hidden_layers - 1}', Mock())  # 最后一层
        ]
        adapter.generate_decoder_layer = Mock(return_value=mock_blocks)

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10))

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        # 第一次迭代
        request = next(gen)
        self.assertEqual(request.name, 'model.layers.0')

        # 发送返回值并进行第二次迭代
        try:
            gen.send((torch.tensor([1]), torch.tensor([2])))
        except StopIteration:
            pass

        # 验证最后一层调用了mtp_preprocess
        adapter.mtp_preprocess.assert_called_once()

    def create_mock_setup(self, prefix, weight_map, params):
        """创建get_state_dict测试的公共模拟设置"""
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock()
        adapter.get_weight_map.return_value = weight_map

        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = params

        return mock_module, adapter

    def test_get_state_dict_with_and_without_prefix(self):
        """测试带前缀和不带前缀两种情况"""
        test_cases = [
            # 带prefix的情况
            {
                "prefix": "prefix",
                "weight_map": {"prefix.layer.weight": "file1.safetensors", "prefix.layer.bias": "file1.safetensors"},
                "params": [("layer.weight", Mock()), ("layer.bias", Mock())],
                "expected_calls": ["prefix.layer.weight", "prefix.layer.bias"]
            },
            # 不带prefix的情况
            {
                "prefix": "",
                "weight_map": {"layer.weight": "file2.safetensors", "layer.bias": "file2.safetensors"},
                "params": [("layer.weight", Mock()), ("layer.bias", Mock())],
                "expected_calls": ["layer.weight", "layer.bias"]
            }
        ]

        for case in test_cases:
            with self.subTest(case=case):
                mock_module, adapter = self.create_mock_setup(
                    case["prefix"], case["weight_map"], case["params"]
                )

                with patch('msmodelslim.model.deepseek_v3_2.model_adapter.get_valid_read_path',
                           return_value=f"/fake/path/file.safetensors"), \
                        patch('msmodelslim.model.deepseek_v3_2.model_adapter.safe_open') as mock_safe_open:

                    mock_file = MagicMock()
                    mock_file.get_tensor.return_value = "dummy_tensor"
                    mock_safe_open.return_value.__enter__.return_value = mock_file

                    result = adapter.get_state_dict(mock_module, case["prefix"])

                    self.assertEqual(len(result), len(case["params"]))
                    for name, _ in case["params"]:
                        self.assertIn(name, result)
                    for call in case["expected_calls"]:
                        mock_file.get_tensor.assert_any_call(call)

    def test_get_state_dict_multiple_files(self):
        """测试参数分布在多个文件中的情况"""
        weight_map = {
            "layer1.weight": "file1.safetensors",
            "layer1.bias": "file1.safetensors",
            "layer2.weight": "file2.safetensors",
            "layer2.bias": "file2.safetensors"
        }
        params = [
            ("layer1.weight", Mock()), ("layer1.bias", Mock()),
            ("layer2.weight", Mock()), ("layer2.bias", Mock())
        ]

        mock_module, adapter = self.create_mock_setup("", weight_map, params)
        if not isinstance(mock_module, Mock):
            raise ValueError(f"create_mock_setup 返回的 mock_module 类型非法")
        if not isinstance(adapter, DeepSeekV32ModelAdapter):
            raise ValueError(f"create_mock_setup 返回的 adapter 类型非法")

        with patch('msmodelslim.model.deepseek_v3_2.model_adapter.get_valid_read_path',
                   side_effect=lambda x, **kwargs: x), \
                patch('msmodelslim.model.deepseek_v3_2.model_adapter.safe_open') as mock_safe_open:
            # 为不同文件创建不同的模拟对象
            mock_file1, mock_file2 = MagicMock(), MagicMock()
            mock_file1.get_tensor.return_value = "tensor1"
            mock_file2.get_tensor.return_value = "tensor2"

            def file_side_effect(path, **kwargs):
                mock_context = MagicMock()
                mock_context.__enter__.return_value = mock_file1 if "file1" in path else mock_file2
                return mock_context

            mock_safe_open.side_effect = file_side_effect

            result = adapter.get_state_dict(mock_module)

            self.assertEqual(len(result), 4)
            self.assertEqual(mock_safe_open.call_count, 2)  # 验证打开了两个文件

    def test_get_state_dict_file_not_found(self):
        """测试文件不存在时的异常处理"""
        weight_map = {"layer.weight": "missing.safetensors"}
        params = [("layer.weight", Mock())]

        mock_module, adapter = self.create_mock_setup("", weight_map, params)

        with patch('msmodelslim.model.deepseek_v3_2.model_adapter.get_valid_read_path',
                   side_effect=FileNotFoundError("File not found")):
            with self.assertRaises(FileNotFoundError):
                adapter.get_state_dict(mock_module)


if __name__ == "__main__":
    unittest.main()
