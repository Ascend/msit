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

import json
import os
import shutil
import tempfile
from typing import Dict, List, Union

import pytest
import torch
from safetensors.torch import safe_open

from msmodelslim.quant.ir import W8A8StaticFakeQuantLinear, W8A8DynamicPerChannelFakeQuantLinear
from .base import FakeLlamaModelAdapter, is_npu_available, invoke_test


def check_quant_model_description(tmp_dir: str, expected_quant_types: Union[str, List[str]]) -> None:
    """检查quant_model_description.json文件的基本内容"""
    quant_desc_file = os.path.join(tmp_dir, "quant_model_description.json")
    assert os.path.exists(quant_desc_file), "Config file should exist"
    
    with open(quant_desc_file, "r") as f:
        config_data = json.load(f)
    
    assert isinstance(config_data, dict), "Config data should be a dictionary"
    assert "version" in config_data, "Config data should have version field"
    assert config_data["version"] == "1.0.0", "version should be 1.0.0"
    assert "model_quant_type" in config_data, "Config data should have model_quant_type field"
    
    if isinstance(expected_quant_types, str):
        assert config_data["model_quant_type"] == expected_quant_types, \
            f"model_quant_type should be {expected_quant_types}, got {config_data['model_quant_type']}"
    else:
        assert config_data["model_quant_type"] in expected_quant_types, \
            f"model_quant_type should be one of {expected_quant_types}, got {config_data['model_quant_type']}"


def check_w8a8_static_export(module: W8A8StaticFakeQuantLinear, name: str,
                             all_tensors: Dict[str, torch.Tensor]) -> None:
    """检查W8A8StaticFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    input_scale_key = f"{name}.input_scale"
    input_offset_key = f"{name}.input_offset"
    quant_bias_key = f"{name}.quant_bias"
    deq_scale_key = f"{name}.deq_scale"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, \
        f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, \
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"

    # 验证输入缩放因子tensor必须存在
    assert input_scale_key in all_tensors, f"Input scale tensor {input_scale_key} must exist in safetensors file"
    input_scale_tensor = all_tensors[input_scale_key]
    assert input_scale_tensor.dtype == torch.float32, \
        f"Input scale tensor {input_scale_key} should be float32, got {input_scale_tensor.dtype}"
    assert input_scale_tensor.shape == (1,), \
        f"Input scale tensor {input_scale_key} shape mismatch: expected {module.input_scale.shape}, got {input_scale_tensor.shape}"

    # 验证输入偏移量tensor必须存在
    assert input_offset_key in all_tensors, f"Input offset tensor {input_offset_key} must exist in safetensors file"
    input_offset_tensor = all_tensors[input_offset_key]
    assert input_offset_tensor.dtype == torch.float32, \
        f"Input offset tensor {input_offset_key} should be float32, got {input_offset_tensor.dtype}"
    assert input_offset_tensor.shape == (1,), \
        f"Input offset tensor {input_offset_key} shape mismatch: expected {module.input_offset.shape}, got {input_offset_tensor.shape}"

    # 验证量化偏置tensor必须存在
    assert quant_bias_key in all_tensors, f"Quant bias tensor {quant_bias_key} must exist in safetensors file"
    quant_bias_tensor = all_tensors[quant_bias_key]
    assert quant_bias_tensor.dtype == torch.int32, \
        f"Quant bias tensor {quant_bias_key} should be int32, got {quant_bias_tensor.dtype}"
    assert quant_bias_tensor.shape == (module.weight.shape[0],), \
        f"Quant bias tensor {quant_bias_key} shape mismatch: expected {module.quant_bias.shape}, got {quant_bias_tensor.shape}"

    # 验证deq缩放因子tensor必须存在
    assert deq_scale_key in all_tensors, f"Deq scale tensor {deq_scale_key} must exist in safetensors file"
    deq_scale_tensor = all_tensors[deq_scale_key]
    assert deq_scale_tensor.dtype == torch.float32, \
        f"Deq scale tensor {deq_scale_key} should be float32, got {deq_scale_tensor.dtype}"
    assert deq_scale_tensor.shape == (module.weight.shape[0],), \
        f"Deq scale tensor {deq_scale_key} shape mismatch: expected {module.deq_scale.shape}, got {deq_scale_tensor.shape}"

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, \
            f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, \
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"


def check_w8a8_dynamic_export(module: W8A8DynamicPerChannelFakeQuantLinear, name: str,
                              all_tensors: Dict[str, torch.Tensor]) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, \
        f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, \
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, \
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    assert weight_scale_tensor.shape == (module.weight.shape[0], 1), \
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {module.weight_scale.shape}, got {weight_scale_tensor.shape}"

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, \
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    assert weight_offset_tensor.shape == (module.weight.shape[0], 1), \
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {module.weight_offset.shape}, got {weight_offset_tensor.shape}"

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, \
            f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, \
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_w8a8_static_per_channel_quantization(test_device, test_dtype):
    """测试W8A8 per_channel量化功能（act: per_tensor, weight: per_channel）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试（w8a8-static-per-channel.yaml使用per_tensor+per_channel）
        model_adapter = invoke_test("w8a8_static_per_channel.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 检查quant_model_description.json包含基本的正确的内容
        check_quant_model_description(tmp_dir, "W8A8")

        # 检查safetensors中包含相应模块的导出内容，dtype和shape都符合预期
        quantized_model = model_adapter.loaded_model
        safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
        assert len(safetensors_files) > 0, "No safetensors files found"

        # 将所有safetensors文件加载到一个字典中，避免频繁打开文件
        all_tensors = {}
        for safetensors_file in safetensors_files:
            file_path = os.path.join(tmp_dir, safetensors_file)
            with safe_open(file_path, framework="pt") as f:
                tensor_keys = list(f.keys())
                assert len(tensor_keys) > 0, f"No tensors found in {safetensors_file}"
                # 将所有tensor加载到字典中
                for key in tensor_keys:
                    all_tensors[key] = f.get_tensor(key)

        # 验证每个W8A8StaticFakeQuantLinear模块的导出内容
        for name, module in quantized_model.named_modules():
            if isinstance(module, W8A8StaticFakeQuantLinear):
                check_w8a8_static_export(module, name, all_tensors)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_w8a8_mixed_quantization(test_device, test_dtype):
    """测试W8A8混合量化功能（MOE模型）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行混合量化测试
        model_adapter = invoke_test("w8a8_per_channel_mix.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 检查quant_model_description.json包含基本的正确的内容
        check_quant_model_description(tmp_dir, ["W8A8", "W8A8_DYNAMIC"])

        # 检查safetensors中包含相应模块的导出内容，dtype和shape都符合预期
        quantized_model = model_adapter.loaded_model
        safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
        assert len(safetensors_files) > 0, "No safetensors files found"

        # 将所有safetensors文件加载到一个字典中，避免频繁打开文件
        all_tensors = {}
        for safetensors_file in safetensors_files:
            file_path = os.path.join(tmp_dir, safetensors_file)
            with safe_open(file_path, framework="pt") as f:
                tensor_keys = list(f.keys())
                assert len(tensor_keys) > 0, f"No tensors found in {safetensors_file}"
                # 将所有tensor加载到字典中
                for key in tensor_keys:
                    all_tensors[key] = f.get_tensor(key)

        # 验证每个量化模块的导出内容
        for name, module in quantized_model.named_modules():
            if isinstance(module, W8A8StaticFakeQuantLinear):
                check_w8a8_static_export(module, name, all_tensors)
            elif isinstance(module, W8A8DynamicPerChannelFakeQuantLinear):
                check_w8a8_dynamic_export(module, name, all_tensors)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_w8a8_dynamic_per_channel_quantization(test_device, test_dtype):
    """测试W8A8 per_token量化功能（act: per_token, weight: per_channel）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_token量化测试（w8a8-dynamic-per-channel.yaml使用per_token+per_channel）
        model_adapter = invoke_test("w8a8_dynamic_per_channel.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 检查quant_model_description.json包含基本的正确的内容
        check_quant_model_description(tmp_dir, "W8A8_DYNAMIC")

        # 检查safetensors中包含相应模块的导出内容，dtype和shape都符合预期
        quantized_model = model_adapter.loaded_model
        safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
        assert len(safetensors_files) > 0, "No safetensors files found"

        # 将所有safetensors文件加载到一个字典中，避免频繁打开文件
        all_tensors = {}
        for safetensors_file in safetensors_files:
            file_path = os.path.join(tmp_dir, safetensors_file)
            with safe_open(file_path, framework="pt") as f:
                tensor_keys = list(f.keys())
                assert len(tensor_keys) > 0, f"No tensors found in {safetensors_file}"
                # 将所有tensor加载到字典中
                for key in tensor_keys:
                    all_tensors[key] = f.get_tensor(key)

        # 验证每个W8A8DynamicPerChannelFakeQuantLinear模块的导出内容
        for name, module in quantized_model.named_modules():
            if isinstance(module, W8A8DynamicPerChannelFakeQuantLinear):
                check_w8a8_dynamic_export(module, name, all_tensors)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
