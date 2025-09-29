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
import os
import shutil
import tempfile

import pytest
import torch

from msmodelslim.quant.ir import W4A4DynamicPerGroupFakeQuantLinear, W4A4DynamicPerChannelFakeQuantLinear
from msmodelslim.quant.ir import W8A8DynamicPerChannelFakeQuantLinear
from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available
from .utils import run_fake_quantization_test, check_w4a4_dynamic_per_group_export, \
    check_w4a4_dynamic_per_channel_export, check_w8a8_dynamic_per_channel_export


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_w4a4_dynamic_per_group_quantization(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_group量化测试
        model_adapter = invoke_test("w4a4_dynamic_per_group.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A4DynamicPerGroupFakeQuantLinear: check_w4a4_dynamic_per_group_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
            module_checkers=module_checkers,
            group_size=32
        )

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
def test_w4a4_dynamic_per_channel_quantization(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试
        model_adapter = invoke_test("w4a4_dynamic_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A4DynamicPerChannelFakeQuantLinear: check_w4a4_dynamic_per_channel_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
            module_checkers=module_checkers
        )

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
def test_w4a4_laos_pipeline(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试（w8a8-static-per-channel.yaml使用per_tensor+per_channel）
        model_adapter = invoke_test("w4a4_laos.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {
            W4A4DynamicPerGroupFakeQuantLinear: check_w4a4_dynamic_per_group_export,
            W8A8DynamicPerChannelFakeQuantLinear: check_w8a8_dynamic_per_channel_export,
        }
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
            module_checkers=module_checkers,
            group_size=32,
        )

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
