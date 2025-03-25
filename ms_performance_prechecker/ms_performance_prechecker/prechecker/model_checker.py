# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import hashlib
from concurrent.futures import ThreadPoolExecutor

from ms_performance_prechecker.prechecker.register import RrecheckerBase, show_check_result, CheckResult
from ms_performance_prechecker.prechecker.utils import logger, get_model_path_from_mindie_config
from ms_performance_prechecker.prechecker.utils import SimpleProgressBar, is_deepseek_model
from ms_performance_prechecker.prechecker.utils import SimpleProgressBar, is_deepseek_model, get_next_dict_item

DEEPSEEK_R1_FP8_WEIGHT_SIZE = 674720176952
DEEPSEEK_R1_FP16_WEIGHT_SIZE = 1368985513488


def get_file_sizes(file_path_regex):
    files = glob.glob(file_path_regex)
    result_dict = {}
    for file_path in files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        result_dict[file_name] = {"size": file_size}
    return result_dict


def get_file_sha256s(file_path_regex):
    files = glob.glob(file_path_regex)
    result_dict = {}
    for file_path in SimpleProgressBar(files, desc="Calculating sha256sum", total=len(files)):
        file_name = os.path.basename(file_path)
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as ff:
            for byte_block in iter(lambda: ff.read(4096), b""):
                sha256_hash.update(byte_block)
        sha256_value = sha256_hash.hexdigest()
        result_dict[file_name] = {"sha256sum": sha256_value}
    return result_dict


class ModelSizeChecker(RrecheckerBase):
    __checker_name__ = "ModelSize"

    def collect_env(self, mindie_service_path=None, **kwargs):
        model_name, model_weight_path = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)

        if not model_name or not model_weight_path:
            return None

        model_json_size = get_file_sizes(os.path.join(model_weight_path, "*.json"))
        model_weight_size = get_file_sizes(os.path.join(model_weight_path, "*.safetensors"))
        logger.debug(f"ModelSizeChecker model_weight_size={get_next_dict_item(model_weight_size)}")
        return {"model_name": model_name, "model_json_size": model_json_size, "model_weight_size": model_weight_size}

    @staticmethod
    def to_g_size(src_size):
        return "{:.2f}G".format(src_size / 1024 / 1024 / 1024)

    def do_precheck(self, model_config, **kwargs):
        if not model_config:
            return

        model_name = model_config.get("model_name", None)
        model_weight_size = model_config.get("model_weight_size", None)
        if not is_deepseek_model(model_name) or not model_weight_size:
            return

        total_weight_size = sum([vv.get("size", 0) for vv in model_weight_size.values()])
        total_weight_size_str = self.to_g_size(total_weight_size)

        min_fp8_size, max_fp8_size = DEEPSEEK_R1_FP8_WEIGHT_SIZE * 0.9, DEEPSEEK_R1_FP8_WEIGHT_SIZE * 1.1
        min_fp16_size, max_fp16_size = DEEPSEEK_R1_FP16_WEIGHT_SIZE * 0.9, DEEPSEEK_R1_FP16_WEIGHT_SIZE * 1.1
        is_valid_fp8_deepseek_size = min_fp8_size < total_weight_size < max_fp8_size
        is_valid_fp16_deepseek_size = min_fp16_size < total_weight_size < max_fp16_size
        if not is_valid_fp8_deepseek_size and not is_valid_fp16_deepseek_size:
            fp8_weight_size_str = self.to_g_size(DEEPSEEK_R1_FP8_WEIGHT_SIZE)
            fp16_weight_size_str = self.to_g_size(DEEPSEEK_R1_FP16_WEIGHT_SIZE)
            show_check_result(
                "Model",
                "size",
                CheckResult.ERROR,
                action="检查当前权重大小：{total_weight_size_str}",
                reason=f"FP8 权重大小应大约 {fp8_weight_size_str}，FP16 权重大小应大约 {fp16_weight_size_str}",
            )
        else:
            show_check_result("Model", f"size: {total_weight_size_str}", CheckResult.OK)


class ModelSha256Collecter(RrecheckerBase):
    __checker_name__ = "ModelSha256"

    def collect_env(self, mindie_service_path=None, **kwargs):
        model_name, model_weight_path = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)

        if not model_name or not model_weight_path:
            return None

        logger.warning("Model checking sha256sum value could take ~30mins depending on model weights size")
        model_json_sha256 = get_file_sha256s(os.path.join(model_weight_path, "*.json"))
        model_weight_sha256 = get_file_sha256s(os.path.join(model_weight_path, "*.safetensors"))
        logger.debug(f"ModelSha256Collecter model_weight_sha256={model_weight_sha256}")
        return {
            "model_name": model_name,
            "model_json_sha256": model_json_sha256,
            "model_weight_sha256": model_weight_sha256,
        }

    def do_precheck(self, model_config, **kwargs):
        logger.warning("Precheck with modelsha256 checker is meaningless. Will skip it")
        return

model_size_checker = ModelSizeChecker()
model_sha256_collecter = ModelSha256Collecter()
