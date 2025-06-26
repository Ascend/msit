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

from msguard.security import open_s

from msprechecker.prechecker.register import PrecheckerBase, show_check_result, CheckResult
from msprechecker.prechecker.utils import logger, get_model_path_from_mindie_config
from msprechecker.prechecker.utils import is_deepseek_model
from msprechecker.prechecker.utils import is_deepseek_model, get_next_dict_item

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


def _should_hash_entire_file(num_blocks, total_size, block_size):
    return num_blocks <= 0 or total_size <= block_size * num_blocks


def _hash_file_sequentially(sha256_hash, file, block_size):
    for chunk in iter(lambda: file.read(block_size), b""):
        sha256_hash.update(chunk)


def _hash_file_sampled(sha256_hash, file, total_size, num_blocks, block_size):
    step = max(1, total_size // num_blocks)
    test_positions = list(range(0, total_size, step)) + [max(0, total_size - block_size)]
    
    for pos in test_positions:
        if 0 <= pos < total_size:
            file.seek(pos, 0)
            chunk = file.read(block_size)
            if chunk:
                sha256_hash.update(chunk)


def update_hash256(sha256_hash, file_path, total_size, num_blocks, block_size):
    block_size = min(block_size, total_size)
    
    with open_s(file_path, "rb") as file:
        if _should_hash_entire_file(num_blocks, total_size, block_size):
            _hash_file_sequentially(sha256_hash, file, block_size)
        else:
            _hash_file_sampled(sha256_hash, file, total_size, num_blocks, block_size)


def get_file_sha256s(file_path_regex, block_size=4096, num_blocks=1000):
    files = glob.glob(file_path_regex)
    result_dict = {}

    for file_path in files:
        file_name = os.path.basename(file_path)
        total_size = os.path.getsize(file_path)
        
        sha256_hash = hashlib.sha256()

        if total_size == 0:
            result_dict[file_name] = {"sha256sum": sha256_hash.hexdigest()}
            continue

        update_hash256(sha256_hash, file_path, total_size, num_blocks, block_size)
        result_dict[file_name] = {"sha256sum": sha256_hash.hexdigest()}

    return result_dict


class ModelSizeChecker(PrecheckerBase):
    __checker_name__ = "ModelSize"
    
    @staticmethod
    def to_g_size(src_size):
        return "{:.2f}G".format(src_size / 1024 / 1024 / 1024)

    def collect_env(self, mindie_service_path=None, **kwargs):
        weight_dir = kwargs.get("weight_dir")
        model_name = "deepseek"
        if not weight_dir:
            model_name, weight_dir = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)
        
        if not model_name or not weight_dir:
            return None

        model_json_size = get_file_sizes(os.path.join(weight_dir, "*.json"))
        model_weight_size = get_file_sizes(os.path.join(weight_dir, "*.safetensors"))
        logger.debug(f"ModelSizeChecker model_weight_size={get_next_dict_item(model_weight_size)}")
        return {"model_name": model_name, "model_json_size": model_json_size, "model_weight_size": model_weight_size}

    def do_precheck(self, envs, **kwargs):
        if not envs:
            return

        model_name = envs.get("model_name", None)
        model_weight_size = envs.get("model_weight_size", None)
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


class ModelSha256Collecter(PrecheckerBase):
    __checker_name__ = "ModelSha256"

    def collect_env(self, mindie_service_path=None, sha256_blocknum=1000, **kwargs):
        model_name, model_weight_path = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)

        if not model_name or not model_weight_path:
            return None

        model_json_sha256 = get_file_sha256s(os.path.join(model_weight_path, "*.json"), num_blocks=0)
        model_weight_sha256 = get_file_sha256s(
            os.path.join(model_weight_path, "*.safetensors"), num_blocks=sha256_blocknum
        )
        logger.debug(f"ModelSha256Collecter model_weight_sha256={model_weight_sha256}")
        return {
            "model_name": model_name,
            "model_json_sha256": model_json_sha256,
            "model_weight_sha256": model_weight_sha256,
        }

    def do_precheck(self, envs, **kwargs):
        logger.warning("Precheck with modelsha256 checker is meaningless. Will skip it")
        return


model_size_checker = ModelSizeChecker()
model_sha256_collecter = ModelSha256Collecter()
