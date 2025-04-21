# -*- coding: utf-8 -*-
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

import json
from pathlib import Path


def convert(config_path: Path, output_path: Path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    _new_dict = {}
    _new_dict["model"] = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelName"]
    _new_dict["world_size"] = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["worldSize"]
    _new_dict["request_rate"] = 15
    _new_dict["concurrency"] = data["ServerConfig"]["maxLinkNum"]
    _new_dict["prefill_batchsize"] = data["BackendConfig"]["ScheduleConfig"]["maxPrefillBatchSize"]
    _new_dict["decode_batchsize"] = data["BackendConfig"]["ScheduleConfig"]["maxBatchSize"]
    _new_dict["select_batch"] = data["BackendConfig"]["ScheduleConfig"]["supportSelectBatch"]
    _new_dict["prefillTimeMsPerReq"] = data["BackendConfig"]["ScheduleConfig"]["prefillTimeMsPerReq"]
    _new_dict["decodeTimeMsPerReq"] = data["BackendConfig"]["ScheduleConfig"]["decodeTimeMsPerReq"]
    with open(output_path.joinpath(config_path.parent.name, "config&param.json"), 'w') as f:
        json.dump(_new_dict, f, indent=4)


def convert_mindie_config_to_v1_config(mindie_origin_config, v1_config):
    with open(mindie_origin_config, 'r') as f:
        data = json.load(f)
    new_dict = {
        "cache_block_size": data["BackendConfig"]["ScheduleConfig"]["cacheBlockSize"],
        "max_seq_len": data["BackendConfig"]["ModelDeployConfig"]["maxSeqLen"],
        "world_size": data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["worldSize"],
        "cpu_mem_size": data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["cpuMemSize"],
        "npu_mem_size": data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["npuMemSize"],
        "max_prefill_tokens": data["BackendConfig"]["ScheduleConfig"]["maxPrefillTokens"],
        "max_prefill_batch_size": data["BackendConfig"]["ScheduleConfig"]["maxPrefillBatchSize"],
        "max_batch_size": data["BackendConfig"]["ScheduleConfig"]["maxBatchSize"],
    }
    with open(v1_config, 'w') as f:
        json.dump(new_dict, f)


def convert_model_config_to_v1_config(model_origin_config, v1_config):
    with open(model_origin_config, 'r') as f:
        data = json.load(f)
    new_dict = {
        "architectures": data["architectures"],
        "hidden_act": data["hidden_act"],
        "initializer_range": data.get("initializer_range", 0.02),
        "intermediate_size": data.get("intermediate_size", 1),
        "max_position_embeddings": data.get("max_position_embeddings", 0),
        "model_type": data["model_type"],
        "num_attention_heads": data.get("num_attention_heads", 32),
        "num_hidden_layers": data.get("num_hidden_layers", 32),
        "tie_word_embeddings": data.get("tie_word_embeddings", 0),
        "torch_dtype": data.get("torch_dtype", "float16"),
        "use_cache": data.get("use_cache", 0),
        "vocab_size": data.get("vocab_size", 0),
        "inference_mode": data.get("inference_mode", 1),
        "is_flash_causal_lm": data.get("is_flash_causal_lm", 1),
        "quantize": data.get("quantize", ""),
        "quantization_config": data.get("quantization_config", "")
    }
    with open(v1_config, 'w') as f:
        json.dump(new_dict, f)


def main():
    _configs = [r"D:\PyProject\state_eval\data\建模数据\long_8b_log\config.json",
                r"D:\PyProject\state_eval\data\建模数据\medium1_8b_log\config.json",
                r"D:\PyProject\state_eval\data\建模数据\medium1_70b_log\config.json",
                r"D:\PyProject\state_eval\data\建模数据\medium2_8b_log\config.json",
                r"D:\PyProject\state_eval\data\建模数据\medium2_70b_log\config.json",
                ]
    for _file in _configs:
        convert(Path(_file), Path(r"/data"))
