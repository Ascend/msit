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

import time
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple, Type, Optional

import numpy as np
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, JsonConfigSettingsSource

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())


class AnalyzeTool(Enum):
    default = "default"
    deepseek = "deepseek"


class BenchMarkPolicy(Enum):
    benchmark = "benchmark"
    custom_benchmark = "custom_benchmark"


class DeployPolicy(Enum):
    single = "single"
    multiple = "multiple"


class OptimizerConfigField(BaseModel):
    name: str = "max_batch_size"
    config_position: str = "BackendConfig.ScheduleConfig.maxBatchSize"
    min: int = 0
    max: int = 100
    dtype: str = "float"
    value: float = 0.0
    dtype_param: Any = None


def map_param_with_value(params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
    _simulate_run_info = []
    for i, v in enumerate(params_field):
        _field = deepcopy(v)
        if v.dtype == "int":
            _field.value = int(params[i])
        elif v.dtype == "bool":
            if params[i] > 0.5:
                _field.value = True
            else:
                _field.value = False
        elif v.dtype == "enum":
            segment = np.linspace(v.min, v.max, len(v.dtype_param) + 1)
            if params[i] <= v.min:
                _field.value = v.dtype_param[0]
            elif params[i] >= v.max:
                _field.value = v.dtype_param[-1]
            else:
                _enum_index = np.searchsorted(segment, params[i]) - 1
                _field.value = v.dtype_param[_enum_index]
        else:
            _field.value = float(params[i])
        _simulate_run_info.append(_field)
    for i, v in enumerate(params_field):
        if v.dtype == "ratio":
            _field = _simulate_run_info[i]
            _t_op = [_op for _op in _simulate_run_info if _op.name == v.dtype_param][0]
            _field.value = int(_field.value * _t_op.value)
    return _simulate_run_info


class PerformanceIndex(BaseModel):
    average_decode_throughput: float
    average_prefill_latency: float
    average_decode_latency: Optional[float] = None
    success_rate: Optional[float] = None


class BenchMarkConfig(BaseModel):
    name: str = "benchmark"
    command: str = r"benchmark --DatasetPath ./data/gsm8k \
        --DatasetType gsm8k --ModelName llama3-8b --ModelPath /data/LLM/llama3-8b \
            --TestType client --MaxOutputLen 256 --Http http://127.0.0.1:7425 \
                --ManagementHttp http://127.0.0.2:7426 --Concurrency 300 --RequestRate 20 \
                    --WarmupSize 20 --Tokenizer True --SaveTokensPath /data/xjt/output/gsm8k.csv"
    output_path: str = r"./Ascend-mindie-server_1.0.RC3_linux-aarch64/instance"
    custom_collect_output_path: str = r"./Ascend-mindie-server_1.0.RC3_linux-aarch64/simulate_logs"
    custom_analysis_output_path: str = r"./model_forward.csv"


class DataStorageConfig(BaseModel):
    store_dir: Path = Path("/data/xjt/output/")


class MindieConfig(BaseModel):
    # 运行mindie时，要修改的mindie config
    process_name: str = "mindieservice_daemon"
    config_path: Path = Path("./conf/config.json")
    config_bak_path: Path = Path("./conf/config_bak.json")
    work_path: str = "./Ascend-mindie-server_1.0.RC3_linux-aarch64"
    command: str = "./bin/mindieservice_daemon"
    log_path: str = "./Ascend-mindie-server_1.0.RC3_linux-aarch64/logs"


class PsoOptions(BaseModel):
    c1: float = 0.5
    c2: float = 0.3
    w: float = 0.9


default_support_field = [
    # max batch size 最小值要大于max_prefill_batch_size的最大值。
    OptimizerConfigField(name="max_batch_size", config_position="BackendConfig.ScheduleConfig.maxBatchSize", min=25,
                         max=300, dtype="int"),
    OptimizerConfigField(name="max_prefill_batch_size",
                         config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize", min=1, max=25,
                         dtype="int"),
    OptimizerConfigField(name="prefill_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq", max=1000, dtype="int"),
    OptimizerConfigField(name="decode_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq", max=1000, dtype="int"),
    OptimizerConfigField(name="support_select_batch",
                         config_position="BackendConfig.ScheduleConfig.supportSelectBatch", max=1,
                         format_func="bool"),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(json_file="config.json")
    output: Path = Path("/data/wdk/output/")
    mindie: MindieConfig = MindieConfig()
    benchmark: BenchMarkConfig = BenchMarkConfig()
    data_storage: DataStorageConfig = DataStorageConfig()
    pso_options: PsoOptions = PsoOptions()
    n_particles: int = 5
    iters: int = 10
    prefill_lam: float = 0.5  # 惩罚系数
    decode_lam: float = 0.5
    success_rate_lam: float = 0.5
    prefill_constrain: float = 0.05
    decode_constrain: float = 0.05
    success_rate_constrain: float = 1.0
    server: List[str] = ["http://127.0.0.1:7425/"]
    float_range_in_best_particle: float = 0.1  # 如果用历史值作为作为初始值，那么允许在初始值的浮动程度。
    target_field: List[OptimizerConfigField] = default_support_field

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, JsonConfigSettingsSource(settings_cls), env_settings, file_secret_settings)


settings = Settings()