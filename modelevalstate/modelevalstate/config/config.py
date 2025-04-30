# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import time
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple, Type, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, JsonConfigSettingsSource

import modelevalstate

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
INSTALL_PATH = Path(modelevalstate.__path__[0])
RUN_PATH = Path(os.getcwd())


MODEL_EVAL_STATE_CONFIG_PATH = "MODEL_EVAL_STATE_CONFIG_PATH"
modelevalstate_config_path = os.getenv(MODEL_EVAL_STATE_CONFIG_PATH) or os.getenv(MODEL_EVAL_STATE_CONFIG_PATH.lower())
if not modelevalstate_config_path :
    modelevalstate_config_path = RUN_PATH.joinpath("config.json")
modelevalstate_config_path = Path(modelevalstate_config_path).absolute().resolve()

CUSTOM_OUTPUT = "CUSTOM_OUTPUT"
custom_output = os.getenv(CUSTOM_OUTPUT) or os.getenv(CUSTOM_OUTPUT.lower())
if custom_output:
    custom_output = Path(custom_output).resolve()
else:
    custom_output = RUN_PATH


class AnalyzeTool(Enum):
    default = "default"
    deepseek = "deepseek"
    profiler = "profiler"


class BenchMarkPolicy(Enum):
    benchmark = "benchmark"
    custom_benchmark = "custom_benchmark"
    profiler_benchmark = "profiler_benchmark"


class DeployPolicy(Enum):
    single = "single"
    multiple = "multiple"


class OptimizerConfigField(BaseModel):
    name: str = "max_batch_size"
    config_position: str = "BackendConfig.ScheduleConfig.maxBatchSize"
    min: int = 0
    max: int = 100
    dtype: str = "float"
    value: Union[int, float, bool] = 0.0
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
        if v.dtype == "factories":
            _field = _simulate_run_info[i]
            _t_op = [_op for _op in _simulate_run_info if _op.name == v.dtype_param["target_name"]][0]
            _field.value = eval(v.dtype_param["dtype"])(v.dtype_param["product"] / _t_op.value)

    return _simulate_run_info


class PerformanceIndex(BaseModel):
    average_decode_throughput: Optional[float] = None
    average_prefill_latency: Optional[float] = None
    average_decode_latency: Optional[float] = None
    success_rate: Optional[float] = None


class BenchMarkConfig(BaseModel):
    name: str = "benchmark"
    work_path: Path = Field(default_factory=lambda: Path(os.getenv()).resolve())
    command: str = "/usr/bin/bash ./run_benchmark.sh"
    output_path: Path = custom_output.joinpath("instance")
    custom_collect_output_path: Path = Field(
        default_factory=lambda: custom_output.joinpath("/result/custom_collect_output_path").resolve(),
        validate_default=True
    )
    custom_analysis_output_path: Path = Field(
        default_factory=lambda: custom_output.joinpath("/result/custom_analysis_output_path").resolve(),
        validate_default=True
    )
    profile_input_path: Path = Field(
        default_factory=lambda: custom_output.joinpath("/result/profile_input_path").resolve(),
        validate_default=True
    )
    profile_output_path: Path = Field(
        default_factory=lambda: custom_output.joinpath("/result/profile_output_path").resolve(),
        validate_default=True
    )

    @field_validator("output_path", "custom_collect_output_path", "custom_analysis_output_path", "profile_input_path",
                     "profile_output_path")
    @classmethod
    def create_path(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("work_path")
    @classmethod
    def check_dir(cls, v: Path) -> Path:
        if not v.exists():
            logger.error(f"FileNotFound: {v}")
        return v


class DataStorageConfig(BaseModel):
    store_dir: Path = Field(
        default_factory=lambda: Path(os.getenv()).joinpath("/result/store").resolve(),
        validate_default=True
    )

    @field_validator("store_dir")
    @classmethod
    def create_path(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    

class LatencyModel(BaseModel):
    base_path: Path = Field(default_factory=lambda: custom_output.joinpath("result/latency_model").resolve())
    model_path: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("bak/base/xgb_model.ubj").resolve())
    ohe_path: Optional[Path] = Field(default_factory=lambda data: data["base_path"].joinpath("ohe").resolve())
    static_file_dir: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("model_static_file").resolve(), validate_default=True)
    req_and_decode_file: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("req_id_and_decode_num.json").resolve())

    @field_validator("static_file_dir")
    @classmethod
    def validate_static_file_dir(cls, v: Optional[None]):
        if v is None:
            v = cls.base_path.joinpath("model_static_file")
            v.mkdir(parents=True)
        return v


class MindieConfig(BaseModel):
    # 运行mindie时，要修改的mindie config
    process_name: str = "mindieservice_daemon"
    config_path: Path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json")
    config_bak_path: Path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config_bak.json")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: str = "/usr/bin/bash ./run_mindie.sh"
    log_path: Path = Path("/usr/local/Ascend/mindie/latest/mindie-service/logs")


    @field_validator("config_path")
    @classmethod
    def check_dir(cls, v: Path) -> Path:
        if not v.exists():
            logger.error(f"FileNotFound: {v}")
        return v

    @field_validator("log_path")
    @classmethod
    def create_path(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


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
                         dtype="bool"),
    OptimizerConfigField(name="max_prefill_token",
                         config_position="BackendConfig.ScheduleConfig.maxPrefillTokens", min=4096, max=409600,
                         dtype="int"),
    OptimizerConfigField(name="max_queue_deloy_mircroseconds",
                         config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds", min=500, max=1000000,
                         dtype="int"),
    OptimizerConfigField(name="prefill_policy_type",
                         config_position="BackendConfig.ScheduleConfig.prefillPolicyType", min=0, max=1,
                         dtype="enum", dtype_param=[0, 1, 3]),
    OptimizerConfigField(name="decode_policy_type",
                         config_position="BackendConfig.ScheduleConfig.decodePolicyType", min=0, max=1,
                         dtype="enum", dtype_param=[0, 1, 3]),
    OptimizerConfigField(name="max_preempt_count",
                         config_position="BackendConfig.ScheduleConfig.maxPreemptCount", min=0, max=1,
                         dtype="ratio", dtype_param="max_batch_size")
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        json_file=[INSTALL_PATH.joinpath("model_eval_state.json"), Path("~/model_eval_state.json").expanduser(),
                   RUN_PATH.joinpath("model_eval_state.json"),
                   INSTALL_PATH.joinpath("config.json"), Path("~/config.json").expanduser(),
                   RUN_PATH.joinpath("config.json"), modelevalstate_config_path],
        env_prefix="model_eval_state_")
    output: Path = Field(default_factory=lambda: Path(os.getcwd()).joinpath("result").resolve(), validate_default=True)
    latency_model: LatencyModel = LatencyModel()
    mindie: MindieConfig = MindieConfig()
    benchmark: BenchMarkConfig = BenchMarkConfig()
    data_storage: DataStorageConfig = DataStorageConfig()
    pso_options: PsoOptions = PsoOptions()
    n_particles: int = 5
    iters: int = 10
    ftol: float = -np.inf
    ftol_iter: int = 1
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
        return (init_settings, env_settings, JsonConfigSettingsSource(settings_cls), file_secret_settings)

    @field_validator("output")
    @classmethod
    def create_path(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


settings = Settings()
logger.debug(f"expect load json file: {settings.model_config['json_file']}")
logger.debug(f"load settings: {settings.model_dump()}")
