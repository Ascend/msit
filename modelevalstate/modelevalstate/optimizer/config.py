# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import time
from pathlib import Path
from typing import Tuple, Type, Callable
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, JsonConfigSettingsSource

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())


class BenchMarkPolicy:
    benchmark = "benchmark"
    custom_benchmark = "custom_benchmark"


class OptimizerConfigField(BaseModel):
    name: str
    config_position: str
    min: int = 0
    max: int = 100
    format_func: Callable = float
    value: float = 0.0


class PerformanceIndex(BaseModel):
    average_decode_throughput: float
    average_prefill_latency: float
    average_decode_latency: float


class BenchMarkConfig(BaseModel):
    source_env: str = ""
    name: str = "benchmark"
    command: str = (r"benchmark --DatasetPath /data/xjt/data/gsm8k --DatasetType gsm8k --ModelName llama3-8b "
                    r"--ModelPath /data/LLM/llama3-8b --TestType client --MaxOutputLen 256 "
                    r"--Http http://127.0.0.1:7425 --ManagementHttp http://127.0.0.2:7426 "
                    r"--Concurrency 300 --RequestRate 20 --WarmupSize 20 --Tokenizer True "
                    r"--SaveTokensPath /data/xjt/output/gsm8k.csv")
    output_path: str = r"/data/xjt/1210test/Ascend-mindie-server_1.0.RC3_linux-aarch64/instance"
    custom_collect_output_path: str = r"/data/xjt/1210test/Ascend-mindie-server_1.0.RC3_linux-aarch64/simulate_logs"
    custom_analysis_output_path: str = r"/data/xjt/model_forward.csv"


class DataStorageConfig(BaseModel):
    store_dir: Path = Path("/data/xjt/output/")


class MindieConfig(BaseModel):
    # 运行mindie时，要修改的mindie config
    process_name: str = "mindieservice_daemon"
    config_path: Path = Path("./conf/config.json")
    config_bak_path: Path = Path("./conf/config_bak.json")
    work_path: str = "/data/xjt/1210test/Ascend-mindie-server_1.0.RC3_linux-aarch64"
    command: str = "./bin/mindieservice_daemon &"
    source_env: str = "source scripts/set_env.sh"
    log_path: str = "/data/xjt/1210test/Ascend-mindie-server_1.0.RC3_linux-aarch64/logs"


class PsoOptions(BaseModel):
    c1: float = 0.5
    c2: float = 0.3
    w: float = 0.9


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
    prefill_constrain: float = 0.05
    decode_constrain: float = 0.05
    float_range_in_best_particle: float = 0.1 # 如果用历史值作为作为初始值，那么允许在初始值的浮动程度。

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


default_support_field = (
    # max batch size 最小值要大于max_prefill_batch_size的最大值。
    OptimizerConfigField(name="max_batch_size", config_position="BackendConfig.ScheduleConfig.maxBatchSize", min=25,
                         max=300, format_func=int),
    OptimizerConfigField(name="max_prefill_batch_size",
                         config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize", min=1, max=25,
                         format_func=int),
    OptimizerConfigField(name="prefill_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq", max=1000, format_func=int),
    OptimizerConfigField(name="decode_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq", max=1000, format_func=int),
    OptimizerConfigField(name="support_select_batch",
                         config_position="BackendConfig.ScheduleConfig.supportSelectBatch", max=1,
                         format_func=lambda x: True if x > 0.5 else False),
)

settings = Settings()
