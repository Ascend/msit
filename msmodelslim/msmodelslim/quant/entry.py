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

import argparse
import os
import time
from typing import Dict, Any

import torch
from pydantic import BaseModel
from torch import nn, distributed as dist

from msmodelslim import logger, set_logger_level
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant import (quant_model,
                               SessionConfig,
                               SaverProcessorConfig,
                               M1ProcessorConfig)
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.registry import PROCESSOR_CONFIG_REGISTRY, PROCESSOR_REGISTRY
from msmodelslim.quant.session.plugin import load_plugin


@PROCESSOR_CONFIG_REGISTRY.register_by_name("load")
class LoadConfig(BaseModel):
    dev_type: str = "npu"


@PROCESSOR_CONFIG_REGISTRY.register_by_name("offload")
class OffloadConfig(BaseModel):
    dev_type: str = "meta"


@PROCESSOR_REGISTRY.register("load")
class ModelLoad(SessionBaseProcessor):
    def __init__(self, model: nn.Module, cfg: LoadConfig, **kwargs: Dict[str, Any]):
        super().__init__(model)
        self.cfg = cfg
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = torch.device(self.cfg.dev_type) \
            if self.cfg.dev_type in ["cpu", "meta"] \
            else torch.device(f"{self.cfg.dev_type}:{self.local_rank}")

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.LOAD_MODEL

    def preprocess(self, request: BatchProcessRequest) -> None:
        request.module.to(self.device)


@PROCESSOR_REGISTRY.register("offload")
class ModelOffload(SessionBaseProcessor):
    def __init__(self, model: nn.Module, cfg: OffloadConfig, **kwargs: Dict[str, Any]):
        super().__init__(model)
        self.cfg = cfg
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = torch.device(self.cfg.dev_type) \
            if self.cfg.dev_type in ["cpu", "meta"] \
            else torch.device(f"{self.cfg.dev_type}:{self.local_rank}")

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.OFFLOAD_MODEL

    def preprocess(self, request: BatchProcessRequest) -> None:
        request.module.to(self.device)


def _set_default_device(dev_type: str):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if dev_type == "cuda":
        torch.set_default_device(f"cuda:{local_rank}")
    elif dev_type == "npu":
        torch.set_default_device(f"npu:{local_rank}")
    elif dev_type == "cpu":
        torch.set_default_device("cpu")
    else:
        raise ValueError(f"不支持的设备类型: {dev_type}")


def _setup_distributed(rank: int, world_size: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:29505", rank=rank, world_size=world_size)
    torch.npu.set_device(f"npu:{rank}")
    dist.barrier()
    logger.info(f"Distributed using backend: {dist.get_backend()}")


def _cleanup_distributed(rank: int, world_size: int, args: argparse.Namespace):
    dist.destroy_process_group()


def _main(args: argparse.Namespace):
    start_time = time.time()

    if args.debug:
        set_logger_level("debug")

    _set_default_device(args.dev_type)

    plugin = load_plugin(args.plugin_path, args)

    default_save_cfg = SaverProcessorConfig(
        save_output_path=args.save_path,
        safetensors_name=f"quant_model_weight_{args.quant_type}.safetensors",
        json_name=f"quant_model_description_{args.quant_type}.json",
        save_type="safe_tensor",
        part_file_size=4,
        model_quant_type=args.quant_type
    )

    model = plugin.load_model()
    calib_data = plugin.load_calib_data()
    quant_cfg = plugin.load_default_quant_cfg()
    save_cfg = plugin.get_save_cfg(default_save_cfg)

    session_config = SessionConfig(
        processor_cfg_map={
            "load": LoadConfig(dev_type=args.dev_type),
            "m1": M1ProcessorConfig(),
            **quant_cfg,
            "save": save_cfg,
            "offload": OffloadConfig(dev_type=args.offload_dev_type)
        },
        calib_data=calib_data
    )

    session_config.model_validate(session_config)

    quant_model(model, session_config)

    plugin.eval_model()

    if dist.is_initialized():
        dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        end_time = time.time()
        logger.info(f"量化总耗时: {end_time - start_time:.2f}秒")


def _dist_main(rank: int, world_size: int, args: argparse.Namespace):
    _setup_distributed(rank, world_size, args)
    _main(args)
    _cleanup_distributed(rank, world_size, args)
