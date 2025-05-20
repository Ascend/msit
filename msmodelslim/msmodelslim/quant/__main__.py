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

import torch
from pydantic import BaseModel
from torch import distributed as dist
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig, W8A8ProcessorConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session import quant_model, SessionConfig
from msmodelslim.quant.session.plugin import load_plugin

DEFAULT_CFG_MAP = {
    "w8a8": W8A8QuantConfig(
        w_cfg=dict(
            bits=8,
            symmetric=True,
            per_channel=True
        ),
        a_cfg=dict(
            bits=8,
            symmetric=True,
            per_channel=False
        )
    )
}


class LoadConfig(BaseModel):
    dev_type: str = "npu"


class OffloadConfig(BaseModel):
    dev_type: str = "meta"


@PROCESSOR_REGISTRY.register("load")
class ModelLoad(SessionBaseProcessor):
    def __init__(self, model: nn.Module, cfg: LoadConfig):
        super().__init__(model)
        self.cfg = cfg

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.LOAD_MODEL

    def preprocess(self, request: BatchProcessRequest) -> None:
        request.module.to(self.cfg.dev_type)


@PROCESSOR_REGISTRY.register("offload")
class ModelOffload(SessionBaseProcessor):
    def __init__(self, model: nn.Module, cfg: OffloadConfig):
        super().__init__(model)
        self.cfg = cfg

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.OFFLOAD_MODEL

    def preprocess(self, request: BatchProcessRequest) -> None:
        request.module.to(self.cfg.dev_type)


def set_default_device(dev_type: str):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if dev_type == "cuda":
        torch.set_default_device(f"cuda:{local_rank}")
    elif dev_type == "npu":
        torch.set_default_device(f"npu:{local_rank}")
    elif dev_type == "cpu":
        torch.set_default_device("cpu")
    else:
        raise ValueError(f"不支持的设备类型: {dev_type}")


def main():
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("-p", "--plugin_path", type=str, required=True, help="插件文件路径")
    parser.add_argument("-t", "--quant_type", type=str, help="量化类型", default="w8a8")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_path", type=str, help="保存路径", default="./save")
    parser.add_argument("--dev_type", type=str, help="设备类型", default="npu")
    parser.add_argument("--offload_dev_type", type=str, help="offload设备类型", default="meta")
    args = parser.parse_args()

    dist.init_process_group()
    set_default_device(args.dev_type)

    plugin = load_plugin(args.plugin_path, args)

    default_linear_config = DEFAULT_CFG_MAP.get(args.quant_type, None)

    default_cfg_map = {
        "*": default_linear_config
    }

    default_save_cfg = SaverProcessorConfig(
        save_output_path=args.save_path,
        safetensors_name="quant_model_weight_w8a8.safetensors",
        json_name="quant_model_description_w8a8.json",
        save_type="safe_tensor"
    )

    default_cfg = W8A8ProcessorConfig(
        cfg_map=default_cfg_map
    )

    model = plugin.load_model()
    calib_data = plugin.load_calib_data()
    quant_cfg = plugin.load_quant_cfg(default_cfg)
    save_cfg = plugin.get_save_cfg(default_save_cfg)

    session_config = SessionConfig(
        processor_cfg_map={
            "load": LoadConfig(dev_type=args.dev_type),
            args.quant_type: quant_cfg,
            "save": save_cfg,
            "offload": OffloadConfig(dev_type=args.offload_dev_type)
        },
        calib_data=calib_data
    )

    session_config.model_validate(session_config)

    quant_model(model, session_config)

    plugin.eval_model()


if __name__ == "__main__":
    main()
