#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, InstanceOf, Field, field_serializer, field_validator

from msmodelslim import logger
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.legacy_runner import LegacyRunner
from msmodelslim.quant.model.base import BaseModelAdapter
from msmodelslim.quant.model.registry import create_model_adapter
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.quant.session.backward import BackwardProcessorMerger
from msmodelslim.quant.session.forward import ForwardProcessorMerger


class SessionConfig(BaseModel):
    processor_cfg_map: Dict[str, InstanceOf[BaseModel]] = Field(default_factory=dict)
    calib_data: Optional[List[Any]] = Field(default=None, exclude=True)

    @field_validator('processor_cfg_map', mode='before')
    @classmethod
    def validate_processor_cfg_map(cls, v):
        if isinstance(v, dict):
            return {k: PROCESSOR_CONFIG_REGISTRY.get_by_name(k).model_validate(val)
                    for k, val in v.items()}
        return v

    @field_serializer('processor_cfg_map')
    @classmethod
    def serialize_processor_cfg_map(cls, v):
        return {k: v.model_dump() for k, v in v.items()}


@dataclass
class SessionContext:
    config: SessionConfig
    model: nn.Module
    adapter: BaseModelAdapter
    runner: BaseRunner


def quant_model(model: nn.Module,
                session_cfg: SessionConfig,
                ):
    """
    根据会话配置对模型进行量化处理。
    
    该方法会根据session_cfg中的配置类型集合构造相应的处理器，并对模型进行处理。
    
    参数:
        model: 需要处理的模型
        session_cfg: 会话配置，描述整个量化策略，包含cfg_map/save_cfg
    """

    logger.info(f"Quant model with cfg: {session_cfg.processor_cfg_map}")

    model_adapter = create_model_adapter(model)
    runner = _create_runner(model, model_adapter)
    session_context = SessionContext(config=session_cfg,
                                     model=model,
                                     adapter=model_adapter,
                                     runner=runner)

    processor_map: Dict[str, SessionBaseProcessor] = {}
    for name, cfg in session_cfg.processor_cfg_map.items():
        processor_map[name] = _create_processor(session_context, name, cfg)

    is_distributed = dist.is_initialized()
    support_map = {name: processor.support_distributed() for name, processor in processor_map.items()}
    support_distributed = all(processor.support_distributed() for processor in processor_map.values())
    if is_distributed and not support_distributed:
        raise ValueError(f"Distributed is enabled but some processor does not support distributed: {support_map}")

    stage_processor_map = {}
    for _, processor in processor_map.items():
        if processor.stage() not in stage_processor_map:
            stage_processor_map[processor.stage()] = []
        stage_processor_map[processor.stage()].append(processor)

    for stage, processors in stage_processor_map.items():

        if not isinstance(stage, ProcessStage):
            raise ValueError(f"Unsupported stage: {stage}")

        stage_container = {
            ProcessStage.LOAD_MODEL: ForwardProcessorMerger,
            ProcessStage.FORWARD_ANTI_OUTLIER: ForwardProcessorMerger,
            ProcessStage.BACKWARD_ANTI_OUTLIER: BackwardProcessorMerger,
            ProcessStage.FORWARD_QUANT: ForwardProcessorMerger,
            ProcessStage.BACKWARD_QUANT: BackwardProcessorMerger,
            ProcessStage.SAVE_MODEL: ForwardProcessorMerger,
            ProcessStage.OFFLOAD_MODEL: ForwardProcessorMerger,
        }

        stage_processor = stage_container[stage](model, processors)

        if not stage_processor.is_data_free() and not session_cfg.calib_data:
            raise ValueError(f"Calib data is required for {stage} stage but not provided")

        stage_data = None if stage_processor.is_data_free() else session_cfg.calib_data
        runner.add_processor(stage_processor, stage_data)

    runner.run()


def _create_processor(session_context: SessionContext, name: str, cfg: BaseModel) -> SessionBaseProcessor:
    processor_cls = PROCESSOR_REGISTRY.get_by_name(name)
    if processor_cls is None:
        raise ValueError(f"Processor for {name} is not found")
    logger.info(f"Create {name} processor with cfg {cfg}")
    kwargs = {
        "adapter": session_context.adapter,
        "runner": session_context.runner,
    }
    return processor_cls(session_context.model, cfg, **kwargs)


def _create_runner(model: nn.Module, model_adapter: BaseModelAdapter) -> BaseRunner:
    try:
        if model_adapter is not None:
            model_adapter.get_decoder_layers()
            return LayerWiseRunner(model)
        return LegacyRunner(model)
    except NotImplementedError:
        logger.warning(f"Can't create layer wise runner, use legacy runner instead")
        return LegacyRunner(model)
    except Exception as e:
        raise e
