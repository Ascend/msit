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

from typing import Dict, Optional, List, Any

import torch.nn as nn
from pydantic import BaseModel

from msmodelslim import logger
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.legacy_runner import LegacyRunner
from msmodelslim.quant.model.base import BaseModelAdapter
from msmodelslim.quant.model.registry import create_model_adapter
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY
from msmodelslim.quant.session.backward import BackwardProcessorMerger
from msmodelslim.quant.session.forward import ForwardProcessorMerger


class SessionConfig(BaseModel):
    processor_cfg_map: Dict[str, BaseModel] = {}
    calib_data: Optional[List[Any]] = None


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

    processor_map = {}
    for name, cfg in session_cfg.processor_cfg_map.items():
        processor_cls = PROCESSOR_REGISTRY.get_by_name(name)
        if processor_cls is None:
            raise ValueError(f"Processor for {name} is not found")
        logger.info(f"Create {name} processor with cfg {cfg}")
        processor_map[name] = processor_cls(model, cfg)

    stage_processor_map = {}
    for _, processor in processor_map.items():
        if processor.stage() not in stage_processor_map:
            stage_processor_map[processor.stage()] = []
        stage_processor_map[processor.stage()].append(processor)

    runner = _create_runner(model, model_adapter)

    for stage, processors in stage_processor_map.items():
        if stage == ProcessStage.PREPARE_MODEL:
            stage_processor = ForwardProcessorMerger(model, processors)
        elif stage == ProcessStage.FORWARD_ANTI_OUTLIER:
            stage_processor = ForwardProcessorMerger(model, processors)
        elif stage == ProcessStage.BACKWARD_ANTI_OUTLIER:
            stage_processor = BackwardProcessorMerger(model, processors)
        elif stage == ProcessStage.FORWARD_QUANT:
            stage_processor = ForwardProcessorMerger(model, processors)
        elif stage == ProcessStage.BACKWARD_QUANT:
            stage_processor = BackwardProcessorMerger(model, processors)
        elif stage == ProcessStage.SAVE_MODEL:
            stage_processor = ForwardProcessorMerger(model, processors)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        if not stage_processor.is_data_free() and session_cfg.calib_data is None:
            raise ValueError(f"Calib data is required for {stage} stage but not provided")

        stage_data = None if stage_processor.is_data_free() else session_cfg.calib_data
        runner.add_processor(stage_processor, stage_data)

    runner.run()


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
