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

import abc
from enum import Enum
from typing import Dict, List, Any, Tuple

from torch import nn as nn

from ascend_utils.common.utils import CallParams
from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import extract_dag
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY


class StatKey(str, Enum):
    STAT_KEY_MAX = "max"
    STAT_KEY_MIN = "min"
    STAT_KEY_SHIFT = "shift"
    STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
    STAT_KEY_THRESHOLD_TENSOR = "thres_t"
    STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
    STAT_KEY_SMOOTH_SCALE = "smooth_scale"
    STAT_KEY_VARIANCE = "std"
    TENSOR = 'tensor'


class SmoothAdapter:

    @abc.abstractmethod
    def get_global_norm_linear_smooth_pair(self) -> Dict[str, List[str]]:
        """
        获取模型全局的用于进行Smooth处理的Norm->Linear子图
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_global_linear_linear_smooth_pair(self) -> Dict[str, List[str]]:
        """
        获取模型全局的用于进行Smooth处理的Linear->Linear子图
        """

        raise NotImplementedError()


@PROCESSOR_REGISTRY.register_by_name("m1")
class BaseSmoothProcessor(SessionBaseProcessor):

    def __init__(self, model: nn.Module, adapter: SmoothAdapter):
        super().__init__(model)
        self.adapter = adapter

    @staticmethod
    def _extract_layer_wise_norm_linear_smooth_pair(prefix: str, module: nn.Module, dummy_input: Any) -> Dict[
        str, List[str]]:

        """
        获取模型逐层的用于进行Smooth处理的Linear->Linear子图
        返回的是模型的全名
        """

        norm_class = [m.__class__ for m in module.modules() if "norm" in m.__class__.__name__.lower()]
        dag = extract_dag(module, dummy_input, hook_nodes=norm_class)
        dag_subgraph = dag.get_norm_linear_subgraph()
        result = {}
        for norm_node, linear_list in dag_subgraph.items():
            full_name = f"{prefix}.{norm_node}" if prefix != "" else norm_node
            linear_list = [f"{prefix}.{linear}" if prefix != "" else linear for linear in linear_list]
            result[full_name] = linear_list
        return result

    def stage(self) -> ProcessStage:
        return ProcessStage.FORWARD_ANTI_OUTLIER

    def is_data_free(self) -> bool:
        _ = self
        return False

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._install_statis_hook(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._apply_smooth(request.name, request.module)

    def _get_local_norm_linear_smooth_pair(self, prefix: str, module: nn.Module, dummy_input: Tuple[List, Dict]) -> \
            Dict[str, List[str]]:

        try:
            global_smooth_pair = self.adapter.get_global_norm_linear_smooth_pair()
            layer_prefix = f"{prefix}." if prefix != "" else ""
            return {key: value for key, value in global_smooth_pair.items() if key.startswith(layer_prefix)}
        except (AttributeError, NotImplementedError) as e:
            logger.warning(f"No global smooth pair found, use dag based layer-wise smooth pair instead")
            return self._extract_layer_wise_norm_linear_smooth_pair(prefix, module,
                                                                    CallParams(*dummy_input[0], **dummy_input[1])
                                                                    if dummy_input else None)
        except Exception as e:
            raise e

    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        pass

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        pass
