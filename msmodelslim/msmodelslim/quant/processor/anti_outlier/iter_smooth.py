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

from functools import partial
from typing import Dict, Callable, List, Any, Literal, Annotated

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import AfterValidator, Field

from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.QAL.qtypes import (
    RMSNormBias,
    IterSmoothConfig,
    OVSubgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    UpDownSubgraph
)
from msmodelslim.core.api import iter_smooth
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.smooth_base import BaseSmoothProcessor, BaseSmoothProcessorConfig
from msmodelslim.quant.processor.anti_outlier.smooth_base import StatKey
from msmodelslim.quant.processor.base import AutoSessionProcessor
from msmodelslim.utils.dist import DistHelper
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.value import validate_normalized_value, is_boolean, is_string_list            


class IterSmoothProcessorConfig(BaseSmoothProcessorConfig):
    type: Literal["iter_smooth"] = "iter_smooth"
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = 0.9
    scale_min: Annotated[float, AfterValidator(validate_normalized_value)] = 1e-5
    symmetric: Annotated[bool, AfterValidator(is_boolean)] = True
    enable_subgraph_type: Annotated[List[str], AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )


@QABCRegistry.register(dispatch_key=IterSmoothProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter()
class IterSmoothProcessor(BaseSmoothProcessor):
    # 子图类型映射表
    SUBGRAPH_TYPE_MAP = {
        NormLinearSubgraph: "norm-linear",
        LinearLinearSubgraph: "linear-linear",
        OVSubgraph: "ov",
        UpDownSubgraph: "up-down"
    }
    
    # 非对称模式支持的子图类型，默认只支持norm-linear；其他子图走对称逻辑
    ASYM_SUPPORT_SUBGRAPH_TYPES = ["norm-linear"]

    def __init__(self, model: nn.Module, config: IterSmoothProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        super().validate_parameters()
        self.act_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.dist_helper = DistHelper(self.model) if dist.is_initialized() else None

        # 存储hook句柄，用于后续删除
        self.hook_handles = {}
        
        # 检测非对称配置并发出警告
        if not self.config.symmetric:
            supported_types = ', '.join(self.ASYM_SUPPORT_SUBGRAPH_TYPES)
            get_logger().warning(
                f"Detected asymmetric IterSmooth; currently only supports {supported_types} subgraph types"
            )

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 先调用父类的 preprocess 方法
        super().preprocess(request)
        # 遍历 adapter_config，将 norm_module 替换为 RMSNormBias
        for adapter_config in self.adapter_config:
            if adapter_config.subgraph_type == "norm-linear":
                # 获取 norm_module 信息
                norm_name = adapter_config.mapping.source
                norm_module = self.model.get_submodule(norm_name) if norm_name else None

                if norm_name and norm_module is not None:
                    try:
                        # 检查 norm_module 是否有 weight 属性
                        if hasattr(norm_module, 'weight'):
                            # 创建 RMSNormBias 实例
                            norm_bias = RMSNormBias(norm_module.weight.shape[-1])
                            norm_bias.weight.data.copy_(norm_module.weight.data)
                            norm_bias.weight.data = norm_bias.weight.data.type(norm_module.weight.data.dtype)
                            if hasattr(norm_module, 'bias') and norm_module.bias is not None:
                                norm_bias.bias.data.copy_(norm_module.bias.data)
                                norm_bias.bias.data = norm_bias.bias.data.type(norm_module.weight.data.dtype)
                            norm_bias.to(norm_module.weight.data.device)
                            self.model.set_submodule(norm_name, norm_bias)
                            get_logger().debug(f"{norm_name}: {type(norm_module)} -> {type(norm_bias)}")
                        else:
                            get_logger().warning(f"Norm module {norm_name} does not have weight attribute")
                    except Exception as e:
                        get_logger().warning(f"Failed to replace norm module {norm_name}: {e}")

        get_logger().debug(f"Processed {len(self.adapter_config)} subgraphs for submodule {request.name}")

    def _get_stats_hook(self, name: str, subgraph_type: str = None) -> Callable:
        def stats_hook(module: nn.Linear, input_tensor: tuple, output: Any) -> None:
            # 使用闭包中的name和subgraph_type变量
            tensor = input_tensor[0]

            if name not in self.act_stats:
                self.act_stats[name] = {}
                # 存储收集的tensor到CPU，避免OOM
                self.act_stats[name][StatKey.TENSOR] = tensor.cpu()

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]

            if self.dist_helper is not None and self.dist_helper.is_shared(name):
                tensor = torch.cat(self.dist_helper.gather_variable_shapes(tensor), dim=0)
            coming_max = torch.max(tensor, dim=0)[0]  # [C]
            coming_min = torch.min(tensor, dim=0)[0]  # [C]

            statis_dict = self.act_stats[name]

            # collect the min-max value
            if StatKey.STAT_KEY_MAX in statis_dict:
                statis_dict[StatKey.STAT_KEY_MAX] = torch.max(statis_dict[StatKey.STAT_KEY_MAX], coming_max)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MAX] = coming_max

            if StatKey.STAT_KEY_MIN in statis_dict:
                statis_dict[StatKey.STAT_KEY_MIN] = torch.min(statis_dict[StatKey.STAT_KEY_MIN], coming_min)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MIN] = coming_min

            # channel shifting
            if StatKey.STAT_KEY_SHIFT in statis_dict:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (statis_dict[StatKey.STAT_KEY_MAX] + statis_dict[
                    StatKey.STAT_KEY_MIN]) / 2  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

            if not self.config.symmetric and subgraph_type in self.ASYM_SUPPORT_SUBGRAPH_TYPES:
                channel_max = torch.max((tensor - statis_dict[StatKey.STAT_KEY_SHIFT]).abs().detach(), dim=0)[0]
            else:
                channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

            if StatKey.STAT_KEY_SMOOTH_SCALE in statis_dict:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = torch.max(statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE],
                                                                       channel_max)
            else:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return partial(stats_hook)

    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        通用的平滑应用方法

        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        try:
            # 获取子图类型名称（
            subgraph_type = self.SUBGRAPH_TYPE_MAP.get(type(subgraph_obj), "unknown")
            
            # 构建SmoothContext
            smooth_context = self._build_smooth_context(linear_modules)
            if subgraph_type not in self.ASYM_SUPPORT_SUBGRAPH_TYPES:
                shift_value = False
                get_logger().debug("[IterSmoothProcessor] Non-asym subgraph detected, setting shift=False")
            else:
                # 如果symmetric为True，shift取False；如果symmetric为False，shift取True
                shift_value = not self.config.symmetric
                get_logger().debug(f"[IterSmoothProcessor] Non-OV subgraph detected, setting shift={shift_value}")

            # 创建平滑配置
            smooth_quant_cfg = IterSmoothConfig(
                alpha=self.config.alpha,
                shift=shift_value,
                scale_min=self.config.scale_min
            )

            # 应用平滑
            iter_smooth(subgraph_obj, smooth_quant_cfg, smooth_context)
            get_logger().info(
                f"[IterSmoothProcessor] Smooth application completed successfully for "
                f"subgraph type: {subgraph_type}, shift: {shift_value}"
            )

        except Exception as e:
            get_logger().error(f"[IterSmoothProcessor] Failed to apply smooth to subgraph: {e}")
            raise
