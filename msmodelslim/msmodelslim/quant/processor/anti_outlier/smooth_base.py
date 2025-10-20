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

from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from msmodelslim.core.QAL.qtypes import (
    LinearLinearSubgraph,
    NormLinearSubgraph,
    SmoothContext,
    UpDownSubgraph,
    OVSubgraph
)
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.quant.processor.anti_outlier.fused_linear import VirtualVModuleFromQKVFused, VirtualVModuleFromKVFused
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import MisbehaviorError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger


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


class BaseSmoothProcessorConfig(AutoProcessorConfig):
    type: Literal["base_smooth"] = "base_smooth"
    alpha: Optional[float] = None  # 可选参数
    beta: Optional[float] = None
    scale_min: Optional[float] = None
    symmetric: Optional[bool] = None
    enable_subgraph_type: Optional[List[str]] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    # 子图处理优先级配置（数字越小优先级越高）
    subgraph_priority: Dict[str, int] = Field(
        default_factory=lambda: {"up-down": 1, "ov": 2, "norm-linear": 3, "linear-linear": 4},
        frozen=True  # 单独为subgraph_priority字段启用frozen
    )

    @model_validator(mode='before')
    @classmethod
    def validate_no_fixed_overrides(cls, data: Any):
        if isinstance(data, dict):
            fixed_fields = {'subgraph_priority'}
            attempted_overrides = fixed_fields.intersection(data.keys())
            if attempted_overrides:
                get_logger().warning(
                    f"Fields {attempted_overrides} have fixed values and cannot be overridden. "
                    f"Using default subgraph_priority configuration: "
                    f"up-down(1) -> ov(2) -> norm-linear(3) -> linear-linear(4)"
                )
                # 移除尝试覆盖的字段，使用默认值
                for override_field in attempted_overrides:
                    data.pop(override_field, None)
        return data


class BaseSmoothProcessor(AutoSessionProcessor):
    # 子图类型到处理方法的映射表
    SUBGRAPH_HANDLERS = {
        "norm-linear": "_apply_norm_linear_smooth",
        "linear-linear": "_apply_linear_linear_smooth",
        "ov": "_apply_ov_smooth",
        "up-down": "_apply_up_down_smooth",
    }

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: Optional[Any] = None):
        super().__init__(model)
        self.adapter = adapter
        self.config = config
        self.act_stats = {}
        self.hook_handles = {}
        self.global_adapter_config = None
        self.adapter_config = None

    def validate_parameters(self):
        """
        验证所有参数的合法性
        """
        # 验证 enable_subgraph_type 中的元素
        valid_subgraph_types = ["norm-linear", "linear-linear", "ov", "up-down"]
        for subgraph_type in self.config.enable_subgraph_type:
            if subgraph_type not in valid_subgraph_types:
                raise SchemaValidateError(
                    f"Elements in enable_subgraph_type must be in {valid_subgraph_types}, "
                    f"current element: {subgraph_type}",
                    action=f"Please use only valid subgraph types: {valid_subgraph_types}")

    def get_num_attention_heads(self):
        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads", "heads_num"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise MisbehaviorError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or modify the config file"
            )
        return num_attention_heads

    def get_num_key_value_heads(self):
        # 如果不存在num_key_value_heads，则设置为num_attention_heads的值
        if not hasattr(self.model.config, "num_key_value_heads"):
            num_key_value_heads = self.model.config.num_attention_heads
            get_logger().warning("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
        else:
            num_key_value_heads = self.model.config.num_key_value_heads

        if not num_key_value_heads:
            raise MisbehaviorError(
                f"the config of model must have num_key_value_heads, \
                                please check or modify the config file"
            )
        return num_key_value_heads

    def is_data_free(self) -> bool:
        _ = self
        return False

    def pre_run(self) -> None:
        """在运行前获取全图子结构配置"""
        self.global_adapter_config = self.adapter.get_adapter_config_for_subgraph()
        get_logger().info(f"Loaded global subgraph configurations")

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 根据当前模块过滤全局配置
        self.adapter_config = self._filter_adapter_configs_by_config(
            self.global_adapter_config,
            self.config,
            request.name
        )
        get_logger().debug(f"Processed {len(self.adapter_config)} subgraphs for submodule {request.name}")
        self._install_statis_hook(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._apply_smooth(request.name, request.module)

    def _filter_adapter_configs_by_config(
            self,
            adapter_configs: List[AdapterConfig],
            config: AutoProcessorConfig,
            scope: str
    ) -> List[AdapterConfig]:
        """
        根据配置过滤适配器配置
        
        Args:
            adapter_configs: 适配器配置列表
            config: 处理器配置
            scope: 作用域前缀
            
        Returns:
            List[AdapterConfig]: 过滤后的适配器配置列表
        """
        result = []
        layer_prefix = f"{scope}." if scope != "" else ""

        include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        for adapter_config in adapter_configs:
            # 1. 检查子图类型是否支持
            if adapter_config.subgraph_type not in config.enable_subgraph_type:
                continue

            # 2. 检查映射配置是否存在
            if not adapter_config.mapping:
                continue

            source_name = adapter_config.mapping.source

            # 3. 检查是否以layer_prefix为前缀
            if not source_name.startswith(layer_prefix):
                continue

            if source_name not in include:
                continue

            if source_name in exclude:
                continue

            result.append(adapter_config)

        return result

    def _build_smooth_context(self, linear_modules: List[nn.Linear]) -> SmoothContext:
        """
        构建 SmoothContext
        
        Args:
            linear_modules: 线性模块列表，用于获取激活统计信息
            
        Returns:
            SmoothContext: 平滑上下文
        """
        a_smooth_scale = None
        shift = None
        tensors = None

        for linear_module in linear_modules:
            # 获取模块名称
            module_name = None
            for name, module in self.model.named_modules():
                if module is linear_module:
                    module_name = name
                    break

            if module_name is None:
                get_logger().warning(f"Cannot find module name for {linear_module}")
                continue

            # 获取激活统计信息
            if module_name in self.act_stats:
                stats = self.act_stats[module_name]

                # 获取 smooth_scale
                if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                    a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]
                else:
                    raise MisbehaviorError(f"smooth_scale is not found for module {module_name}")

                # 获取 shift
                if StatKey.STAT_KEY_SHIFT in stats:
                    shift = stats[StatKey.STAT_KEY_SHIFT]
                else:
                    shift = None

                if StatKey.TENSOR in stats:
                    tensors = stats[StatKey.TENSOR]
                else:
                    tensors = None
                
                # 找到第一个有效的数据后就可以退出循环
                break

        # 检查是否成功获取到激活平滑尺度
        if a_smooth_scale is None:
            raise MisbehaviorError(
                "Failed to get activation smooth scale from any linear module",
                action="Please ensure that statistics collection hooks are properly installed "
                       "and data collection is completed"
            )

        w_smooth_scale = torch.ones_like(a_smooth_scale)

        # 创建扩展信息
        ext = {}

        # 创建 SmoothContext
        smooth_context = SmoothContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            w_smooth_scale=w_smooth_scale,
            tensors=tensors,
            shift=shift,
            ext=ext
        )

        return smooth_context

    def _get_stats_hook(self, name: str) -> Callable:
        """
        获取统计钩子的抽象方法，子类必须实现
        
        Args:
            name: 模块名称
            
        Returns:
            Callable: 统计钩子函数
        """
        raise NotImplementedError("Subclasses must implement _get_stats_hook method")

    def _get_target_names_for_hook(self, adapter_config: AdapterConfig) -> List[str]:
        """
        根据子图类型获取需要安装钩子的模块名称列表
        
        Args:
            adapter_config: 适配器配置
            
        Returns:
            List[str]: 目标模块名称列表
        """
        subgraph_type = adapter_config.subgraph_type

        if subgraph_type == "norm-linear" or subgraph_type == "up-down":
            # Norm-Linear: 为所有target_names安装钩子
            return adapter_config.mapping.targets

        if subgraph_type == "linear-linear" or subgraph_type == "ov":
            # Linear-Linear: 为linear2模块安装钩子
            target_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ''
            return [target_name] if target_name else []

        # 默认情况：返回空列表
        return []

    def _install_hook_for_module(self, module_name: str, subgraph_type: str = None) -> None:
        """
        为指定模块安装统计钩子
        
        Args:
            module_name: 模块名称
            subgraph_type: 子图类型
        """
        try:
            module = self.model.get_submodule(module_name)
            if isinstance(module, nn.Linear):
                # 保存hook句柄，用于后续删除
                hook_handle = module.register_forward_hook(self._get_stats_hook(module_name, subgraph_type))
                self.hook_handles[module_name] = hook_handle
                get_logger().debug(f"Successfully installed statistics hook for module {module_name} "
                                 f"(subgraph_type: {subgraph_type})")
            else:
                get_logger().warning(f"Module {module_name} is not Linear type, skipping hook installation")
        except Exception as e:
            get_logger().warning(f"Failed to install statistics hook for module {module_name}: {e}")

    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        """
        为所有子图中的linear模块安装统计钩子
        
        Args:
            name: 模块名称
            module: 目标模块
        """
        for adapter_config in self.adapter_config:
            # 根据子图类型获取需要安装钩子的模块名称
            target_names = self._get_target_names_for_hook(adapter_config)

            # 为每个目标模块安装钩子，传递子图类型信息
            for target_name in target_names:
                if target_name:
                    self._install_hook_for_module(target_name, adapter_config.subgraph_type)

    def _remove_all_hooks(self) -> None:
        """
        删除所有已安装的hook
        """
        for module_name, hook_handle in self.hook_handles.items():
            try:
                hook_handle.remove()
                get_logger().debug(f"Successfully removed hook for module {module_name}")
            except Exception as e:
                get_logger().warning(f"Failed to remove hook for module {module_name}: {e}")

        # 清空hook句柄字典
        self.hook_handles.clear()

    def _apply_norm_linear_smooth(self, adapter_config: AdapterConfig) -> None:
        """应用Norm-Linear平滑"""
        # 获取模块对象
        source_module = self.model.get_submodule(adapter_config.mapping.source)
        target_modules = [self.model.get_submodule(name) for name in adapter_config.mapping.targets]
        target_modules = [m for m in target_modules if m is not None]

        if not source_module or not target_modules:
            get_logger().warning(f"Failed to get modules for norm-linear subgraph: {adapter_config.mapping.source}")
            return

        # 应用平滑
        self._apply_smooth_to_subgraph(
            NormLinearSubgraph(source_module, target_modules),
            target_modules
        )

    def _apply_linear_linear_smooth(self, adapter_config: AdapterConfig) -> None:
        """应用Linear-Linear平滑"""
        # 获取模块对象
        source_module = self.model.get_submodule(adapter_config.mapping.source)
        target_modules = [self.model.get_submodule(name) for name in adapter_config.mapping.targets]
        target_modules = [m for m in target_modules if m is not None]

        if not source_module or not target_modules:
            get_logger().warning(f"Failed to get modules for linear-linear subgraph: {adapter_config.mapping.source}")
            return

        # 获取第一个目标模块（Linear-Linear通常只有一个目标）
        target_module = target_modules[0]

        # 应用平滑
        self._apply_smooth_to_subgraph(
            LinearLinearSubgraph(source_module, target_module),
            [target_module]
        )

    def _apply_ov_smooth(self, adapter_config: AdapterConfig) -> None:
        """应用OV平滑（输出-值投影）"""
        # 获取融合配置
        fusion = adapter_config.fusion
        fusion_flag = fusion is not None and fusion.fusion_type != "none"

        try:
            # 根据融合标志选择平滑方法
            if fusion_flag:
                self._apply_qkv_fusion_smooth(adapter_config)
            else:
                self._apply_standard_ov_smooth(adapter_config)

        except Exception as e:
            get_logger().error(f"Error occurred while applying OV smoothing: {e}")

    def _apply_qkv_fusion_smooth(self, adapter_config: AdapterConfig) -> None:
        """
        应用QKV融合平滑
        
        Args:
            adapter_config: 适配器配置
        """
        v_name = adapter_config.mapping.source
        o_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ''
        v_module = self.model.get_submodule(v_name)
        o_module = self.model.get_submodule(o_name)
        fusion = adapter_config.fusion

        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V module {v_name} is not Linear type, skipping QKV fusion")
            return

        if not o_module:
            get_logger().warning(f"O module {o_name} not found, skipping QKV fusion")
            return

        # 创建虚拟V模块
        if fusion.fusion_type == "kv":
            virtual_v_module = VirtualVModuleFromKVFused(v_module,
                                                         num_attention_heads=fusion.num_attention_heads,
                                                         qk_nope_head_dim=fusion.custom_config["qk_nope_head_dim"],
                                                         v_head_dim=fusion.custom_config["v_head_dim"])
        elif fusion.fusion_type == "qkv":
            virtual_v_module = VirtualVModuleFromQKVFused(v_module,
                                                          num_attention_heads=fusion.num_attention_heads,
                                                          num_key_value_heads=fusion.num_key_value_heads)
        else:
            raise UnsupportedError(f"Unsupported fusion type: {fusion.fusion_type}")

        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=virtual_v_module,
                num_attention_heads=fusion.num_attention_heads,
                key_value_heads=fusion.num_key_value_heads
            ),
            [o_module]
        )

        # 更新原始QKV模块权重
        virtual_v_module.update_weights()

        get_logger().debug(f"Successfully applied QKV fusion smoothing: {v_name} -> {o_name}")

    def _apply_standard_ov_smooth(self, adapter_config: AdapterConfig) -> None:
        """
        应用标准OV平滑
        
        Args:
            adapter_config: 适配器配置
        """
        v_name = adapter_config.mapping.source
        o_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ''
        v_module = self.model.get_submodule(v_name)
        o_module = self.model.get_submodule(o_name)

        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V module {v_name} is not Linear type, skipping standard OV smoothing")
            return

        if not o_module:
            get_logger().warning(f"O module {o_name} not found, skipping standard OV smoothing")
            return

        # 获取注意力头数量
        num_attention_heads = self.get_num_attention_heads()
        num_key_value_heads = self.get_num_key_value_heads()

        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads
            ),
            [o_module]
        )

        get_logger().debug(f"Successfully applied standard OV smoothing: {v_name} -> {o_name}")

    def _apply_up_down_smooth(self, adapter_config: AdapterConfig) -> None:
        """应用Up-Down平滑（MLP门控机制）"""
        # 获取模块对象
        up_module = self.model.get_submodule(adapter_config.mapping.source)
        target_modules = [self.model.get_submodule(name) for name in adapter_config.mapping.targets]
        target_modules = [m for m in target_modules if m is not None]

        if not up_module or not target_modules:
            get_logger().warning(f"Failed to get modules for up-down subgraph: {adapter_config.mapping.source}")
            return

        # 获取Up-Down特定的模块
        down_module = target_modules[0] if len(target_modules) > 0 else None
        gate_module = target_modules[1] if len(target_modules) > 1 else None

        if not all([up_module, down_module]):
            get_logger().warning(
                f"Up-Down subgraph missing necessary module information: "
                f"up={up_module is not None}, down={down_module is not None}")
            return

        # 应用平滑
        self._apply_smooth_to_subgraph(
            UpDownSubgraph(up_module, down_module, gate_module),
            [down_module]
        )

    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        应用平滑到子图
        
        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        raise NotImplementedError("Subclasses must implement _apply_smooth_to_subgraph method")

    def _process_single_subgraph(self, adapter_config: AdapterConfig) -> None:
        """
        处理单个子图
        
        Args:
            adapter_config: 适配器配置
        """
        subgraph_type = adapter_config.subgraph_type

        get_logger().debug(
            f"Processing subgraph type: {subgraph_type}, source: {adapter_config.mapping.source}"
        )

        # 根据子图类型调用相应的处理方法
        handler_name = self.SUBGRAPH_HANDLERS.get(subgraph_type)
        if handler_name:
            handler = getattr(self, handler_name)
            handler(adapter_config)
        else:
            get_logger().warning(f"Unsupported subgraph type: {subgraph_type}")

    def _process_subgraphs_by_priority(self) -> None:
        """
        按优先级顺序处理子图
        
        优先级顺序（可配置）：
        1. up-down (最高优先级，MLP门控机制)
        2. ov (中等优先级，注意力机制)
        3. norm-linear (较低优先级，归一化层)
        4. linear-linear (最低优先级，线性层)
        """
        # 使用配置中的优先级设置
        priority_order = self.config.subgraph_priority

        # 按优先级排序子图
        sorted_adapter_configs = sorted(
            self.adapter_config,
            key=lambda x: priority_order.get(x.subgraph_type, 999)  # 未知类型优先级最低
        )

        get_logger().debug(f"Subgraph processing order after priority sorting:")
        for i, adapter_config in enumerate(sorted_adapter_configs):
            priority = priority_order.get(adapter_config.subgraph_type, 999)
            get_logger().debug(
                f"  {i + 1}. {adapter_config.subgraph_type} (priority: {priority}) - {adapter_config.mapping.source}")

        # 按优先级顺序处理子图
        for adapter_config in sorted_adapter_configs:
            try:
                priority = priority_order.get(adapter_config.subgraph_type, 999)
                self._process_single_subgraph(adapter_config)
            except Exception as e:
                get_logger().error(f"Error occurred while processing subgraph {adapter_config.mapping.source}: {e}")
                continue

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        get_logger().debug(f"Starting smoothing application to module: {name}")

        # 按优先级顺序处理子图：up-down -> ov -> norm-linear -> linear-linear
        self._process_subgraphs_by_priority()

        # 清理统计信息
        self.act_stats.clear()

        # 删除所有已安装的hook
        self._remove_all_hooks()

        get_logger().debug(f"Completed smoothing, cleared statistics and hooks")
