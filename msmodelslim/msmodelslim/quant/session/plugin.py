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
import importlib.util
import os
import sys
import argparse
from abc import abstractmethod
from typing import Any, Optional

from torch import nn

from msmodelslim.quant.processor.quant.w8a8 import W8A8ProcessorConfig
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig


class QuantPlugin:
    """量化插件基类

    该类定义了量化插件必须实现的接口。插件开发者需要继承此类并实现必要的方法
    来完成模型加载、校准数据加载等操作。
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def load_model(self) -> nn.Module:
        """加载模型

        Returns:
            nn.Module: 加载的模型实例
        """
        raise NotImplementedError("Plugin must implement load_model method")

    @abstractmethod
    def load_calib_data(self) -> Any:
        """加载校准数据

        Returns:
            Any: 校准数据，具体类型由插件实现决定
        """
        raise NotImplementedError("Plugin must implement load_calib_data method")

    def load_quant_cfg(self, default_cfg: W8A8ProcessorConfig) -> W8A8ProcessorConfig:
        """加载配置

        Args:
            default_cfg: 默认配置
        """
        _ = self
        return default_cfg

    def get_save_cfg(self, default_cfg: Optional[SaverProcessorConfig] = None) -> Optional[SaverProcessorConfig]:
        """获取保存配置

        Args:
            default_cfg: 默认保存配置

        Returns:
            Optional[SaverProcessorConfig]: 保存配置，如果返回None则使用默认配置
        """
        _ = self
        return default_cfg
    
    def eval_model(self):
        """评估模型
        """
        _ = self
        pass


def load_plugin(plugin_path: str, args: argparse.Namespace) -> Optional[QuantPlugin]:
    """
    从指定的Python文件加载插件

    Args:
        plugin_path: 插件文件路径

    Returns:
        加载的插件实例
    """
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Plugin file not found: {plugin_path}")

    # 获取模块名（不含.py后缀）
    module_name = os.path.splitext(os.path.basename(plugin_path))[0]

    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load plugin from {plugin_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 查找并实例化插件类
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and
                issubclass(attr, QuantPlugin) and
                attr != QuantPlugin):
            return attr(args)

    raise ImportError(f"No valid plugin class found in {plugin_path}")
