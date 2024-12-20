# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

from typing import List


class LayerConfigDictSelector(abc.ABC):
    """
    自动选择每层量化配置的基类。子类需要实现select函数用于实现特定的选层逻辑。
    """

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def select(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def match(self, *args, **kwargs) -> bool:
        pass


class LayerConfigSelector:

    @abc.abstractmethod
    def select(self, *args, **kwargs):
        pass


global_selectors: List[LayerConfigDictSelector] = []


def register_selector(selector):
    global global_selectors
    global_selectors.append(selector)
    return


def select_layer_config(*args, **kwargs):
    global global_selectors
    for selector in global_selectors:
        if selector.match(*args, **kwargs):
            selector.setup(*args, **kwargs)
            selector.select(*args, **kwargs)
            break
