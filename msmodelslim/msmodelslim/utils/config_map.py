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

import fnmatch
from collections.abc import Mapping
from typing import Dict, Any, Generic, TypeVar

T = TypeVar('T')


class ConfigMap(Generic[T], Mapping):
    def __init__(self, cfg_map: Dict[str, T]):
        self.cfg_map: Dict[str, Any] = cfg_map if cfg_map is not None else {}

    def __getitem__(self, key: str) -> T:
        if key in self.cfg_map:
            return self.cfg_map[key]
        for pattern in self.cfg_map:
            if fnmatch.fnmatchcase(key, pattern):
                return self.cfg_map[pattern]
        raise KeyError(f"Key '{key}' not found in config map")

    def __contains__(self, key: str) -> bool:
        if key in self.cfg_map:
            return True
        for pattern in self.cfg_map:
            if fnmatch.fnmatchcase(key, pattern):
                return True
        return False

    def __iter__(self):
        return iter(self.cfg_map)

    def __len__(self):
        return len(self.cfg_map)
