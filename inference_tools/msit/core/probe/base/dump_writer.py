# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import os

from msit.common.log import logger
from msit.common.dirs import DirPool
from msit.common.constants import PathConst, DumpConst
from msit.utils.io import save_json
from msit.utils.path import MsitPath
from msit.utils.toolkits import get_current_time


class DataWriter:
    cache_dump_json = {}
    cache_dump_json_size = 0
    max_cache_size = DumpConst._1KB

    @staticmethod
    def _generate_tensor_path(tensor_dir, tensor_name):
        tensor_path = MsitPath(os.path.join(tensor_dir, tensor_name), PathConst.FILE, "w", size_limitation=None, \
                               suffix=PathConst.SUFFIX_NPY).check()
        return tensor_path

    @staticmethod
    def _to_valid_name(name: str):
        if name and name[0] == "/":
            name = name.lstrip("/")
        return name.replace(".", "_").replace("/", "_")

    @staticmethod
    def _to_valid_type(np_data):
        try:
            module_name = np_data.__class__.__module__
            class_name = np_data.__class__.__name__
            return f"{module_name}.{class_name}"
        except Exception:
            logger.warning(f"Unrecognized type pattern: {type(np_data)}")
            return None

    @classmethod
    def init_dump_json(cls, **kwargs):
        cls.cache_dump_json.update({
            DumpConst.TASK: kwargs.get(DumpConst.TASK, None),
            DumpConst.LEVEL: kwargs.get(DumpConst.LEVEL, None),
            DumpConst.DUMP_DATA_DIR: kwargs.get(DumpConst.DUMP_DATA_DIR, None),
            DumpConst.DATA: {}
        })
        return cls.cache_dump_json

    @classmethod
    def update_dump_json(cls, dump_dic, parent_node, kwargs: dict, last_save=False):
        if parent_node not in dump_dic:
            dump_dic[parent_node] = {}
        dump_dic[parent_node].update(kwargs)
        cls.cache_dump_json_size += len(str(kwargs).encode("utf-8"))
        if cls.cache_dump_json_size >= cls.max_cache_size:
            cls._save_dump_json()
            cls.cache_dump_json_size = 0
        elif last_save:
            cls._save_dump_json()
        else:
            pass

    @classmethod
    def _save_dump_json(cls):
        dump_json_path = os.path.join(DirPool.get_dump_path(), DumpConst.DUMP_JSON)
        save_json(cls.cache_dump_json, dump_json_path, indent=4)

    @classmethod
    def _generate_tensor_name(cls, name, node_id):
        return f"{get_current_time(False, True)}_{cls._to_valid_name(name)}_index{node_id}{PathConst.SUFFIX_NPY}"
