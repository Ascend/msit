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

import sys
from abc import ABC, abstractmethod
from inspect import stack

from msit.common.dirs import DirPool
from msit.common.stat import DataStat
from msit.utils.constants import CfgConst, DumpConst, PathConst
from msit.utils.io import save_json, save_npy
from msit.utils.log import get_current_timestamp, logger
from msit.utils.path import MsitPath, join_path
from msit.utils.toolkits import get_valid_name

_SIZE_1M = 1_048_576  # 1024 * 1024
_STACK_FILTER_PATH = ["msit/core", "msit/base", "msit/common", "msit/utils", "torch/nn/modules/module.py"]
_WITHOUT_CALL_STACK = "The call stack retrieval failed."


class WriterDump(ABC):
    def __init__(self, dump_format):
        self.dump_format = dump_format
        self.max_cache_size = _SIZE_1M
        self.cache_dump_json = {}
        self.cache_dump_json_size = 0
        self.dump_json = self.init_dump_json()
        self.cache_stack_json = {}
        self.cache_stack_json_size = 0
        self.tensor_path = ""
        self.net_output_nodes = []

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name == "summ_dump_data" and callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if self.net_output_nodes:
                    save_json(
                        self.net_output_nodes, join_path(DirPool.get_model_dir(), DumpConst.NET_OUTPUT_NODES_JSON)
                    )
                self._flush_remaining_cache()
                return result

            return wrapper
        return attr

    @staticmethod
    def _generate_path(tensor_dir, tensor_name):
        tensor_path = MsitPath(
            join_path(tensor_dir, tensor_name), PathConst.FILE, "w", size_limitation=None, suffix=PathConst.SUFFIX_NPY
        ).check(path_exist=False)
        return tensor_path

    @staticmethod
    def _generate_name(name, in_out, node_id):
        name = ".".join(
            [str(get_current_timestamp(False, True)), get_valid_name(name), in_out, str(node_id) + PathConst.SUFFIX_NPY]
        )
        return name

    @staticmethod
    def _call_stack(name: str):
        try:
            _stack = stack()[:5]
        except Exception as e:
            logger.warning(f"The call stack of {name} failed to retrieve, {e}.")
            _stack = None
        stack_str = []
        if _stack:
            for _, path, line, func, code, _ in _stack:
                if not code:
                    continue
                if any(filter_path in path for filter_path in _STACK_FILTER_PATH):
                    continue
                stack_line = f"File {path}, line {str(line)}, in {func}, \n {code[0].strip()}"
                stack_str.append(stack_line)
        else:
            stack_str.append(_WITHOUT_CALL_STACK)
        stack_info = {name: stack_str}
        return stack_info

    @staticmethod
    def _remove_colon(node_name: str):
        if ":" in node_name:
            return node_name.split(":")[0]
        return node_name

    @abstractmethod
    def summ_dump_data(self):
        pass

    def init_dump_json(self, **kwargs):
        self.cache_dump_json.update(
            {
                CfgConst.LEVEL: kwargs.get(CfgConst.LEVEL, None),
                CfgConst.FRAMEWORK: kwargs.get(CfgConst.FRAMEWORK, None),
                DumpConst.DUMP_DATA_DIR: kwargs.get(DumpConst.DUMP_DATA_DIR, None),
                DumpConst.DATA: {},
            }
        )
        return self.cache_dump_json

    def update_stat(self, name=None, in_out=None, args_name=None, npy_data=None):
        self._update_dump_json(
            self.cache_dump_json[DumpConst.DATA][get_valid_name(name)],
            in_out,
            {**{"name": get_valid_name(args_name)}, **DataStat.collect_stats_for_numpy(npy_data)},
        )

    def update_stack(self, name):
        self.cache_stack_json.update(self._call_stack(get_valid_name(name)))
        self.cache_stack_json_size += sys.getsizeof(self.cache_stack_json)
        if self.cache_stack_json_size >= self.max_cache_size:
            self._save_stack_json()
            self.cache_stack_json_size = 0

    def through_inputs(self, inputs, node_name, input_map):
        for i, input_item in enumerate(inputs):
            input_name = input_item if isinstance(input_item, str) else input_item.name
            mapped_value = input_map.get(get_valid_name(input_name))
            self.update_stat(node_name, DumpConst.INPUT_ARGS, input_name, mapped_value)
            if self.dump_format == DumpConst.DUMP_FORMAT_TENSOR:
                self._save_tensor_data(node_name, DumpConst.INPUT, i, mapped_value)
        logger.debug(f"Processed the input data of {node_name}.")

    def through_outputs(self, outputs, node_name, output_map):
        for i, output_item in enumerate(outputs):
            output_name = output_item if isinstance(output_item, str) else output_item.name
            mapped_value = output_map.get(get_valid_name(output_name))
            self.update_stat(node_name, DumpConst.OUTPUT_ARGS, output_name, mapped_value)
            if self.net_output_nodes and self._remove_colon(output_name) in self.net_output_nodes:
                logger.info(
                    f"net_output node index is: {self.net_output_nodes.index(self._remove_colon(output_name))}, "
                    f"node name: {output_name}."
                )
            if self.dump_format == DumpConst.DUMP_FORMAT_TENSOR:
                self._save_tensor_data(node_name, DumpConst.OUTPUT, i, mapped_value)
        logger.debug(f"Processed the output data of {node_name}.")

    def _flush_remaining_cache(self):
        if self.cache_dump_json_size > 0:
            self._save_dump_json()
            self.cache_dump_json_size = 0
        if self.cache_stack_json_size > 0:
            self._save_stack_json()
            self.cache_stack_json_size = 0

    def _update_dump_json(self, dump_dic, in_out, kwargs: dict):
        if in_out not in dump_dic:
            dump_dic[in_out] = []
        dump_dic.get(in_out).append(kwargs)
        self.cache_dump_json_size += sys.getsizeof(kwargs)
        if self.cache_dump_json_size >= self.max_cache_size:
            self._save_dump_json()
            self.cache_dump_json_size = 0

    def _save_dump_json(self):
        dump_json_path = join_path(DirPool.get_rank_dir(), DumpConst.DUMP_JSON)
        save_json(self.cache_dump_json, dump_json_path, indent=4)

    def _save_stack_json(self):
        stack_json_path = join_path(DirPool.get_rank_dir(), DumpConst.STACK_JSON)
        save_json(self.cache_stack_json, stack_json_path, indent=4)

    def _save_tensor_data(self, name, in_out, ind, npy_data):
        DirPool.make_tensor_dir()
        self.cache_dump_json[DumpConst.DUMP_DATA_DIR] = DirPool.get_tensor_dir()
        file_name = self._generate_name(name, in_out, ind)
        self.tensor_path = self._generate_path(DirPool.get_tensor_dir(), file_name)
        save_npy(npy_data, self.tensor_path)
