# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
import re 
import json 

from typing import Optional
import torch 

from components.debug.compare.utils.base_dump_reader import DumpFileReader


class TorchDumpFileReader(DumpFileReader):
    def __init__(self, cpu_path: str, json_path: str):
        self.path = cpu_path
        self.json_path = json_path 
        self.key_to_folder = self._map_keys_to_folders()

    def get_tensor(self, key: str) -> torch.Tensor:
        cpu_tensor = None 
        folder_name = self.key_to_folder[key]
        key_with_root = f'root.{folder_name}'
        folder_path = os.path.join(self.path, key_with_root)
        for file_name in os.listdir(folder_path):
            if file_name.startswith('output'):
                key_path = os.path.join(folder_path, file_name)
                cpu_tensor = torch.load(key_path)
                return cpu_tensor
            else:
                continue 

        return cpu_tensor

    def _filter_keys(self, key_to_fold: dict) -> dict:
        keys = list(key_to_fold.values())
        keys.sort(key=len, reverse=True)
        filtered_keys = set()
        for key in keys:
            is_contained = any(other_key.startswith(key + '.') for other_key in keys if other_key != key)
            if not is_contained:
                filtered_keys.add(key)

        filtered_key_to_folder = {fusion_op: key for fusion_op, key in key_to_fold.items() if key in filtered_keys}

        return filtered_key_to_folder

    def _map_keys_to_folders(self) -> dict:
        key_to_folder = {}
        key_to_id = {}
        json_path = os.path.join(self.json_path, 'op_map_updated.json')

        with open(json_path, 'r') as f:
            data = json.load(f)
            for fusion_op, details in data.items():
                id_ = details.get('id', float('inf'))
                jit_node = details.get('jit_node', '')
                if jit_node:
                    key = self._extract_key_from_jit_node(jit_node)
                    if key is not None:
                        key_to_folder[fusion_op] = key
                        key_to_id[key] = id_

        key_to_folder = self._filter_keys(key_to_folder)
        self.key_to_id = key_to_id

        return key_to_folder

    def _extract_key_from_jit_node(self, jit_node: str) -> Optional[str]:
        # Here is an example of jit_node: "%968 : Float(1, 64, 56, 56) = aten::relu(%input.9), 
        # scope: __module.layer1/__module.layer1.0/__module.layer1.0.relu # 
        # /home/site-packages/torch/nn/functional.py:1469:0"
        key_none = None
        if len(jit_node.split("scope:")) < 2:
            return key_none
        if "#" in jit_node:
            jit_node_front = jit_node.split("#")[0]
            if jit_node_front:
                jit_node_front = jit_node_front.strip()
                jit_node_front_list = jit_node_front.rsplit("__module.", 1)
                if jit_node_front_list:
                    key_none = jit_node_front_list[-1]
        return key_none

    def _get_keys(self) -> set:
        return set(self.key_to_folder.keys())
    