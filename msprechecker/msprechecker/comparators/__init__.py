# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from collections import defaultdict

from ..utils import CompareErrorHandler, ErrorType, get_handler
from ..utils.path_io import to_user_path


class Comparator:
    MISSING_VALUE = "missing"

    def __init__(self, *, error_handler: CompareErrorHandler = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_COMPARE)

    @staticmethod
    def _flatten(data, visited_path, file_path, path=()):
        stack = [(data, path)]

        while stack:
            current_data, current_path = stack.pop()

            if isinstance(current_data, dict):
                for k, v in current_data.items():
                    new_path = current_path + (k,)
                    if k == "npuDeviceIds":
                        visited_path[new_path][file_path] = v
                    else:
                        stack.append((v, new_path))
            elif isinstance(current_data, list):
                for i, v in enumerate(reversed(current_data)):
                    new_path = current_path + (i,)
                    stack.append((v, new_path))
            else:
                visited_path[current_path][file_path] = current_data

    @staticmethod
    def _unflatten(flat_dict):
        root = {}

        for path, value in flat_dict.items():
            node = root
            for i, part in enumerate(path[:-1]):
                next_part = path[i + 1]
                if isinstance(next_part, int):
                    if part not in node:
                        node[part] = [{}]
                    while len(node[part]) <= next_part:
                        node[part].append({})
                else:
                    if part not in node:
                        node[part] = {}
                node = node[part]
            node[path[-1]] = value

        return root

    @staticmethod
    def _all_values_equal(values, all_paths) -> bool:
        ref_value = next(iter(values.values()))

        return all(path in values and values[path] == ref_value for path in all_paths)

    def compare(self, path_to_data):
        flat_diff = {}

        for path, data in path_to_data.items():
            for conf_type, conf_data in data.items():
                flat_diff.setdefault(conf_type, defaultdict(dict))
                conf_diff = flat_diff[conf_type]

                self._flatten(conf_data, conf_diff, path, ())

        filtered_flat_diff = {
            conf_type: self._unflatten(
                {
                    path: {
                        to_user_path(file_path): values.get(file_path, self.MISSING_VALUE) for file_path in path_to_data
                    }
                    for path, values in conf_diff.items()
                    if not self._all_values_equal(values, path_to_data)
                }
            )
            for conf_type, conf_diff in flat_diff.items()
        }

        for collect_type, values in filtered_flat_diff.items():
            if values:
                self.error_handler.add_diff(collect_type, values)

        return self.error_handler
