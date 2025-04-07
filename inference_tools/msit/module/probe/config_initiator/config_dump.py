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

from argparse import Namespace

from msit.base import BaseConfig
from msit.common.validation import (
    valid_device,
    valid_dump_extra,
    valid_dump_ge_graph,
    valid_dump_graph_level,
    valid_dump_last_logits,
    valid_dump_mode,
    valid_dump_path,
    valid_dump_time,
    valid_dump_weight,
    valid_exec,
    valid_fusion_switch_file,
    valid_input_json,
    valid_onnx_fusion_switch,
    valid_op_id,
    valid_weight_path,
)
from msit.utils.constants import DumpConst


class DumpConfig(BaseConfig):
    def check_config(self, task: str = None, step: list = None, dump_path: str = None, args: Namespace = None):
        self.common_check(task, step, args)
        self.task_config = self.config[self.config.get(DumpConst.TASK)]
        self.config[self.config.get(DumpConst.TASK)] = self._check_task_dic(dump_path, args)
        return self.config

    def _check_task_dic(self, dump_path: str = None, args=None):
        self._update_config(
            self.task_config, DumpConst.EXEC, args, valid_exec, self.task_config.get(DumpConst.EXEC, [])
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_PATH,
            args,
            valid_dump_path,
            dump_path or self.task_config.get(DumpConst.DUMP_PATH, "./"),
        )
        self._update_config(
            self.task_config, DumpConst.DUMP_MODE, args, valid_dump_mode, self.task_config.get(DumpConst.DUMP_MODE, "")
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_EXTRA,
            args,
            valid_dump_extra,
            self.task_config.get(DumpConst.DUMP_EXTRA, []),
        )
        self._update_config(
            self.task_config, DumpConst.DUMP_TIME, args, valid_dump_time, self.task_config.get(DumpConst.DUMP_TIME, "")
        )
        self._update_config(
            self.task_config, DumpConst.OP_ID, args, valid_op_id, self.task_config.get(DumpConst.OP_ID, [])
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_LAST_LOGITS,
            args,
            valid_dump_last_logits,
            self.task_config.get(DumpConst.DUMP_LAST_LOGITS, False),
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_WEIGHT,
            args,
            valid_dump_weight,
            self.task_config.get(DumpConst.DUMP_WEIGHT, False),
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_GE_GRAPH,
            args,
            valid_dump_ge_graph,
            self.task_config.get(DumpConst.DUMP_GE_GRAPH, "2"),
        )
        self._update_config(
            self.task_config,
            DumpConst.DUMP_GRAPH_LEVEL,
            args,
            valid_dump_graph_level,
            self.task_config.get(DumpConst.DUMP_GRAPH_LEVEL, "3"),
        )
        self._update_config(
            self.task_config,
            DumpConst.FUSION_SWITCH_FILE,
            args,
            valid_fusion_switch_file,
            self.task_config.get(DumpConst.FUSION_SWITCH_FILE, ""),
        )
        self._update_config(
            self.task_config, DumpConst.DEVICE, args, valid_device, self.task_config.get(DumpConst.DEVICE, "")
        )
        self._update_config(
            self.task_config,
            DumpConst.INPUT_JSON,
            args,
            valid_input_json,
            self.task_config.get(DumpConst.INPUT_JSON, ""),
        )
        self._update_config(
            self.task_config,
            DumpConst.ONNX_FUSION_switch,
            args,
            valid_onnx_fusion_switch,
            self.task_config.get(DumpConst.ONNX_FUSION_switch, True),
        )
        self._update_config(
            self.task_config,
            DumpConst.WEIGHT_PATH,
            args,
            valid_weight_path,
            self.task_config.get(DumpConst.WEIGHT_PATH, ""),
        )
        return self.task_config
