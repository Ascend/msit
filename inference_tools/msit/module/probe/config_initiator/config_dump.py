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
from msit.module.probe.config_initiator.validate_params import (
    valid_device,
    valid_dump_extra,
    valid_dump_format,
    valid_dump_ge_graph,
    valid_dump_graph_level,
    valid_dump_last_logits,
    valid_dump_mode,
    valid_dump_path,
    valid_dump_time,
    valid_dump_weight,
    valid_fusion_switch_file,
    valid_input,
    valid_list,
    valid_onnx_fusion_switch,
    valid_op_id,
    valid_saved_model_signature,
    valid_saved_model_tag,
    valid_weight_path,
)
from msit.utils.constants import CfgConst, DumpConst


class DumpConfig(BaseConfig):
    def check_config(self, dump_path: str = None, step: list = None, args: Namespace = None):
        self.common_check(step, args)
        self.dump_config = self.config[self.config.get(CfgConst.SERVICE)]
        self.config[self.config.get(CfgConst.SERVICE)] = self._check_dump_dic(dump_path)
        return self.config

    def _check_dump_dic(self, dump_path: str = None):
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_PATH,
            valid_dump_path,
            dump_path or self.dump_config.get(DumpConst.DUMP_PATH, "./"),
        )
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_FORMAT,
            valid_dump_format,
            self.dump_config.get(DumpConst.DUMP_FORMAT, "stat"),
        )
        self._update_config(self.dump_config, DumpConst.LIST, valid_list, self.dump_config.get(DumpConst.LIST, {}))
        self._update_config(
            self.dump_config, DumpConst.DUMP_MODE, valid_dump_mode, self.dump_config.get(DumpConst.DUMP_MODE, "all")
        )
        self._update_config(
            self.dump_config, DumpConst.DUMP_EXTRA, valid_dump_extra, self.dump_config.get(DumpConst.DUMP_EXTRA, [])
        )
        self._update_config(
            self.dump_config, DumpConst.DUMP_TIME, valid_dump_time, self.dump_config.get(DumpConst.DUMP_TIME, "3")
        )
        self._update_config(self.dump_config, DumpConst.OP_ID, valid_op_id, self.dump_config.get(DumpConst.OP_ID, []))
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_LAST_LOGITS,
            valid_dump_last_logits,
            self.dump_config.get(DumpConst.DUMP_LAST_LOGITS, False),
        )
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_WEIGHT,
            valid_dump_weight,
            self.dump_config.get(DumpConst.DUMP_WEIGHT, False),
        )
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_GE_GRAPH,
            valid_dump_ge_graph,
            self.dump_config.get(DumpConst.DUMP_GE_GRAPH, "2"),
        )
        self._update_config(
            self.dump_config,
            DumpConst.DUMP_GRAPH_LEVEL,
            valid_dump_graph_level,
            self.dump_config.get(DumpConst.DUMP_GRAPH_LEVEL, "3"),
        )
        self._update_config(
            self.dump_config,
            DumpConst.FUSION_SWITCH_FILE,
            valid_fusion_switch_file,
            self.dump_config.get(DumpConst.FUSION_SWITCH_FILE, ""),
        )
        self._update_config(
            self.dump_config, DumpConst.DEVICE, valid_device, self.dump_config.get(DumpConst.DEVICE, "")
        )
        self._update_config(self.dump_config, DumpConst.INPUT, valid_input, self.dump_config.get(DumpConst.INPUT, []))
        self._update_config(
            self.dump_config,
            DumpConst.ONNX_FUSION_switch,
            valid_onnx_fusion_switch,
            self.dump_config.get(DumpConst.ONNX_FUSION_switch, True),
        )
        self._update_config(
            self.dump_config,
            DumpConst.SAVED_MODEL_TAG,
            valid_saved_model_tag,
            self.dump_config.get(DumpConst.SAVED_MODEL_TAG, ["serve"]),
        )
        self._update_config(
            self.dump_config,
            DumpConst.SAVED_MODEL_SIGN,
            valid_saved_model_signature,
            self.dump_config.get(DumpConst.SAVED_MODEL_SIGN, "serving_default"),
        )
        self._update_config(
            self.dump_config, DumpConst.WEIGHT_PATH, valid_weight_path, self.dump_config.get(DumpConst.WEIGHT_PATH, "")
        )
        return self.dump_config
