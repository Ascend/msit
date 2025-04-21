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

from msit.common.dirs import DirPool
from msit.utils.constants import CfgConst, DumpConst, MsgConst, PathConst
from msit.utils.env import evars
from msit.utils.exceptions import MsitException
from msit.utils.log import logger
from msit.utils.path import is_enough_disk_space
from msit.utils.toolkits import run_subprocess, seed_all


class AtbModelConfiguration:
    def __init__(self, dump_path, **kwargs):
        self.dump_path = dump_path
        self.task = kwargs.get("task", CfgConst.TASK_STAT)
        self.dump_level = kwargs.get("dump_level", [CfgConst.LEVEL_KERNEL])
        self.step = kwargs.get("step", [])
        self.rank = kwargs.get("rank", [])
        self.seed = kwargs.get("seed")
        self.log_level = kwargs.get("log_level")
        self.dump_extra = kwargs.get("dump_extra", [])
        self.dump_time = kwargs.get("dump_time", "3")
        self.op_id = kwargs.get("op_id", [])
        self.op_name = kwargs.get("op_name", "")
        self.exec = kwargs.get("exec", [])

    def set_env_vars(self):
        self._set_dump_path()
        self._set_task()
        self._set_dump_level()
        self._set_step()
        self._set_rank()
        self._set_seed()
        self._set_log_level()
        self._set_dump_extra()
        self._set_dump_time()
        self._set_op_id()
        self._set_op_name()
        logger.info("The ATB dump parameters have been set.")

    def execute_dump(self):
        if not is_enough_disk_space(DirPool.get_msit_dir(), PathConst.SIZE_2G):
            raise MsitException(MsgConst.RISK_ALERT, "Please reserve at least 2GB of disk space for saving dump data.")
        run_subprocess(self.exec)

    def _set_dump_path(self):
        evars.set(DumpConst.ENVVAR_MSIT_OUTPUT_DIR, self.dump_path)

    def _set_task(self):
        evars.set(DumpConst.ENVVAR_MSIT_DUMP_TASK, self.task)

    def _set_dump_level(self):
        evars.set(DumpConst.ENVVAR_MSIT_DUMP_LEVEL, ",".join(self.dump_level))

    def _set_step(self):
        evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_RANGE, ",".join([str(i) for i in self.step]))

    def _set_rank(self):
        if self.rank:
            evars.set(DumpConst.ENVVAR_MSIT_DEVICE_ID, ",".join([str(i) for i in self.rank]))
        else:
            evars.delete(DumpConst.ENVVAR_MSIT_DEVICE_ID)

    def _set_seed(self):
        if self.seed:
            seed_all(self.seed)

    def _set_log_level(self):
        evars.set(DumpConst, logger.get_level_id(self.log_level))

    def _set_dump_extra(self):
        options = {
            "desc": DumpConst.ENVVAR_MSIT_SAVE_TENSOR,
            "tiling": DumpConst.ENVVAR_MSIT_SAVE_TILING,
            "child_op": DumpConst.ENVVAR_MSIT_SAVE_CHILD,
            "cpu_profiling": DumpConst.ENVVAR_MSIT_SAVE_CPU_PROFILING,
            "onnx": DumpConst.ENVVAR_MSIT_SAVE_ONNX,
        }
        for key, env_var in options.items():
            evars.set(env_var, "1" if key in self.dump_extra else "0")

    def _set_dump_time(self):
        if self.dump_time == "3":
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_IN_BEFORE_OUT_AFTER, "1")
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_TIME, "1")
        else:
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_IN_BEFORE_OUT_AFTER, "0")
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_TIME, self.dump_time)

    def _set_op_id(self):
        if self.op_id:
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_IDS, ",".join([str(i) for i in self.op_id]))
        else:
            evars.delete(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_IDS)

    def _set_op_name(self):
        if self.op_name:
            evars.set(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_RUNNER, str(self.op_name).lower())
        else:
            evars.delete(DumpConst.ENVVAR_MSIT_SAVE_TENSOR_RUNNER)
