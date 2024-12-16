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

from msit.base.component.manager import Scheduler, Component
from msit.common.log import logger, print_log_with_star
from msit.common.constants import DumpConst, PathConst, CompConst


class Service:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _is_saved_model_scene(exec: list):
        saved_model_pb = os.path.join(exec[0], DumpConst.SAVED_MODEL_PB)
        if not os.path.isfile(saved_model_pb):
            return False
        variables_dir = os.path.join(exec[0], DumpConst.VARIABLES)
        return os.path.isdir(variables_dir)

    @staticmethod
    def _get_suffix(exec: list):
        _, extension = os.path.splitext(exec[0])
        return extension

    def run(self):
        print_log_with_star(f"The currently executing dump task is {self.args.task}.")
        if len(self.args.exec) == 2:
            pass
        elif len(self.args.exec) == 1:
            logger.info("Start offline model data dump...")
            if self.args.device == DumpConst.CPU:
                self._on_cpu()
            elif self.args.device == DumpConst.NPU:
                self._on_npu()
        print_log_with_star("msit probe dump completed successfully.")

    def _rivet(self, read_comp, execute_comp, writer_comp):
        reader = Component.get(read_comp)(self.args)
        execute = Component.get(execute_comp)(self.args)
        writer = Component.get(writer_comp)(self.args)
        execute.subscribe(reader)
        writer.subscribe(execute)
        scheduler = Scheduler()
        scheduler.add([reader, execute, writer])

    def _on_cpu(self):
        logger.info("Deploy data dump task on the CPU.")
        if self._is_saved_model_scene(self.args.exec):
            return None
        elif self._get_suffix(self.args.exec) == PathConst.SUFFIX_PB:
            return None
        elif self._get_suffix(self.args.exec) == PathConst.SUFFIX_ONNX:
            return self._rivet(CompConst.ONNXREADER, CompConst.ONNXACTUATOR, CompConst.ONNXWRITER)
        else:
            return None

    def _on_npu(self):
        logger.info("Deploy data dump task on the NPU.")
        if self._is_saved_model_scene(self.args.exec):
            return None
        elif self._get_suffix(self.args.exec) == PathConst.SUFFIX_OM:
            return None
        else:
            return None
