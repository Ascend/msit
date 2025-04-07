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

from msit.base import Component, Service
from msit.common.dirs import DirPool
from msit.core.probe.base import ServiceDump
from msit.utils.constants import CompConst, DumpConst
from msit.utils.io import savedmodel2pb


@Service.register(DumpConst.STATISTICS)
class ServiceStatistics(ServiceDump):
    pass


@Service.register(DumpConst.TENSOR)
class ServiceTensor(ServiceDump):
    def _construct_for_saved_model_on_npu(self):
        self.cfg.exec[0] = savedmodel2pb(
            self.cfg.exec[0], self.cfg.saved_model_tag, self.cfg.saved_model_signature, DirPool.get_model_dir()
        )
        self._construct_for_frozen_graph_model_on_npu()

    def _construct_for_frozen_graph_model_on_npu(self):
        self.actuator = Component.get(CompConst.FROZEN_GRAPH_ACTUATOR_COMP_NPU)(
            priority=20,
            model_path=self.cfg.exec[0],
            input_shape=self.cfg.input_shape,
            input_path=self.cfg.input_path,
            dump_mode=self.cfg.dump_mode,
            fsf=self.cfg.fusion_switch_file,
        )
        self.setter = Component.get(CompConst.FROZEN_GRAPH_SET_GE_COMP_NPU)(
            priority=10,
            work_path=DirPool.get_msit_dir(),
            dump_ge_graph=self.cfg.dump_ge_graph,
            dump_graph_level=self.cfg.dump_graph_level,
            dump_graph_path=DirPool.get_model_dir(),
        )
