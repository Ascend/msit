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

from tqdm import tqdm

from msit.base import BaseService, Component, Dict2Class, Service
from msit.common.dirs import DirPool
from msit.module.probe.components.dumper_offline_model import OfflineModelActuatorComp
from msit.module.probe.config_initiator import DumpConfig
from msit.utils.constants import CfgConst, CmdConst, CompConst, MsgConst, PathConst
from msit.utils.exceptions import MsitException
from msit.utils.io import savedmodel2pb
from msit.utils.log import logger, print_log_with_star
from msit.utils.path import get_name_and_ext, is_file, is_saved_model_scene


@Service.register(CmdConst.DUMP)
class ServiceDump(BaseService):
    def __init__(self, config_path="", step=None, dump_path="", args=None):
        super().__init__()
        self.current_iter = 0
        if config_path or dump_path:
            self.is_from_cmd = False
            config = DumpConfig(config_path).check_config(dump_path, step)
        elif args:
            self.is_from_cmd = True
            config = DumpConfig(args.config).check_config(args=args)
        else:
            raise MsitException(
                MsgConst.REQUIRED_ARGU_MISSING,
                "When using MSIT's Python interface for data dump, 'config_path' or 'dump_path' must be set.",
            )
        self.cfg = Dict2Class(config)
        logger.set_level(self.cfg.log_level)
        DirPool.make_msit_dir(self.cfg.dump_path)
        DirPool.make_model_dir()

    @property
    def _is_offline_model(self):
        if len(self.cfg.exec) == 1:
            if is_file(self.cfg.exec[0]) and get_name_and_ext(self.cfg.exec[0])[1] in PathConst.SUFFIX_OFFLINE_MODEL:
                return True
            elif is_saved_model_scene(self.cfg.exec[0]):
                return True
            else:
                return False
        return False

    @property
    def _offline_model_comps_map(self):
        model_map_for_cpu = {
            PathConst.SUFFIX_ONNX: "_construct_for_onnx_model",
            PathConst.SUFFIX_PB: "_construct_for_frozen_graph_model_on_cpu",
            "saved_model": "_construct_for_saved_model_on_cpu",
            PathConst.SUFFIX_PROTOTXT: "_construct_for_caffe_model",
        }
        model_map_for_npu = {
            PathConst.SUFFIX_OM: "_construct_for_om_model",
            PathConst.SUFFIX_PB: "_construct_for_frozen_graph_model_on_npu",
            "saved_model": "_construct_for_saved_model_on_npu",
        }
        device_handlers = {"cpu": model_map_for_cpu, "npu": model_map_for_npu}
        return device_handlers

    @property
    def _online_model_comps_map(self):
        framework_handlers = {CfgConst.FRAMEWORK_MINDIE_LLM: "_construct_for_atb_model"}
        return framework_handlers

    def init_start(self):
        print_log_with_star(f"Launching {self.cfg.service} service...")
        DirPool.make_step_dir(self.current_iter)
        DirPool.make_rank_dir()

    def ignore_actuator(self, attr):
        if not self.is_from_cmd and isinstance(attr, OfflineModelActuatorComp):
            self.comps.remove(attr)

    def finalize_start(self):
        print_log_with_star(f"{self.cfg.service} completed successfully.")

    def step(self, *args, **kwargs):
        self.current_iter += 1

    def stop(self, *args, **kwargs):
        pass

    def construct(self):
        if self._is_offline_model:
            device_handler = self._offline_model_comps_map.get(self.cfg.device)
            if not device_handler:
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    '"device" must be set to either "cpu" or "npu" when dumping the offline model.',
                )
            exec_type = self.cfg.exec[0]
            model_key = (
                "saved_model"
                if is_saved_model_scene(exec_type)
                else next((key for key in device_handler if exec_type.endswith(key)), None)
            )
            handler_name = device_handler.get(model_key)
        else:
            handler_name = self._online_model_comps_map.get(self.cfg.framework)
        if handler_name:
            getattr(self, handler_name)()

    def run_cli(self):
        if self._is_offline_model:
            if isinstance(self.cfg.input_shape, list) and len(self.cfg.input_shape) > 1:
                for inshape in tqdm(self.cfg.input_shape, desc="Processing"):
                    self.cfg.input_shape = inshape
                    self.start()
                    self.step()
            else:
                self.start()
        else:
            self.start()

    def _construct_for_om_model(self):
        self.actuator = Component.get(CompConst.OM_ACTUATOR_COMP)(
            priority=20,
            model_path=self.cfg.exec[0],
            input_shape=self.cfg.input_shape,
            input_path=self.cfg.input_path,
            device_id=self.cfg.rank[0],
        )

    def _construct_for_atb_model(self):
        self.actuator = Component.get(CompConst.ATB_ACTUATOR_COMP)(
            priority=100,
            dump_path=DirPool.get_msit_dir(),
            dump_format=self.cfg.dump_format,
            dump_level=self.cfg.level,
            step=self.cfg.step,
            rank=self.cfg.rank,
            seed=self.cfg.seed,
            log_level=self.cfg.log_level,
            dump_extra=self.cfg.dump_extra,
            dump_time=self.cfg.dump_time,
            op_id=self.cfg.op_id,
            op_name=self.cfg.list,
            exec=self.cfg.exec,
        )

    def _construct_for_onnx_model(self):
        self.actuator = Component.get(CompConst.ONNX_ACTUATOR_COMP)(
            priority=20,
            model_path=self.cfg.exec[0],
            input_shape=self.cfg.input_shape,
            input_path=self.cfg.input_path,
            onnx_fusion_switch=self.cfg.onnx_fusion_switch,
        )
        self.dumper = Component.get(CompConst.ONNX_DUMPER_COMP)(priority=10)
        self.writer = Component.get(CompConst.ONNX_WRITER_COMP)(
            priority=15, dump_format=self.cfg.dump_format, dump_mode=self.cfg.dump_mode
        )
        self.writer.subscribe(self.dumper)

    def _construct_for_caffe_model(self):
        self.actuator = Component.get(CompConst.CAFFE_ACTUATOR_COMP)(
            priority=20,
            model_path=self.cfg.exec[0],
            input_shape=self.cfg.input_shape,
            input_path=self.cfg.input_path,
            weight_path=self.cfg.weight_path,
        )
        self.dumper = Component.get(CompConst.CAFFE_DUMPER_COMP)(priority=10)
        self.writer = Component.get(CompConst.CAFFE_WRITER_COMP)(
            priority=15, dump_format=self.cfg.dump_format, dump_mode=self.cfg.dump_mode
        )
        self.writer.subscribe(self.dumper)

    def _construct_for_frozen_graph_model_on_cpu(self):
        self.actuator = Component.get(CompConst.FROZEN_GRAPH_ACTUATOR_COMP_CPU)(
            priority=20, model_path=self.cfg.exec[0], input_shape=self.cfg.input_shape, input_path=self.cfg.input_path
        )
        self.dumper = Component.get(CompConst.FROZEN_GRAPH_DUMPER_COMP_CPU)(priority=10)
        self.writer = Component.get(CompConst.FROZEN_GRAPH_WRITER_COMP_CPU)(
            priority=15, dump_format=self.cfg.dump_format, dump_mode=self.cfg.dump_mode
        )
        self.writer.subscribe(self.dumper)

    def _construct_for_saved_model_on_cpu(self):
        self.cfg.exec[0] = savedmodel2pb(
            self.cfg.exec[0], self.cfg.saved_model_tag, self.cfg.saved_model_signature, DirPool.get_model_dir()
        )
        self._construct_for_frozen_graph_model_on_cpu()

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
