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

from msit.common.log import logger
from msit.common.cli import MsitCommand
from msit.common.constants import CmdConst, DumpConst
from msit.core.probe.service.dump import Service
from msit.core.probe.cli.args_validation import CheckExec, CheckDumpPath, CheckRankorStep, CheckInputShape, \
  CheckInputPath, CheckDymShapeRange


class DumpCommand(MsitCommand):
    def __init__(self):
        super().__init__(prog_name=CmdConst.DUMP, help_info=CmdConst.HELP_PROBE_DUMP)

    @staticmethod
    def add_required_arguments(parser):
        req = parser.add_argument_group("Required arguments")
        req.add_argument(
            "-e", "--exec", dest="exec", nargs="+", action=CheckExec, required=True,
            help="""<str> If `--exec xxx` ends with .sh or .py, execute an online dump. Default: None
        If it ends with .pb/saved_model (TensorFlow), .onnx (ONNX), or .om (Ascend model), execute an offline dump."""
        )

    @staticmethod
    def add_optional_arguments(parser):
        opt = parser.add_argument_group("Optional arguments")
        opt.add_argument(
            "-path", "--dump-path", dest="dump_path", type=str, action=CheckDumpPath, default="./", \
            help=f'<str> Directory for data dump. Default: {"./"}'
        )
        opt.add_argument(
            "-task", "--dump-task", dest="task", type=str, default=DumpConst.STATISTICS, 
            choices=[DumpConst.STATISTICS, DumpConst.TENSOR, DumpConst.OVERFLOW_CHECK], \
            help=f"<str> Select a task to perform the dump. Default: {DumpConst.STATISTICS}"
        )
        opt.add_argument(
            "-logl", "--log-level", dest="log_level", type=str, default="info", \
            choices=["debug", "info", "warning", "error"], \
            help="<str> Set the logging level (e.g., debug, info, warning, error). Default: info"
        )
        opt.add_argument(
            "-s", "--seed", dest="seed", type=int, help="<int> Random seed. Default: None"
        )

    @staticmethod
    def add_online_model_arguments(parser):
        onl_args = parser.add_argument_group("Supports only online model arguments")
        onl_args.add_argument(
            "-rank", "--device-id", dest="rank", type=str, nargs="+", default=[], action=CheckRankorStep, \
            help="<list> Device ID for dumping data. Default: []"
        )
        onl_args.add_argument(
            "-step", "--execute-round", dest="step", type=str, nargs="+", default=[], action=CheckRankorStep, \
            help="<list> Execution rounds for dumping data. Default: []"
        )
        onl_args.add_argument(
            "-mode", "--dump-mode", dest="dump_mode", type=str, default="all", \
            choices=["input", "output", "all"], help="""<str> The portion to dump tensors. Default: all
            input: dump input tensor
            output: dump output tensor
            all: dump input tensor and output tensor"""
        )
        onl_args.add_argument(
            "-level", "--dump-level", dest="dump_level", type=str, nargs="+", default=["operator"], \
            help="""<list> Dump level, include: model, module, layer, operator, kernel. Default: ["operator"]
            Note that different frameworks support different levels:
                - ATB: model, layer, operator, kernel;
                - PyTorch: module, operator."""
        )
        onl_args.add_argument(
            "-list", "--dump-list", dest="dump_list", type=str, nargs="+", default=[], \
            help="<list> Dump only specific modules, layers, ops, kernels, or tokens. Default: []"
        )

    @staticmethod
    def add_atb_model_arguments(parser):
        atb_args = parser.add_argument_group("Supports only Ascend Transformer Boost (ATB) arguments")
        atb_args.add_argument(
            "-extra", "--dump-extra", dest="dump_extra", type=str, nargs="+", default=[], \
            choices=["desc", "tiling", "child_op", "cpu_profiling", "onnx"], \
            help="""<list> Whether dump tensor's description information, tiling data, child operators, 
        cpu_profiling and ONNX graph. Default: []"""
        )
        atb_args.add_argument(
            "-time", "--dump-time", dest="dump_time", type=int, default=3, choices=[0, 1, 2, 3], \
            help="""<int> Timing of tensor dumping. Default: 3
        0: dump tensors before execution
        1: dump tensors after execution
        2: dump tensors before and after execution
        3: dump input tensors before execution and output tensors after execution"""
        )

    @staticmethod
    def add_pytorch_arguments(parser):
        torch_args = parser.add_argument_group("Supports only PyTorch arguments")
        torch_args.add_argument(
            "-llogits", "--dump-last-logits", dest="dump_last_logits", action="store_true", default=False, \
            help="<bool> Whether dump last logits. Default: False"
        )
        torch_args.add_argument(
            "-weight", "--dump-weight", dest="dump_weight", action="store_true", default=False, \
            help="<bool> Whether dump weight. Default: False"
        )

    @staticmethod
    def add_acl_dump_arguments(parser):
        torchair_args = parser.add_argument_group("Supports only Torch-air arguments")
        torchair_args.add_argument(
            "-fsj", "--fusion-switch-json", dest="fusion_switch_json", type=str, \
            help="<str> A JSON file Whether dump fusion data in GE mode. Default: None"
        )

    @staticmethod
    def add_offline_model_arguments(parser):
        offline_args = parser.add_argument_group("Supports only offline model arguments")
        offline_args.add_argument(
            "-d", "--device", dest="device", type=str, default=DumpConst.CPU, choices=[DumpConst.CPU, DumpConst.NPU], \
            help="""<str> Device type, supporting CPU and NPU; currently, 
        only saved_model format is supported on NPU. Default: cpu"""
        )
        offline_args.add_argument(
            "-inpath", "--input-path", dest="input_path", nargs="+", default=[], action=CheckInputPath, \
            help="""<list> Model input path supports .npy and .bin. Default: [] e.g., -i ./input_0.npy ./input_1.bin"""
        )
        offline_args.add_argument(
            "-inshape", "--input-shape", dest="input_shape", nargs="+", default={}, action=CheckInputShape, \
            help="""<dict> The shape of the model input file. Default: {}
        e.g., -ins input_0:1,224,224,3 input_1:5,224,224,6 """
        )
        offline_args.add_argument(
            "-dshape", "--dym-shape-range", dest="dym_shape_range", nargs="+", default={}, action=CheckDymShapeRange, \
            help="""<dict> Threshold range for dynamic shapes. Default: {}
        if set, shapes will be dumped sequentially based on the list of shapes provided in the parameter. """
        )
        offline_args.add_argument(
            "-ofs", "--onnx-fusion-switch", dest="onnx_fusion_switch", action="store_false", default=True, \
            help="<bool> ONNX Runtime operator fusion switch. Default: True"
        )
        offline_args.add_argument(
            "-sign", "--saved-model-signature", dest="saved_model_signature", type=str, default="serving_default", \
            help="""<str> Required signature for loading a saved_model. Default: serving_default"""
        )
        offline_args.add_argument(
            "-tfj", "--tensorflow-json", dest="tensorflow_json", type=str, \
            help="""<str> A JSON file for the operator set to dump saved_model data on CPU; 
        required for CPU dumps. Default: None"""
        )

    @staticmethod 
    def execute(args):
        logger.set_level(args.log_level)
        Service(args).run()

    def add_arguments(self, parser):
        self.add_required_arguments(parser)
        self.add_optional_arguments(parser)
        self.add_online_model_arguments(parser)
        self.add_atb_model_arguments(parser)
        self.add_pytorch_arguments(parser)
        self.add_acl_dump_arguments(parser)
        self.add_offline_model_arguments(parser)
        parser.set_defaults(run=DumpCommand.execute)
