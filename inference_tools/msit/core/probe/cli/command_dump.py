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

from msit.base import MsitCommand
from msit.common.validation import (
    CheckDevice,
    CheckDumpExtra,
    CheckDumpGeGraph,
    CheckDumpGraphLevel,
    CheckDumpLevel,
    CheckDumpMode,
    CheckDumpPath,
    CheckDumpTask,
    CheckDumpTime,
    CheckExec,
    CheckFusionSwitchFile,
    CheckInputJson,
    CheckLogLevel,
    CheckOpId,
    CheckSeed,
    CheckStepOrRank,
    CheckWeightPath,
)
from msit.utils.constants import CmdConst, DumpConst


class DumpCommand(MsitCommand):
    def __init__(self):
        super().__init__(prog_name=CmdConst.DUMP, help_info=CmdConst.HELP_PROBE_DUMP)

    @staticmethod
    def add_required_arguments(parser):
        req = parser.add_argument_group("Required arguments")
        req.add_argument(
            "-e",
            "--exec",
            dest=DumpConst.EXEC,
            nargs="+",
            action=CheckExec,
            required=True,
            help="""<str> The `--exec` option supports passing either an executable command line or 
        a file for an offline model. Offline models should have extensions such as .pb/saved_model (TensorFlow), 
        .onnx (ONNX), .om (Ascend model), or .prototxt (Caffe). Default: None""",
        )

    @staticmethod
    def add_optional_arguments(parser):
        opt = parser.add_argument_group("Optional arguments")
        opt.add_argument(
            "-o",
            "--dump-path",
            dest=DumpConst.DUMP_PATH,
            action=CheckDumpPath,
            default="./",
            help=f'<str> Directory for data dump. Default: {"./"}',
        )
        opt.add_argument(
            "-task",
            "--dump-task",
            dest=DumpConst.TASK,
            action=CheckDumpTask,
            default=DumpConst.STATISTICS,
            help=f"<str> Select a task to perform the dump. Default: {DumpConst.STATISTICS}",
        )
        opt.add_argument(
            "-level",
            "--dump-level",
            dest=DumpConst.LEVEL,
            action=CheckDumpLevel,
            nargs="+",
            default=["api"],
            help="""<list> Dump level, include: module, layer, api, kernel. Default: ["api"]
        Note: The offline model data dump only supports kernel level.""",
        )
        opt.add_argument(
            "-mode",
            "--dump-mode",
            dest=DumpConst.DUMP_MODE,
            action=CheckDumpMode,
            default="all",
            help="""<str> The portion to dump tensors. Default: all
        input: dump input tensor
        output: dump output tensor
        all: dump input and output tensor""",
        )
        opt.add_argument(
            "-logl",
            "--log-level",
            dest=DumpConst.LOG_LEVEL,
            default="info",
            action=CheckLogLevel,
            help="<str> Set the logging level (e.g., debug, info, warning, error). Default: info",
        )
        opt.add_argument(
            "-s", "--seed", dest=DumpConst.SEED, type=int, action=CheckSeed, help="<int> Random seed. Default: None"
        )

    @staticmethod
    def add_online_model_arguments(parser):
        onl_args = parser.add_argument_group("Supports only online model arguments")
        onl_args.add_argument(
            "-step",
            "--execute-round",
            dest=DumpConst.STEP,
            nargs="+",
            default=[],
            action=CheckStepOrRank,
            help="<list> Execution rounds for dumping data. Default: []",
        )
        onl_args.add_argument(
            "-rank",
            "--device-id",
            dest=DumpConst.RANK,
            nargs="+",
            default=[],
            action=CheckStepOrRank,
            help="<list> Device ID for dumping data. Default: []",
        )
        onl_args.add_argument(
            "-list",
            "--dump-list",
            dest=DumpConst.LIST,
            nargs="+",
            default=[],
            help="""<list> Determine the level from the `--dump-level`, 
        then dump the specific module, layer, API, or kernel. Default: []""",
        )

    @staticmethod
    def add_atb_model_arguments(parser):
        atb_args = parser.add_argument_group("Supports only Ascend Transformer Boost (ATB) arguments")
        atb_args.add_argument(
            "-extra",
            "--dump-extra",
            dest=DumpConst.DUMP_EXTRA,
            nargs="+",
            default=[],
            action=CheckDumpExtra,
            help="""<list> Whether dump desc (tensor's description information), tiling (tiling data), 
        child_op (child operators), cpu_profiling and onnx (ONNX graph). Default: []""",
        )
        atb_args.add_argument(
            "-time",
            "--dump-time",
            dest=DumpConst.DUMP_TIME,
            default="3",
            action=CheckDumpTime,
            help="""<int> Timing of tensor dumping. Default: 3
        0: dump tensors before execution
        1: dump tensors after execution
        2: dump tensors before and after execution
        3: dump input tensors before execution and output tensors after execution""",
        )
        atb_args.add_argument(
            "-op-id",
            "--operation-id",
            dest=DumpConst.OP_ID,
            nargs="+",
            default=[],
            action=CheckOpId,
            help="""<list> Set the operator ID for data dump. Default: []
        e.g., 2, 3_1, or 3_1_2. """,
        )

    @staticmethod
    def add_pytorch_arguments(parser):
        torch_args = parser.add_argument_group("Supports only PyTorch arguments")
        torch_args.add_argument(
            "-llogits",
            "--dump-last-logits",
            dest=DumpConst.DUMP_LAST_LOGITS,
            default=False,
            action="store_true",
            help="<bool> Whether dump last logits. Default: False",
        )
        torch_args.add_argument(
            "-w",
            "--dump-weight",
            dest=DumpConst.DUMP_WEIGHT,
            default=False,
            action="store_true",
            help="<bool> Whether dump weight. Default: False",
        )

    @staticmethod
    def add_acl_dump_arguments(parser):
        ge_args = parser.add_argument_group("Supports only GE dump arguments")
        ge_args.add_argument(
            "-geg",
            "--dump-ge-graph",
            dest=DumpConst.DUMP_GE_GRAPH,
            default="2",
            action=CheckDumpGeGraph,
            help="""<int> Control GE graph dump scope. Default: 2
        1: Full dump with edges and data info;
        2: Basic dump without weights or extra data;
        3: Minimal dump showing only node relationships.""",
        )
        ge_args.add_argument(
            "-gegl",
            "--dump-graph-level",
            dest=DumpConst.DUMP_GRAPH_LEVEL,
            default="3",
            action=CheckDumpGraphLevel,
            help="""<int> Control GE graph dump count. Default: 3
        1: Dump all graphs;
        2: Dump all except subgraphs;
        3: Dump the final generated graph after GE optimization and compilation;
        4: Dump the earliest generated graph after operator parsing and mapping, 
            before compilation and optimization.""",
        )
        ge_args.add_argument(
            "-fsf",
            "--fusion-switch-file",
            dest=DumpConst.FUSION_SWITCH_FILE,
            action=CheckFusionSwitchFile,
            help="<str> A JSON file that configures whether to enable or disable fusion pass switch. Default: None",
        )

    @staticmethod
    def add_offline_model_arguments(parser):
        offline_args = parser.add_argument_group("Supports only offline model arguments")
        offline_args.add_argument(
            "-d",
            "--device",
            dest=DumpConst.DEVICE,
            action=CheckDevice,
            help="""<str> Device type, supporting CPU and NPU. Default: None
        Currently only saved_model, .pb and .om format are supported on NPU. """,
        )
        offline_args.add_argument(
            "-injson",
            "--input-json",
            dest=DumpConst.INPUT_JSON,
            default="",
            action=CheckInputJson,
            help="""<str> A formatted JSON file for model inference, with the following structure: 
        [
            {"name": "xx1", "dym_shape": [], "shape": [1, 2], "path": "xxx.npy"}, 
            {"name": "xx2", "dym_shape": ["1~3",3], "shape": [], "path": "xxx.bin"}, 
            {"name": "xx3", "dym_shape": ["1~3~2",3], "shape": [], "path": "xxx.bin"}
        ]
        Default: "".""",
        )
        offline_args.add_argument(
            "-ofs",
            "--onnx-fusion-switch",
            dest=DumpConst.ONNX_FUSION_switch,
            action="store_false",
            default=True,
            help="<bool> ONNX Runtime operator fusion switch. Default: True",
        )
        offline_args.add_argument(
            "-tag",
            "--saved-model-tag",
            dest="saved_model_tag",
            nargs="+",
            default=["serve"],
            help='<list> In TensorFlow v2.6.5, tags used for loading a model. Default: ["serve"]',
        )
        offline_args.add_argument(
            "-sign",
            "--saved-model-signature",
            dest="saved_model_signature",
            default="serving_default",
            help="""<str> Required signature for loading a saved_model. Default: serving_default""",
        )
        offline_args.add_argument(
            "-weight",
            "--weight-path",
            dest="weight_path",
            default="",
            action=CheckWeightPath,
            help="""<str> The option must be configured when loading a Caffe model. Default: "" """,
        )

    def register(self, subparser):
        parser = subparser.add_parser(self.prog_name, help=self.help_info, formatter_class=self.formatter_class)
        self.add_required_arguments(parser)
        self.add_optional_arguments(parser)
        self.add_online_model_arguments(parser)
        self.add_atb_model_arguments(parser)
        self.add_pytorch_arguments(parser)
        self.add_acl_dump_arguments(parser)
        self.add_offline_model_arguments(parser)
