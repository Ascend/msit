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


class CmdConst:
    """
    Class for command line const
    """

    PROBE = "probe"
    DUMP = "dump"
    COMPARE = "compare"
    OPCHECK = "opcheck"

    SURGEON = "surgeon"
    LIST = "list"
    EVALUATE = "evaluate"
    OPTIMIZE = "optimize"
    EXTRACT = "extract"
    CONCAT = "concat"

    HELP_MODULE_MAP = {
        PROBE: "A module for diagnosing inference accuracy issues, including data dump, compare, operator check, etc.",
        SURGEON: "Graph scan and modification Tool.",
    }

    HELP_TOOL_MAP = {
        DUMP: "Inference dump tools for Ascend devices.",
        COMPARE: "Accuracy compare tools for msit dump task.",
        OPCHECK: "Operator check tools for msit dump task",
        LIST: "List all knowledge bases that currently support auto-tuning.",
        EVALUATE: "Search for onnx models that can be optimized by a specified knowledge base.",
        OPTIMIZE: "Use the specified knowledge base to optimize the specified onnx.",
        EXTRACT: "Cutting model subgraphs.",
        CONCAT: "Connecting to the model.",
    }


class PathConst:
    """
    Class for file or dir path const
    """

    FILE = "file"
    DIR = "dir"

    SIZE_20M = 20_971_520  # 20 * 1024 * 1024
    SIZE_500M = 524_288_000  # 500 * 1024 * 1024
    SIZE_2G = 2_147_483_648  # 2 * 1024 * 1024 * 1024
    SIZE_10G = 10_737_418_240  # 10 * 1024 * 1024 * 1024
    SIZE_30G = 32_212_254_720  # 30 * 1024 * 1024 * 1024
    SIZE_50G = 53_687_091_200  # 50 * 1024 * 1024 * 1024

    SUFFIX_PY = ".py"
    SUFFIX_SH = ".sh"
    SUFFIX_ONLINE_SCRIPT = (SUFFIX_PY, SUFFIX_SH)
    SUFFIX_PB = ".pb"
    SUFFIX_ONNX = ".onnx"
    SUFFIX_OM = ".om"
    SUFFIX_PROTOTXT = ".prototxt"
    SUFFIX_CAFFEMODEL = ".caffemodel"
    SUFFIX_OFFLINE_MODEL = (SUFFIX_PB, SUFFIX_ONNX, SUFFIX_OM, SUFFIX_PROTOTXT)
    SUFFIX_TXT = ".txt"
    SUFFIX_CONVERT_MODEL = (SUFFIX_OM, SUFFIX_TXT)
    SUFFIX_NPY = ".npy"
    SUFFIX_BIN = ".bin"
    SUFFIX_YAML = ".yaml"
    SUFFIX_JSON = ".json"
    SUFFIX_FSF = (SUFFIX_JSON, ".cfg")
    SUFFIX_CSV = ".csv"
    SUFFIX_SO = ".so"


class MsgConst:
    """
    Class for log messages const
    """

    INVALID_ARGU = "[ERROR] invalid argument."
    INVALID_DATA_TYPE = "[ERROR] invalid data type."
    REQUIRED_ARGU_MISSING = "[ERROR] Required argument missing."
    RISK_ALERT = "[ERROR] Risk alert."
    NO_PERMISSION = "[ERROR] No permission."
    IO_FAILURE = "[ERROR] I/O failure."
    PATH_NOT_FOUND = "[ERROR] Path not found."
    VALUE_NOT_FOUND = "[ERROR] Value not found."
    PARSING_FAILED = "[ERROR] Parsing failed."
    CANN_FAILED = "[ERROR] CANN enabling failed."
    ATTRIBUTE_ERROR = "[ERROR] Attribute not found."
    CALL_FAILED = "[ERROR] Call failed."
    MAX_RECURSION_DEPTH = 5


class CompConst:
    """
    Class for component name const
    """

    ONNX_ACTUATOR_COMP = "OnnxActuatorComp"
    ONNX_DUMPER_COMP = "OnnxDumperComp"
    ONNX_WRITER_COMP = "OnnxWriterComp"

    FROZEN_GRAPH_ACTUATOR_COMP_CPU = "FrozenGraphActuatorCompCPU"
    FROZEN_GRAPH_DUMPER_COMP_CPU = "FrozenGraphDumperCompCPU"
    FROZEN_GRAPH_WRITER_COMP_CPU = "FrozenGraphWriterCompCPU"

    FROZEN_GRAPH_ACTUATOR_COMP_NPU = "FrozenGraphActuatorCompNPU"
    FROZEN_GRAPH_SET_GE_COMP_NPU = "FrozenGraphSetGECompNPU"
    FROZEN_GRAPH_WRITER_COMP_NPU = "FrozenGraphWriterCompNPU"

    CAFFE_ACTUATOR_COMP = "CaffeActuatorComp"
    CAFFE_DUMPER_COMP = "CaffeDumperComp"
    CAFFE_WRITER_COMP = "CaffeWriterComp"

    ATB_ACTUATOR_COMP = "ATBActuatorComp"


class CfgConst:
    """
    Class for config items
    """

    CONFIG_PATH = "config_path"
    TASK = "task"
    TASK_STAT = "statistics"
    TASK_TENSOR = "tensor"
    ALL_TASK = {TASK_STAT, TASK_TENSOR}
    EXEC = "exec"
    FRAMEWORK = "framework"
    FRAMEWORK_MINDIE_LLM = "mindie_llm"
    FRAMEWORK_TORCH_AIR = "torch_air"
    FRAMEWORK_MINDIE_TORCH = "mindie_torch"
    FRAMEWORK_ONNX = "ONNX"
    FRAMEWORK_TF = "TensorFlow"
    FRAMEWORK_OM = "Ascend OM"
    FRAMEWORK_CAFFE = "Caffe"
    ALL_FRAMEWORK = {FRAMEWORK_MINDIE_LLM, FRAMEWORK_TORCH_AIR, FRAMEWORK_MINDIE_TORCH}
    RANK = "rank"
    STEP = "step"
    LEVEL = "level"
    LEVEL_MODULE = "module"
    LEVEL_LAYER = "layer"
    LEVEL_API = "api"
    LEVEL_KERNEL = "kernel"
    ALL_LEVEL = {LEVEL_MODULE, LEVEL_LAYER, LEVEL_API, LEVEL_KERNEL}
    LOG_LEVEL = "log_level"
    SEED = "seed"


class DumpConst:
    """
    Class for dump const
    """

    DEVICE = "device"
    INPUT_ARGS = "input_args"
    OUTPUT_ARGS = "output_args"
    INPUT = "input"
    OUTPUT = "output"
    INPUT_ALL = [INPUT, "all"]
    OUTPUT_ALL = [OUTPUT, "all"]
    ALL_DATA_MODE = [INPUT, OUTPUT, "all"]

    DUMP_PATH = "dump_path"
    LIST = "list"
    DATA_MODE = "data_mode"
    DUMP_EXTRA = "dump_extra"
    ALL_DUMP_EXTRA = ["desc", "tiling", "child_op", "cpu_profiling", "onnx"]
    DUMP_TIME = "dump_time"
    ALL_DUMP_TIME = ["0", "1", "2", "3", 0, 1, 2, 3]
    OP_ID = "op_id"
    DUMP_LAST_LOGITS = "dump_last_logits"
    DUMP_WEIGHT = "dump_weight"
    DUMP_GE_GRAPH = "dump_ge_graph"
    ALL_DUMP_GE_GRAPH = ["1", "2", "3", 1, 2, 3]
    DUMP_GRAPH_LEVEL = "dump_graph_level"
    ALL_DUMP_GRAPH_LEVEL = ["1", "2", "3", "4", 1, 2, 3, 4]
    FUSION_SWITCH_FILE = "fusion_switch_file"
    ONNX_FUSION_switch = "onnx_fusion_switch"
    SAVED_MODEL_TAG = "saved_model_tag"
    SAVED_MODEL_SIGN = "saved_model_signature"
    WEIGHT_PATH = "weight_path"

    DUMP_DATA_DIR = "dump_data_dir"
    DATA = "data"
    DUMP_JSON = "dump.json"
    STACK_JSON = "stack.json"
    NET_OUTPUT_NODES_JSON = "net_output_nodes.json"

    ENVVAR_DUMP_GE_GRAPH = "DUMP_GE_GRAPH"
    ENVVAR_DUMP_GRAPH_LEVEL = "DUMP_GRAPH_LEVEL"
    ENVVAR_DUMP_GRAPH_PATH = "DUMP_GRAPH_PATH"
    ENVVAR_ASCEND_WORK_PATH = "ASCEND_WORK_PATH"

    ENVVAR_MSIT_OUTPUT_DIR = "ATB_OUTPUT_DIR"
    ENVVAR_MSIT_DUMP_TASK = "ATB_DUMP_TASK"
    ENVVAR_MSIT_DUMP_LEVEL = "ATB_DUMP_LEVEL"
    ENVVAR_MSIT_SAVE_TENSOR_RANGE = "ATB_SAVE_TENSOR_RANGE"
    ENVVAR_MSIT_DEVICE_ID = "ATB_DEVICE_ID"
    ENVVAR_MSIT_LOG_LEVEL = "ATB_LOG_LEVEL"
    ENVVAR_MSIT_SAVE_TENSOR = "ATB_SAVE_TENSOR"
    ENVVAR_MSIT_SAVE_TILING = "ATB_SAVE_TILING"
    ENVVAR_MSIT_SAVE_CHILD = "ATB_SAVE_CHILD"
    ENVVAR_MSIT_SAVE_CPU_PROFILING = "ATB_SAVE_CPU_PROFILING"
    ENVVAR_MSIT_SAVE_ONNX = "ATB_SAVE_ONNX"
    ENVVAR_MSIT_SAVE_TENSOR_IN_BEFORE_OUT_AFTER = "ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER"
    ENVVAR_MSIT_SAVE_TENSOR_TIME = "ATB_SAVE_TENSOR_TIME"
    ENVVAR_MSIT_SAVE_TENSOR_IDS = "ATB_SAVE_TENSOR_IDS"
    ENVVAR_MSIT_SAVE_TENSOR_RUNNER = "ATB_SAVE_TENSOR_RUNNER"
