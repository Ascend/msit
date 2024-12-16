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

import numpy as np

from msit.utils.toolkits import get_current_time
from msit.common.dependencies import import_tensorflow

tf = import_tensorflow()


class CmdConst:
    """
    Class for command line const
    """
    MSIT = "msit"
    DESCRIPTION = """
    msit (MindStudio Inference Tools), [Powered by MindStudio].
    Providing one-site debugging and optimization toolkits for inference on Ascend devices.
    For any issue, refer FAQ first. Gitee repo: Ascend/msit, wiki.
    """
    COMMAND = "command"
    SUBCOMMAND = "subcommand"
    RUN = "run"

    CMD_LEVEL_2 = 1
    CMD_LEVEL_3 = 2
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

    HELP_PROBE = "A module for diagnosing inference accuracy issues, including data dump, compare, operator check, etc."
    HELP_PROBE_DUMP = "Inference dump tools for Ascend devices."
    HELP_PROBE_COMPARE = "Accuracy compare tools for msit dump task."
    HELP_PROBE_OPCHECK = "Operator check tools for msit dump task"

    HELP_SURGEON = "Graph scan and modification Tool."
    HELP_SURGEON_LIST = "List all knowledge bases that currently support auto-tuning."
    HELP_SURGEON_EVALUATE = "Search for onnx models that can be optimized by a specified knowledge base."
    HELP_SURGEON_OPTIMIZE = "Use the specified knowledge base to optimize the specified onnx."
    HELP_SURGEON_EXTRACT = "Cutting model subgraphs."
    HELP_SURGEON_CONCAT = "Connecting to the model."


class PathConst:
    """
    Class for file or dir path const
    """
    MAX_PATH_LENGTH = 4096
    MAX_LAST_NAME_LENGTH = 255
    MAX_DIR_DEPTH = 32
    VALID_PATH_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    FILE = "file"
    DIR = "dir"

    MODE_READ = ["r", "rb"]
    MODE_WRITE = ["w", "wb", "a", "ab", "a+"]
    MODE_EXEC = "exec"
    MODE = MODE_READ + MODE_WRITE + [MODE_EXEC]
    BINARY_MODE = "b"
    SIZE_20M = 20_971_520  # 20 * 1024 * 1024
    SIZE_500M = 524_288_000  # 500 * 1024 * 1024
    SIZE_10G = 10_737_418_240  # 10 * 1024 * 1024 * 1024
    SIZE_20G = 21_474_836_480  # 20 * 1024 * 1024 * 1024
    SIZE_50G = 53_687_091_200  # 50 * 1024 * 1024 * 1024

    INTERPRETER_PYTHON = "python"
    INTERPRETER_BASH = "bash"
    INTERPRETER = (INTERPRETER_PYTHON, INTERPRETER_BASH)
    SUFFIX_PY = ".py"
    SUFFIX_SH = ".sh"
    SUFFIX_ONLINE_SCRIPT = (SUFFIX_PY, SUFFIX_SH)
    SUFFIX_PB = ".pb"
    SUFFIX_ONNX = ".onnx"
    SUFFIX_OM = ".om"
    SUFFIX_OFFLINE_MODEL = (SUFFIX_PB, SUFFIX_ONNX, SUFFIX_OM)
    SUFFIX_NPY = ".npy"
    SUFFIX_BIN = ".bin"
    SUFFIX_YAML = ".yaml"
    SUFFIX_JSON = ".json"
    SUFFIX_CSV = ".csv"

    CPUEXECUTE = "CPUExecutionProvider"

    AUTHORITY_DIR = 0o750
    AUTHORITY_FILE = 0o640

    SUBDIR_INPUT = "input"
    SUBDIR_MODEL = "model"
    SUBDIR_TENSOR = "tensor"
    SUBDIR_RANK = "rank"
    INDEX = "index"


class MsgConst:
    """
    Class for log messages const
    """
    STAR = "*"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    LOG_LEVEL = [DEBUG, INFO, WARNING, ERROR]

    class LogLevel:

        class DEBUG:
            value = 0

        class INFO:
            value = 1

        class WARNING:
            value = 2

        class ERROR:
            value = 3

    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\v", "\u000B", "%08", "%09", "%0a", "%0b", \
                    "%0c", "%0d", "%7f", "//", "\\", "&"]
    INVALID_ARGU = "[ERROR] invalid argument."
    INVALID_DATA_TYPE = "[ERROR] invalid data type."
    REQUIRED_ARGU_MISSING = "[ERROR] Required argument missing."
    RISK_ALERT = "[ERROR] Risk alert."
    NO_PERMISSION = "[ERROR] No permission."
    IO_FAILURE = "[ERROR] I/O failure."
    PATH_NOT_FOUND = "[ERROR] Path not found."
    TILDE_NUM_PATTERN = r"^\d+(~\d+)*$"
    DSR_PATTERN = r"^[\d,~-]+$"
    DSR_ERROR = 'The format of the dynamic shape range is not "input1:1,3,200~224,224-230."'
    LOAD_ERROR = "Failed to load the path {} using <{}>."
    SAVE_ERROR = "Failed to save {} to {} using <{}>. Please check permissions or disk space."
    INT_BORDER = [0, 1e6]

    TOTAL_CHAR_LENGTH = 80
    INITIALIZED = "initialized"


class CompConst:
    """
    Class for component name const
    """
    ONNXREADER = "OnnxReader"
    ONNXACTUATOR = "OnnxActuator"
    ONNXWRITER = "OnnxWriter"


class DumpConst:
    """
    Class for dump const
    """
    CURRENT_DUMP_FOLDER = f"msit_dump_{get_current_time(used_for_log=False)}/"
    CPU = "cpu"
    NPU = "npu"
    SAVED_MODEL_PB = "saved_model.pb"
    VARIABLES = "variables"

    TYPE2DTYPE_MAP = {
        "tensor(int)": np.int32, 
        "tensor(int8)": np.int8, 
        "tensor(int16)": np.int16, 
        "tensor(int32)": np.int32, 
        "tensor(int64)": np.int64, 
        "tensor(uint8)": np.uint8, 
        "tensor(uint16)": np.uint16, 
        "tensor(uint32)": np.uint32, 
        "tensor(uint64)": np.uint64, 
        "tensor(float)": np.float32, 
        "tensor(float16)": np.float16, 
        "tensor(double)": np.double, 
        "tensor(bool)": np.bool_, 
        "tensor(complex64)": np.complex64, 
        "tensor(complex128)": np.complex_, 
        tf.float16: np.float16, 
        tf.float32: np.float32, 
        tf.float64: np.float64, 
        tf.int8: np.int8, 
        tf.int16: np.int16, 
        tf.int32: np.int32, 
        tf.int64: np.int64,
    }

    NEW_ = "new_"
    MAX_PROTOBUF_2G = 2_147_483_648  # 2 * 1024 * 1024 * 1024

    NAME = "name"
    SHAPE = "shape"
    TYPE = "type"
    DTYPE = "dtype"
    MAX = "Max"
    MIN = "Min"
    MEAN = "Mean"
    NORM = "Norm"

    INPUT_ARGS = "input_args"
    OUTPUT_ARGS = "output_args"
    _1KB = 1024

    TASK = "task"
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    SAVE_OPTION = [TENSOR, OVERFLOW_CHECK]
    LEVEL = "level"
    OPERATOR = "operator"
    DUMP_DATA_DIR = "dump_data_dir"
    DATA = "data"
    DUMP_JSON = "dump.json"
