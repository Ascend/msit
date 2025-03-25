# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import argparse

from components.utils.file_open_check import FileStat
from components.utils.log import logger


def check_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path can not be written. Please check.")
    return path_value


def get_algorithm_path():
    try:
        cann_path = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
        algorithm_path = os.path.join(cann_path, "tools", "operator_cmp", "load_balancing")
        if not os.path.exists(algorithm_path):
            raise FileNotFoundError(f"Algorithm path does not exist: {algorithm_path}")
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + algorithm_path
    except KeyError as e:
        logger.error(f"Environment variable error: {e}")
    except Exception as e:
        logger.error(f"Error setting paths: {e}")