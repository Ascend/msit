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
from functools import wraps

from msit.common.exceptions import MsitException


def temporary_tf_log_level(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_log_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 在导入TF包前，修改TF日志等级，仅打印warning和error
        res = func(*args, **kwargs)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = original_log_level
        return res 
    return wrapper


@temporary_tf_log_level
def import_tensorflow():
    try:
        import tensorflow as tf
        if tf.__version__ != "2.6.5":
            raise MsitException("[ERROR] Incompatible versions.", "Currently only supports TensorFlow v2.6.5.")
        return tf
    except ImportError:
        return None


def import_torch():
    try:
        import torch as tch
        return tch
    except ImportError:
        return None


def get_rank_for_pytorch():
    if import_torch().distributed.is_initialized():
        return import_torch().distributed.get_rank()
    return ""


def import_torch_npu():
    try:
        import torch_npu as tch_npu
        return tch_npu
    except ImportError:
        return None
