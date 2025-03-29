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
from importlib import import_module

from msit.utils.exceptions import MsitException
from msit.utils.log import logger

import_warnings_shown = set()


def safely_import(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            dependency = args[1]
            if dependency not in import_warnings_shown:
                logger.warning(f"{dependency} is not installed. Please install it if needed.")
                import_warnings_shown.add(dependency)
            return None

    return wrapper


def temporary_tf_log_level(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_log_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只打印 warning、error
        result = func(*args, **kwargs)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = original_log_level
        return result

    return wrapper


class DependencyManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DependencyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._dependencies = {}

    def get(self, package_name):
        return self._dependencies.get(package_name, self._import_package(package_name))

    def get_tensorflow(self):
        tf = self.get("tensorflow")
        re_writer_config = self.get("tensorflow/RewriterConfig")
        sm2pb = self.get("tensorflow/convert_variables_to_constants")
        return tf, re_writer_config, sm2pb

    @safely_import
    def _import_package(self, package_name):
        if package_name in self._dependencies:
            return self._dependencies[package_name]
        if package_name == "tensorflow":
            return self._import_tensorflow()
        module = import_module(package_name)
        self._dependencies[package_name] = module
        return module

    @temporary_tf_log_level
    def _import_tensorflow(self):
        module = import_module("tensorflow")
        if module.__version__ != "2.6.5":
            raise MsitException("[ERROR] Incompatible versions. Currently only supports TensorFlow v2.6.5.")
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        from tensorflow.python.framework.graph_util import convert_variables_to_constants

        self._dependencies["tensorflow/convert_variables_to_constants"] = convert_variables_to_constants
        self._dependencies["tensorflow/RewriterConfig"] = RewriterConfig
        self._dependencies["tensorflow"] = module
        return module


dependent = DependencyManager()
