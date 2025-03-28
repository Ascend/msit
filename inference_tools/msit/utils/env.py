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

from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger


class EnvVarManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EnvVarManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.prefix = ""

    @staticmethod
    def _log(msg):
        logger.debug(msg)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get(self, key, default=None, cast_type=None, required=True):
        full_key = f"{self.prefix}{key}"
        value = os.environ.get(full_key, default)
        self._log(f"Accessed environment variable {full_key}, Value: {value}.")
        if required and value is None:
            raise MsitException(
                MsgConst.REQUIRED_ARGU_MISSING,
                f"Environment variable {key} is required but not set. "
                f"Please check the current environment configuration by `echo ${key}`.",
            )
        if value is not None and cast_type:
            try:
                value = cast_type(value)
                self._log(f"Casted {full_key} to {cast_type.__name__}, Result: {value}.")
            except Exception as e:
                raise MsitException(
                    MsgConst.INVALID_DATA_TYPE, f"Failed to cast environment variable {key} to {cast_type}."
                ) from e
        return value

    def set(self, key, value):
        full_key = f"{self.prefix}{key}"
        os.environ[full_key] = str(value)
        self._log(f"Set environment variable {full_key} to {value}.")

    def delete(self, key):
        full_key = f"{self.prefix}{key}"
        if full_key in os.environ:
            os.environ.pop(full_key, None)
            self._log(f"Deleted environment variable {full_key}.")
        else:
            self._log(f"{full_key} not found to delete.")

    def list_all(self):
        if self.prefix:
            filtered_env = {k: v for k, v in os.environ.items() if k.startswith(self.prefix)}
            self._log(f"Listed environment variables with prefix {self.prefix}: {filtered_env}.")
            return filtered_env
        else:
            self._log(f"Listed all environment variables: {dict(os.environ)}.")
            return dict(os.environ)


evars = EnvVarManager()
