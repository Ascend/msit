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

from zlib import crc32

import numpy as np

from msit.utils.log import logger
from msit.utils.toolkits import safely_compute


class DataStat:
    @staticmethod
    def _get_valid_type(np_data):
        try:
            module_name = np_data.__class__.__module__
            class_name = np_data.__class__.__name__
            return f"{module_name}.{class_name}"
        except Exception:
            logger.warning(f"Unrecognized type pattern: {type(np_data)}.")
            return None

    @staticmethod
    @safely_compute
    def _get_dtype(npy):
        return npy.dtype

    @staticmethod
    @safely_compute
    def _get_shape(npy):
        return npy.shape

    @staticmethod
    @safely_compute
    def _get_max(npy):
        return float(npy.max())

    @staticmethod
    @safely_compute
    def _get_min(npy):
        return float(npy.min())

    @staticmethod
    @safely_compute
    def _get_mean(npy):
        return float(npy.mean())

    @staticmethod
    @safely_compute
    def _get_norm(npy):
        return float(np.linalg.norm(npy))

    @staticmethod
    @safely_compute
    def _get_crc32_hash(npy):
        npy_bytes = npy.tobytes()
        crc32_hash = crc32(npy_bytes)
        return f"{crc32_hash:08x}"

    @classmethod
    def collect_stats_for_numpy(cls, npy: np.ndarray):
        stat_dict = {}
        stat_dict["type"] = cls._get_valid_type(npy)
        stat_dict["dtype"] = cls._get_dtype(npy)
        stat_dict["shape"] = cls._get_shape(npy)
        stat_dict["Max"] = cls._get_max(npy)
        stat_dict["Min"] = cls._get_min(npy)
        stat_dict["Mean"] = cls._get_mean(npy)
        stat_dict["Norm"] = cls._get_norm(npy)
        stat_dict["md5"] = cls._get_crc32_hash(npy)
        return stat_dict
