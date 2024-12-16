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

from msit.common.constants import DumpConst


class DataStat:
    @classmethod
    def summ_npy(cls, npy: np.ndarray):
        stat_dict = {}
        stat_dict[DumpConst.DTYPE] = npy.dtype
        stat_dict[DumpConst.SHAPE] = npy.shape
        stat_dict[DumpConst.MAX] = npy.max()
        stat_dict[DumpConst.MIN] = npy.min()
        stat_dict[DumpConst.MEAN] = npy.mean()
        stat_dict[DumpConst.NORM] = np.linalg.norm(npy)
        return stat_dict
