# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

import datetime
import os

import numpy as np
import pandas as pd
import torch

from msit_llm.common.tool import read_atb_data
from msit_llm.common.utils import load_file_to_read_common_check
from msit_llm.common.constant import (TOKEN_ID, DATA_ID, GOLDEN_DATA_PATH, MY_DATA_PATH,
                                      CMP_FAIL_REASON, GOLDEN_DTYPE, GOLDEN_SHAPE,
                                      GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE,
                                      GOLDEN_MEAN_VALUE, MY_DTYPE, MY_SHAPE,
                                      MY_MAX_VALUE, MY_MIN_VALUE, MY_MEAN_VALUE,
                                      CSV_GOLDEN_HEADER, GLOBAL_HISTORY_AIT_DUMP_PATH_LIST)
from msit_llm.common.log import logger
from components.utils.cmp_algorithm import CMP_ALG_MAP, CUSTOM_ALG_MAP

MIN_LAYER_NUMBER = 10


class BasicDataInfo:
    count_data_id = 0  # Count data_id, increment by 1 every time creating a new instance
    TORCH_UNSUPPORTED_D_TYPE_MAP = {"uint16": "int32", "uint32": "int64"}

    def __init__(self, golden_data_path, my_data_path, token_id=None, data_id=None):
        self.my_data_path, self.golden_data_path = my_data_path, golden_data_path
        self.token_id = self.get_token_id(my_data_path) if token_id is None else token_id
        self.data_id = self.count_data_id if data_id is None else data_id
        self._count()

    @classmethod
    def _count(cls):
        cls.count_data_id += 1

    def to_dict(self):
        return {
            TOKEN_ID: str(self.token_id),
            DATA_ID: str(self.data_id),
            GOLDEN_DATA_PATH: self.golden_data_path,
            MY_DATA_PATH: self.my_data_path,
        }

    def get_token_id(self, cur_path):
        dump_filename_idx = 4
        dump_tensor_idx = 3
        dirseg = cur_path.split(os.path.sep)
        if len(dirseg) < dump_filename_idx:
            return 0
        flag1 = dirseg[-dump_tensor_idx] == "tensors" or dirseg[-dump_tensor_idx] == "torch_tensors"
        flag2 = any([dirseg[-dump_filename_idx].startswith(x) for x in GLOBAL_HISTORY_AIT_DUMP_PATH_LIST])
        if flag1 and flag2:
            try:
                token_id = int(dirseg[-1])
            except (IndexError, AttributeError, TypeError, ValueError) as e:
                msg = f"get_token_id error, dirseg: {dirseg}, error: {e}"
                logger.debug(msg)
                token_id = 0
        else:
            token_id = self.get_token_id(os.path.dirname(cur_path))
        return token_id


def fill_row_data(data_info: BasicDataInfo, loaded_my_data=None, loaded_golden_data=None, is_broadcast_tensor=False):
    # 第三个参数“is_broadcast_tensor”用于两个模型batch size不一致时将低维的tensor广播到高维进行比较
    # 创建一条比较数据
    golden_data_path, my_data_path = data_info.golden_data_path, data_info.my_data_path
    logger.debug(f"[fill_row_data], golden_data_path: {golden_data_path}, my_data_path: {my_data_path}")
    row_data = data_info.to_dict()
    if loaded_golden_data is None and not os.path.isfile(golden_data_path):
        row_data[CMP_FAIL_REASON] = f"golden_data_path: {golden_data_path} is not a file."
        return row_data
    if loaded_my_data is None and not os.path.isfile(my_data_path):
        row_data[CMP_FAIL_REASON] = f"my_data_path: {my_data_path} is not a file."
        return row_data
    golden_data = load_as_torch_tensor(golden_data_path, loaded_golden_data)
    my_data = load_as_torch_tensor(my_data_path, loaded_my_data)

    if is_broadcast_tensor:
        try:
            broadcast_golden_data, broadcast_my_data = torch.broadcast_tensors(golden_data, my_data)
        except RuntimeError as e:
            logger.debug(f"torch.broadcast_tensors RuntimeError: {e}")
            broadcast_golden_data, broadcast_my_data = align_tensors(golden_data, my_data)
        row_data.update(compare_data(broadcast_golden_data, broadcast_my_data))
    else:
        row_data.update(compare_data(golden_data, my_data))
    row_data.update(set_tensor_basic_info_in_row_data(golden_data, my_data))

    return row_data


def load_as_torch_tensor(data_path, loaded_data=None):
    if loaded_data is not None:
        if str(loaded_data.dtype) in BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP:
            loaded_data = loaded_data.astype(BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP.get(loaded_data.dtype))
        return loaded_data if isinstance(loaded_data, torch.Tensor) else torch.from_numpy(loaded_data)
    else:
        return read_data(data_path)


def set_tensor_basic_info_in_row_data(golden_data, my_data):
    row_data = {}
    row_data[GOLDEN_DTYPE] = str(golden_data.dtype)
    row_data[GOLDEN_SHAPE] = str(list(golden_data.shape))
    if 0 not in golden_data.shape:
        golden_data = golden_data.float()
        row_data[GOLDEN_MAX_VALUE] = golden_data.max().item()
        row_data[GOLDEN_MIN_VALUE] = golden_data.min().item()
        row_data[GOLDEN_MEAN_VALUE] = golden_data.mean().item()

    row_data[MY_DTYPE] = str(my_data.dtype)
    row_data[MY_SHAPE] = str(list(my_data.shape))
    if 0 not in my_data.shape:
        my_data = my_data.float()
        row_data[MY_MAX_VALUE] = my_data.max().item()
        row_data[MY_MIN_VALUE] = my_data.min().item()
        row_data[MY_MEAN_VALUE] = my_data.mean().item()
    return row_data


def save_compare_reault_to_csv(gathered_row_data, output_path=".", columns=CSV_GOLDEN_HEADER):
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError:
        logger.error("cannot create file directory under output path, please check it!")

    cur_time = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
    csv_save_path = os.path.join(output_path, f"msit_cmp_report_{cur_time}.csv")

    # 过滤不宜展示的数据，int8建议只与int8比较
    for row_data in gathered_row_data:
        if GOLDEN_DTYPE in row_data and MY_DTYPE in row_data:
            if (row_data[GOLDEN_DTYPE] == 'torch.int8') ^ (row_data[MY_DTYPE] == 'torch.int8'):
                gathered_row_data.remove(row_data)

    data_frame = pd.DataFrame(gathered_row_data, columns=columns)
    data_frame.fillna(value="", inplace=True)
    data_frame.dropna(axis=0, how="all", inplace=True)
    data_frame.to_csv(csv_save_path, index=False)
    logger.info(f"Saved comparing results: {csv_save_path}")
    return csv_save_path


def compare_data(golden_data, my_data):
    golden_data_fp32 = golden_data.reshape(-1).float()
    my_data_fp32 = my_data.reshape(-1).float()
    return compare_tensor(golden_data_fp32, my_data_fp32)


def read_data(data_path):
    data_path = load_file_to_read_common_check(data_path)

    if data_path.endswith(".npy"):
        data = torch.as_tensor(np.load(data_path))
    elif data_path.endswith(".bin"):
        data = read_atb_data(data_path)
    elif data_path.endswith(".pth") or data_path.endswith(".pt"):
        data = torch.load(data_path, weights_only=True, map_location=torch.device("cpu"))
    else:
        logger.error("Unsupported data format %s", data_path)
        raise TypeError("Unsupported data format.")
    
    return data.cpu()


def compare_tensor(golden_data_fp32, my_data_fp32):
    row_data, fail_messages = {}, []

    # 检查tensor的shape是否一致、是否存在NAN或inf
    tensor_pass, message = check_tensor(golden_data_fp32, my_data_fp32)
    if not tensor_pass:
        logger.debug(f"check_tensor failed: {message}")
        row_data[CMP_FAIL_REASON] = message
        return row_data

    for name, cmp_func in list(CMP_ALG_MAP.items()) + list(CUSTOM_ALG_MAP.items()):
        result, message = cmp_func(golden_data_fp32, my_data_fp32)
        row_data[name] = result
        if len(message) > 0:
            fail_messages.append(message)
    row_data[CMP_FAIL_REASON] = " ".join(fail_messages)
    return row_data


def check_tensor(golden_data_fp32, my_data_fp32):
    tensor_pass = True
    fail_reasons = []

    # 检验golden tensor和my tensor的shape是否一致
    if len(golden_data_fp32) != len(my_data_fp32):
        fail_reasons.append("data shape doesn't match.")
        tensor_pass = False
    # 检验golden_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(golden_data_fp32)):
        fail_reasons.append("golden_data includes NAN or inf.")
        tensor_pass = False
    # 检验my_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(my_data_fp32)):
        fail_reasons.append("my_data includes NAN or inf.")
        tensor_pass = False
    return tensor_pass, " ".join(fail_reasons)


def align_tensors(tensor1, tensor2, dim=0):
    """
    将两个shape不一致的tensor对齐为一致
    :param tensor1: 第一个张量
    :param tensor2: 第二个张量
    :param dim: 需要对齐的维度, 默认为0
    :return: 对齐后的两个张量
    """
    tensor1_shape = list(tensor1.shape)
    tensor2_shape = list(tensor2.shape)
    if tensor1_shape[dim] > tensor2_shape[dim]:
        larger_tensor, smaller_tensor = tensor1, tensor2
        larger_shape, smaller_shape = tensor1_shape, tensor2_shape
    else:
        larger_tensor, smaller_tensor = tensor2, tensor1
        larger_shape, smaller_shape = tensor2_shape, tensor1_shape

        # 计算需要对齐的倍数和余数
    multiplier = larger_shape[dim] // smaller_shape[dim]
    remainder = larger_shape[dim] % smaller_shape[dim]

    # 如果倍数不为整数或有余数，则无法简单对齐
    if multiplier * smaller_shape[dim] != larger_shape[dim] or remainder != 0:
        raise ValueError("Cannot align tensors by simply replicating the smaller tensor along the specified dimension.")

        # 复制较小张量并拼接以匹配较大张量的形状
    tiles = [1] * len(smaller_shape)
    tiles[dim] = multiplier
    smaller_replicated = smaller_tensor.repeat(tiles)

    # 如果开始时tensor1是较小的张量，现在需要交换回来
    if tensor1_shape[dim] < tensor2_shape[dim]:
        return smaller_replicated, larger_tensor
    else:
        return larger_tensor, smaller_replicated
