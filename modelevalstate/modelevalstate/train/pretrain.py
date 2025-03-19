# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
1. 训练基础模型
2. 用基础模型拟合概率
3. 用新增加数据继续训练模型
"""
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from math import ceil
from typing import Tuple, Optional, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from modelevalstate.train.common import computer_speed_with_second, get_train_sub_path, update_gloabal_coefficient
from modelevalstate.train.xgb_state_model import StateXgbModel
from modelevalstate.data_feature.dataset import MyDataSet, CustomOneHotEncoder, CustomLabelEncoder, preset_category_data, DecodeDataSet
from modelevalstate.inference.constant import OpAlgorithm
from modelevalstate.inference.common import HistInfo, model_op_size, OP_EXPECTED_FIELD_MAPPING, OP_SACLE_FIELD_MAPPING

@dataclass
class NodeInfo:
    stage: str # 当前模型状态的类型 P/D
    batch_size: int # 当前模型状态处理的请求个数


class PreTrainModel:
