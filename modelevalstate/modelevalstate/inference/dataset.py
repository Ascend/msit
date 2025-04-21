# -*- coding: utf-8 -*-
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

import pickle
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from pathlib import Path

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from modelevalstate.inference.constant import DTYPE_CATEGORY, ALL_ASCEND_NAME, ALL_HIDDEN_ACT, ALL_MODEL_TYPE, \
    ALL_QUANTIZE, \
    ALL_KV_QUANT_TYPE, ALL_GROUP_SIZE, ALL_REDUCE_QUANT_TYPE, ALL_BATCH_STAGE
from modelevalstate.inference.data_format_v1 import BatchField, RequestField, ModelOpField, ModelStruct, ModelConfig, \
    MindieConfig, \
    EnvField, HardWare, BATCH_FIELD, REQUEST_FIELD, MODEL_OP_FIELD, MODEL_STRUCT_FIELD, MODEL_CONFIG_FIELD, \
    MINDIE_FIELD, ENV_FIELD, HARDWARE_FIELD
from modelevalstate.inference.utils import PreprocessTool, TOTAL_OUTPUT_LENGTH, TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN


@dataclass
class InputData:
    batch_field: BatchField
    request_field: Tuple[RequestField, ...]
    model_op_field: Optional[Tuple[ModelOpField, ...]] = None
    model_struct_field: Optional[ModelStruct] = None
    model_config_field: Optional[ModelConfig] = None
    mindie_field: Optional[MindieConfig] = None
    env_field: Optional[EnvField] = None
    hardware_field: Optional[HardWare] = None


@dataclass
class CategoryInfo:
    name: str = "batch_stage"
    ohe_path: Path = Path("batch_stage.pt")
    all_value: Tuple[str] = ("prefill", "decode")


preset_category_data = [
    CategoryInfo("batch_stage", Path("batch_stage_ohe.pkl"), ALL_BATCH_STAGE),
    CategoryInfo("soc_name", Path("soc_name_ohe.pkl"), ALL_ASCEND_NAME),
    CategoryInfo("hidden_act", Path("hidden_act_ohe.pkl"), ALL_HIDDEN_ACT),
    CategoryInfo("model_type", Path("model_type_ohe.pkl"), ALL_MODEL_TYPE),
    CategoryInfo("torch_dtype", Path("torch_dtype_ohe.pkl"), DTYPE_CATEGORY),
    CategoryInfo("quantize", Path("quantize_ohe.pkl"), ALL_QUANTIZE),
    CategoryInfo("kv_quant_type", Path("kv_quant_type_ohe.pkl"), ALL_KV_QUANT_TYPE),
    CategoryInfo("group_size", Path("group_size_ohe.pkl"), ALL_GROUP_SIZE),
    CategoryInfo("reduce_quant_type", Path("reduce_quant_type_ohe.pkl"), ALL_REDUCE_QUANT_TYPE)
]


class CustomOneHotEncoder:
    def __init__(self, one_hots: Optional[List[CategoryInfo]] = None, save_dir: Optional[Path] = None):
        if one_hots:
            self.one_hots: List[CategoryInfo] = one_hots
        else:
            self.one_hots = []
        if save_dir:
            for _one_hot in self.one_hots:
                _one_hot.ohe_path = save_dir.joinpath(_one_hot.ohe_path)
        self.one_hot_encoders: List[OneHotEncoder] = []

    def fit(self, load: bool = False):
        self.one_hot_encoders = []
        for _one_hot in self.one_hots:
            if load:
                with open(_one_hot.ohe_path, "rb") as f:
                    _cur_one_hot = pickle.load(f)
            else:
                _cur_one_hot = OneHotEncoder(handle_unknown='infrequent_if_exist')
                _cur_one_hot.fit([[k] for k in _one_hot.all_value])
            self.one_hot_encoders.append(_cur_one_hot)

    def save(self):
        for i, _one_hot in enumerate(self.one_hots):
            with open(_one_hot.ohe_path, "wb") as f:
                pickle.dump(self.one_hot_encoders[i], f)

    def transformer(self, x: DataFrame):
        for i, _one_hot_encoder in enumerate(self.one_hot_encoders):
            _one_hot_info = self.one_hots[i]
            encode_value = _one_hot_encoder.transform(x[_one_hot_info.name].values.reshape(-1, 1)).toarray()
            column_names = []
            for category in _one_hot_encoder.categories_:
                for __ in category:
                    column_names.append(f"{_one_hot_info.name}__{i}")
            _encode_df = pd.DataFrame(encode_value, columns=column_names)
            x = pd.concat([_encode_df, x], axis=1)
            x = x.drop(_one_hot_info.name, axis=1)
        return x


class CustomLabelEncoder:
    def __init__(self, category_info: Optional[List[CategoryInfo]] = None, save_dir: Optional[Path] = None):
        if category_info:
            self.category_info: List[CategoryInfo] = category_info
        else:
            self.category_info = []
        if save_dir:
            for _category in self.category_info:
                _category.ohe_path = save_dir.joinpath(_category.ohe_path)
        self.category_encoders: List[LabelEncoder] = []

    def fit(self, load: bool = False):
        self.category_encoders = []
        for _cate_info in self.category_info:
            if load:
                with open(_cate_info.ohe_path, "rb") as f:
                    _cur_encoder = pickle.load(f)
            else:
                _cur_encoder = LabelEncoder()
                _cur_encoder.fit([[k] for k in _cate_info.all_value])
            self.category_encoders.append(_cur_encoder)

    def save(self):
        for i, _cate_info in enumerate(self.category_info):
            with open(_cate_info.ohe_path, "wb") as f:
                pickle.dump(self.category_encoders[i], f)

    def transformer(self, x: DataFrame):
        for i, _cate_encoder in enumerate(self.category_encoders):
            _cate_info = self.category_info[i]
            if _cate_info.name not in x.columns:
                continue
            encode_value = _cate_encoder.transform(x[_cate_info.name].values)
            x[_cate_info.name] = encode_value
        return x


class DataProcessor:
    def __init__(self, custom_encoder: Optional[Union[CustomOneHotEncoder, CustomLabelEncoder]] = None):
        self.custom_encoder = custom_encoder

    def preprocessor(self, input_data: InputData) -> np.ndarray:
        batch_series = PreprocessTool.generate_series(input_data.batch_field, BATCH_FIELD)
        request_series = PreprocessTool.generate_series_with_request_info(input_data.request_field, REQUEST_FIELD)
        batch_series[TOTAL_OUTPUT_LENGTH] = request_series[TOTAL_OUTPUT_LENGTH]
        batch_series[TOTAL_SEQ_LENGTH] = batch_series[TOTAL_PREFILL_TOKEN] + request_series[TOTAL_OUTPUT_LENGTH]
        request_series = request_series.drop(TOTAL_OUTPUT_LENGTH)
        _load_data = [batch_series, request_series]
        if input_data.model_op_field:
            model_op_series = PreprocessTool.generate_series_with_op_info(input_data.model_op_field, MODEL_OP_FIELD)
            _load_data.append(model_op_series)
        if input_data.model_struct_field:
            model_struct_series = PreprocessTool.generate_series_with_struct_info(input_data.model_struct_field,
                                                                                  MODEL_STRUCT_FIELD)
            _load_data.append(model_struct_series)
        if input_data.model_config_field:
            model_config_series = PreprocessTool.gene_series_with_model_config(input_data.model_config_field,
                                                                               MODEL_CONFIG_FIELD)
            _load_data.append(model_config_series)
        if input_data.mindie_field:
            mindie_series = PreprocessTool.generate_series(input_data.mindie_field, MINDIE_FIELD)
            _load_data.append(mindie_series)
        if input_data.env_field:
            env_series = PreprocessTool.generate_series(input_data.env_field, ENV_FIELD)
            _load_data.append(env_series)
        if input_data.hardware_field:
            hardware_series = PreprocessTool.generate_series(input_data.hardware_field, HARDWARE_FIELD)
            _load_data.append(hardware_series)
        feature_series = pd.concat(_load_data)
        feature_df = feature_series.to_frame().T
        feature_df = self.custom_encoder.transformer(feature_df)
        return feature_df.values
