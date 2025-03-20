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
"""
系统状态评估
1. 需要安装xgboost, pandas, numpy, sklearn
2. 获取训练好的模型及编码器。

功能：
1. 预测v1版本的数据。参考： demo_predict_v1


"""
import sys
import os

current_dir = os.path.dirname(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.append(0, current_dir)

from typing import Tuple
from pathlib import Path

import xgboost

from modelevalstate.inference.dataset import InputData, DataProcessor, CustomLabelEncoder, preset_category_data
from modelevalstate.inference.data_format_v1 import BatchField, RequestField, ConfigPath
from modelevalstate.inference.file_reader import FileHanlder, StaticFile


class XGBStateEvaluate:
    """
    1. 预处理数据
    2. 进行预测
    """

    def __init__(self, xgb_model_path: Path, dataprocessor: DataProcessor):
        self.xgb_model_path = xgb_model_path
        self.prefill_type = "prefill"
        self.decode_type = "decode"
        self.xgb_model = self.load_model(self.xgb_model_path)
        self.data_processor = dataprocessor

    @staticmethod
    def load_model(model_path):
        _model = xgboost.Booster()
        _model.load_model(model_path)
        return _model

    def predict(self, input_data: InputData) -> Tuple[float, float]:
        """
        根据输入预测Up，Ud.
        :param config_info: 运行系统的配置信息
        :param input_data: 运行系统的状态
        :return: Up, Ud
        """
        # 1. 处理数据
        stage = input_data.batch_field.batch_stage.lower()
        line_info = self.data_processor.preprocessor(input_data)
        # 2. 进行预测
        res = self.xgb_model.predict(xgboost.DMatrix(line_info, feature_names=self.xgb_model.feature_names))[0].item()
        _up = _ud = -1
        if stage == self.prefill_type:
            _up = res
        elif stage == self.decode_type:
            _ud = res
        else:
            raise ValueError(
                f"Data error. expected Data Type {self.prefill_type, self.decode_type}. got is {stage}")
        return _up, _ud


# 接口提供三个参数1个batch字段，1个request字段，1个config 字段。

def predict_v1(batch_info: BatchField, request_info: Tuple[RequestField, ...], config_path: ConfigPath):
    # 读取其他字段数据
    static_file = StaticFile(base_path=config_path.static_file_dir)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    # 组合为input data
    input_data = InputData(
        batch_field=batch_info,
        request_field=request_info,
        model_op_field=fh.get_op_field(batch_info.batch_stage, batch_info.batch_size, batch_info.max_seq_len,
                                       fh.prefill_op_data, fh.decode_op_data),
        model_struct_field=fh.model_struct_info,
        model_config_field=fh.model_config_info,
        mindie_field=fh.mindie_info,
        env_field=fh.env_info,
        hardware_field=fh.hardware
    )
    # 进行预测
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=config_path.ohe_path)
    custom_encoder.fit(load=True)
    # 加载模型
    data_processor = DataProcessor(custom_encoder)
    xgb_state_eval = XGBStateEvaluate(
        xgb_model_path=config_path.model_path,
        dataprocessor=data_processor)
    # 预测
    res = xgb_state_eval.predict(input_data)
    return res


def predict_v1_with_cache(batch_info: BatchField, request_info: Tuple[RequestField, ...], config_path: ConfigPath,
                          fh: FileHanlder, data_processor: DataProcessor):
    # 组合为input data
    input_data = InputData(
        batch_field=batch_info,
        request_field=request_info,
        model_op_field=fh.get_op_field(batch_info.batch_stage, batch_info.batch_size, batch_info.max_seq_len,
                                       fh.prefill_op_data, fh.decode_op_data),
        model_struct_field=fh.model_struct_info,
        model_config_field=fh.model_config_info,
        mindie_field=fh.mindie_info,
        env_field=fh.env_info,
        hardware_field=fh.hardware
    )
    xgb_state_eval = XGBStateEvaluate(
        xgb_model_path=config_path.model_path,
        dataprocessor=data_processor)
    # 预测
    res = xgb_state_eval.predict(input_data)
    return res


def demo_predict_v1():
    """
    使用：
    1. 先调用inference/generate_env_hardware.py，生成env.json和hardware。json
    2. 将env.json和hardware.json放到模型的静态文件目录下。例如llama3-8b-12-13
    3. 再启动仿真程序调用predict_v1函数。

    """
    batch_field = BatchField('decode', 20, 20.0, 580.0, 29.0)
    request_field = (RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2))
    config_path = ConfigPath(Path(r".\PyProject\state_eval\tmp\pd_content\train\151\bak\base\xgb_model.ubj"),
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\151\ohe"),
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\151\deepseek_r1")
                             )
    batch_field = BatchField('prefill', 20, 20.0, 580.0, 29.0)




def demo_predict_v1_with_cache():
    batch_field = BatchField('decode', 20, 20.0, 580.0, 29.0)
    request_field = (RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2))
    config_path = ConfigPath(Path(r".\PyProject\state_eval\tmp\pd_content\train\114\bak\base\xgb_model.ubj"),
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\114\ohe"),
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\114\llama3-8b")
                             )
    static_file = StaticFile(base_path=config_path.static_file_dir)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=config_path.ohe_path)
    custom_encoder.fit(load=True)
    data_processor = DataProcessor(custom_encoder)
    batch_field = BatchField('prefill', 20, 20.0, 580.0, 29.0)
   


def demo_predict_v1_with_cache_with_simple_data_processor():
    batch_field = BatchField('decode', 20, 20.0, 580.0, 29.0)
    request_field = (RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2), RequestField(29.0, 1, 2),
                     RequestField(29.0, 1, 2), RequestField(29.0, 1, 2))
    config_path = ConfigPath(
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\155\bak\base\xgb_model.ubj"),
                             Path(r".\PyProject\state_eval\tmp\pd_content\train\155\ohe"),
                             Path(r".\PyProject\ModelEvalState\data\v1.0.0\deepseek_r1_forward_0")
                             )
    static_file = StaticFile(base_path=config_path.static_file_dir)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=config_path.ohe_path)
    custom_encoder.fit(load=True)
    data_processor = DataProcessor(custom_encoder)
    batch_field = BatchField('prefill', 20, 20.0, 580.0, 29.0)
    

