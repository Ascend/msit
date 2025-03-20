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
训练预测每个状态速度的线性模型
"""
import re
from pathlib import Path
from typing import Optional, Union
from collections import namedtuple

from sklearn.model_selection import train_test_split
from loguru import logger

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

from modelevalstate.inference.constant import OpAlgorithm
from modelevalstate.inference.data_format_v1 import (
    MODEL_OP_FIELD, 
    MODEL_STRUCT_FIELD, 
    MODEL_CONFIG_FIELD, 
    MINDIE_FIELD, 
    ENV_FIELD, 
    HARDWARE_FIELD
)
from modelevalstate.inference.dataset import CustomOneHotEncoder, CustomLabelEncoder, preset_category_data
from modelevalstate.inference.dataset import PreprocessTool, TOTAL_OUTPUT_LENGTH, TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN

from pandas import DataFrame, Series
from matplotlib import pyplot as plt


class MyDataSet:
    def __init__(self, custom_encoder: Optional[Union[CustomOneHotEncoder, CustomLabelEncoder]] = None,
                 predict_field="model_execute_time",
                 test_size=0.1, shuffle=True, op_algorithm: OpAlgorithm = OpAlgorithm.EXPECTED):
        self.predict_field = predict_field
        self.test_size = test_size
        self.shuffle = shuffle
        if custom_encoder:
            self.custom_encoder = custom_encoder
        else:
            self.custom_encoder = CustomOneHotEncoder()
        self.features = None
        self.labels = None
        self.load_data = None
        self.op_algorithm = op_algorithm
        self.sub_columns = []
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

    @staticmethod
    def convert_batch_info(row: str, index: str) -> Series:
        origin_row = eval(row)
        return PreprocessTool.generate_series(origin_row, eval(index))

    @staticmethod
    def convert_request_info(row: str, index: str) -> Series:
        origin_index = eval(index)
        origin_row = eval(row)
        RequestInfo = namedtuple("RequestInfo", origin_index)
        _row_request_info = [RequestInfo(*[int(float(i)) for i in _row]) for _row in origin_row]
        return PreprocessTool.generate_series_with_request_info(_row_request_info, origin_index)

    @staticmethod
    def convert_op_info(row: str, index: str) -> Series:
        return PreprocessTool.generate_series_with_op_info(eval(row), eval(index))

    @staticmethod
    def convert_op_info_with_ratio(row: str, index: str) -> Series:
        return PreprocessTool.generate_series_with_op_info_use_ratio(eval(row), eval(index))

    @staticmethod
    def convert_struct_info(row: str, index: str) -> Series:
        return PreprocessTool.generate_series_with_struct_info(eval(row), eval(index))

    @staticmethod
    def convert_config_info(row: str, index: str) -> Series:
        return PreprocessTool.gene_series_with_model_config(eval(row), eval(index))

    @staticmethod
    def convert_mindie_info(row: str, index: str) -> Series:
        return PreprocessTool.generate_series(eval(row), eval(index))

    @staticmethod
    def convert_env_info(row: str, index: str) -> Series:
        return PreprocessTool.generate_series(eval(row), eval(index))

    @staticmethod
    def convert_hardware_info(row: str, index: str) -> Series:
        return PreprocessTool.generate_series(eval(row), eval(index))
    
    @staticmethod
    def get_all_request_info(row: str, index: str) -> DataFrame:
        # 获取所有request原始数据特征，用来分析原始数据
        origin_index = eval(index)
        origin_row = eval(row)
        _row_request_info = []
        for _row in origin_row:
            _row_request_info.append([int(float(i)) for i in _row])
        return pd.DataFrame(_row_request_info, columns=origin_index)

    def preprocess(self, lines_data: Optional[DataFrame] = None):
        # 数据预处理
        # 将各个特征数据转换为列数据
        batch_df = lines_data.iloc[:, 0].apply(self.convert_batch_info, args=(lines_data.columns[0],))
        request_df = lines_data.iloc[:, 1].apply(self.convert_request_info, args=(lines_data.columns[1],))
        batch_df[TOTAL_OUTPUT_LENGTH] = request_df[TOTAL_OUTPUT_LENGTH]
        batch_df[TOTAL_SEQ_LENGTH] = batch_df[TOTAL_OUTPUT_LENGTH] + batch_df[TOTAL_PREFILL_TOKEN]
        request_df = request_df.drop(TOTAL_OUTPUT_LENGTH, axis=1)
        self.sub_columns = [batch_df.columns.tolist(), request_df.columns.tolist()]
        _load_data = [batch_df, request_df]
        for i, _cur_columns in enumerate(lines_data.columns[2:]):
            if eval(_cur_columns) == MODEL_OP_FIELD:
                if self.op_algorithm == OpAlgorithm.EXPECTED:
                    model_op_df = lines_data.iloc[:, i+2].apply(self.convert_op_info, args=(_cur_columns, ))
                else:
                    model_op_df = lines_data.iloc[:, i+2].apply(self.convert_op_info_with_ratio, args=(_cur_columns, ))
                self.sub_columns.append(model_op_df.columns.tolist())
                _load_data.append(model_op_df)
            elif eval(_cur_columns) == MODEL_STRUCT_FIELD:
                model_struct_df = lines_data.iloc[:, i+2].apply(self.convert_struct_info, args=(_cur_columns, ))
                self.sub_columns.append(model_struct_df.columns.tolist())
                _load_data.append(model_struct_df)
            elif eval(_cur_columns) == MODEL_CONFIG_FIELD:
                model_config_df = lines_data.iloc[:, i+2].apply(self.convert_config_info, args=(_cur_columns, ))
                self.sub_columns.append(model_config_df.columns.tolist())
                _load_data.append(model_config_df)
            elif eval(_cur_columns) == MINDIE_FIELD:
                mindie_df = lines_data.iloc[:, i+2].apply(self.convert_mindie_info, args=(_cur_columns, ))
                self.sub_columns.append(mindie_df.columns.tolist())
                _load_data.append(mindie_df)
            elif eval(_cur_columns) == ENV_FIELD:
                env_df = lines_data.iloc[:, i+2].apply(self.convert_env_info, args=(_cur_columns, ))
                self.sub_columns.append(env_df.columns.tolist())
                _load_data.append(env_df)
            elif eval(_cur_columns) == HARDWARE_FIELD:
                hardware_df = lines_data.iloc[:, i+2].apply(self.convert_hardware_info, args=(_cur_columns, ))
                self.sub_columns.append(hardware_df.columns.tolist())
                _load_data.append(hardware_df)
        # 提取 features 和labels
        self.load_data = pd.concat(_load_data, axis=1)
        self.labels = pd.DataFrame(batch_df[self.predict_field], columns=[self.predict_field])
        batch_df = batch_df.drop(self.predict_field, axis=1)
        self.features = pd.concat([batch_df, *_load_data[1:]], axis=1)
        # 使用sklearn 进行 one-hot
        self.features = self.custom_encoder.transformer(self.features)
        return self.features, self.labels

    def construct_data(self, lines_data: Optional[DataFrame] = None, plt_data: bool = True,
                       middle_save_path: Optional[Path] = None):
        features, labels = self.preprocess(lines_data)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(features, labels,
                                                                                test_size=self.test_size,
                                                                                shuffle=self.shuffle)
        # 检查处理的数据是否有重复的column name
        if len(self.features.columns) != len(self.features.columns.unique()):
            raise ValueError("Duplicate column names found in the features.")
        if plt_data:
            self.plt_data(lines_data, middle_save_path)

    @classmethod
    def plot_custom_pairplot(self, df: DataFrame, middle_save_path: Optional[Path] = None,
                             file_name: str = "pairplot.png"):
        col_num = df.shape[1]
        fig, axs = plt.subplots(col_num, col_num, figsize=(4 * col_num, 4 * col_num))
        for i in range(col_num):
            for j in range(col_num):
                if i == j:
                    if df.columns[i].lower() in ["max_seq_len", "input_length", "total_prefill_token"]:
                        sns.histplot(df.iloc[:, i], ax=axs[i, j], bins=100)
                    else:
                        sns.histplot(df.iloc[:, i], ax=axs[i, j])
                elif j > i:
                    continue
                else:
                    sns.regplot(x=df.iloc[:, i], y=df.iloc[:, j], ax=axs[i, j])
        plt.tight_layout()
        if middle_save_path:
            plt.savefig(middle_save_path.joinpath(file_name))
        else:
            plt.show()
        plt.close()

    def analysis_batch_feature(self, middle_save_path: Optional[Path] = None):
        cur_batch_df = self.load_data.iloc[:, 0:len(self.sub_columns[0])]
        custom_label_encoder = CustomLabelEncoder([preset_category_data[0]])
        custom_label_encoder.fit()
        cur_batch_df = custom_label_encoder.transformer(cur_batch_df)
        self.plot_custom_pairplot(cur_batch_df, middle_save_path,
                                  "batch_pairplot.png")

    def plt_data(self, line_data: DataFrame, middle_save_path: Optional[Path] = None):
        self.analysis_batch_feature(middle_save_path)
        self.analysis_origin_request_hist(line_data, middle_save_path)
    
    def save(self, save_path: Path):
        self.features.to_csv(save_path.joinpath("features_preprocess.csv"), index=False)
        self.load_data.to_csv(save_path.joinpath("load_data.csv"), index=False)
        self.test_x.to_csv(save_path.joinpath("test_x.csv"), index=False)
        self.test_y.to_csv(save_path.joinpath("test_y.csv"), index=False)
        self.train_x.to_csv(save_path.joinpath("train_x.csv"), index=False)
        self.train_y.to_csv(save_path.joinpath("train_y.csv"), index=False)

    def analysis_origin_request_hist(self, df: DataFrame, middle_save_path: Optional[Path] = None):
        request_series = df.iloc[:, 1].apply(self.get_all_request_info, args=(df.columns[1],))
        request_df = pd.concat(request_series.values, ignore_index=True)
        self.plot_custom_pairplot(request_df, middle_save_path, "request_pairplot.png")


class SimpleDataset(MyDataSet):
    def preprocess(self, lines_data: Optional[DataFrame] = None):
        # 数据预处理
        # 将各个特征数据转换为列数据
        batch_df = lines_data.iloc[:, 0].apply(self.convert_batch_info, args=(lines_data.columns[0],))
        request_df = lines_data.iloc[:, 1].apply(self.convert_request_info, args=(lines_data.columns[1],))
        batch_df[TOTAL_OUTPUT_LENGTH] = request_df[TOTAL_OUTPUT_LENGTH]
        batch_df[TOTAL_SEQ_LENGTH] = batch_df[TOTAL_OUTPUT_LENGTH] + batch_df[TOTAL_PREFILL_TOKEN]
        request_df = request_df.drop(TOTAL_OUTPUT_LENGTH, axis=1)
        model_config_df = lines_data.iloc[:, 4].apply(self.convert_config_info, args=(lines_data.columns[4],))
        mindie_df = lines_data.iloc[:, 5].apply(self.convert_mindie_info, args=(lines_data.columns[5],))
        env_df = lines_data.iloc[:, 6].apply(self.convert_env_info, args=(lines_data.columns[6],))
        hardware_df = lines_data.iloc[:, 7].apply(self.convert_hardware_info, args=(lines_data.columns[7],))
        self.sub_columns = [batch_df.columns.tolist(), request_df.columns.tolist(), model_config_df.columns.tolist(),
                            mindie_df.columns.tolist(), 
                            env_df.columns.tolist(), hardware_df.columns.tolist()]
        # 提取 features 和labels
        self.load_data = pd.concat([batch_df, request_df, model_config_df, mindie_df, 
                                    env_df, hardware_df], axis=1)
        self.labels = pd.DataFrame(batch_df[self.predict_field], columns=[self.predict_field])
        batch_df = batch_df.drop(self.predict_field, axis=1)
        self.features = pd.concat([batch_df, request_df, model_config_df, mindie_df,
                                   env_df, hardware_df], axis=1)
        # 使用sklearn 进行 one-hot
        self.features = self.custom_encoder.transformer(self.features)
        return self.features, self.labels


class DecodeDataSet:
    # 处理request 和question的数据，生成用来预测请求的decode轮数信息
    def __init__(self, predict_field: str = "output_length", test_size=0.1, shuffle=True, ):
        self.predict_field = predict_field
        self.test_size = test_size
        self.shuffle = shuffle
        self.load_data = None
        self.features = None
        self.labels = None
        self.train_x = None 
        self.test_x = None 
        self.train_y = None
        self.test_y = None

    @staticmethod
    def count_punctuation_marks(line: str):
        # 定义一个包含所有标点符号的正则表达式模式
        punctuation_pattern = (r'[.,!?;:"\(\)\{\}\[\]\#\$\%\&\*\<\>\\\/\@\u3000\u3001\u3002\u300A\u300B\uFF01'
                               r'\uFF03\uFF0E\u2018\u2019\u201C\u201D\u201F\u201E\u2032\u2036\u2039\u203B\u2045'
                               r'\u205F\u301F\uFE50\uFF5E\uFF5F\uFF61\uFF62\uFF64\uFF65\uFF66\uFF67\uFF68\uFF69'
                               r'\uFF6A\uFF6B\uFF6C\uFF6D\uFF6E\uFF6F\uFF70\uFF71\uFF72\uFF73\uFF74\uFF75\uFF76'
                               r'\uFF77\uFF78\uFF79\uFF7A\uFF7B\uFF7C\uFF7D\uFF7E\uFF7F\uFF80\uFF81\uFF82\uFF83'
                               r'\uFF84\uFF85\uFF86\uFF87\uFF88\uFF89\uFF8A\uFF8B\uFF8C\uFF8D\uFF8E\uFF8F\uFF90'
                               r'\uFF91\uFF92\uFF93\uFF94\uFF95\uFF96\uFF97\uFF98\uFF99\uFF9A\uFF9B\uFF9C\uFF9D'
                               r'\uFF9E\uFF9F\uFFA0\uFFA1\uFFA2\uFFA3\uFFA4\uFFA5\uFFA6\uFFA7\uFFA8\uFFA9\uFFAA'
                               r'\uFFAB\uFFAC\uFFAD\uFFAE\uFFAF\uFFB0\uFFB1\uFFB2\uFFB3\uFFB4\uFFB5\uFFB6\uFFB7'
                               r'\uFFB8\uFFB9\uFFBA\uFFBB\uFFBC\uFFBD\uFFBE\uFFBF\uFFC0\uFFC1\uFFC2\uFFC3\uFFC4'
                               r'\uFFC5\uFFC6\uFFC7\uFFC8\uFFC9\uFFCA\uFFCB\uFFCC\uFFCD\uFFCE\uFFCF\uFFD0\uFFD1'
                               r'\uFFD2\uFFD3\uFFD4\uFFD5\uFFD6\uFFD7\uFFD8\uFFD9\uFFDA\uFFDB\uFFDC\uFFDD\uFFDE'
                               r'\uFFDF\uFFE0\uFFE1\uFFE2\uFFE3\uFFE4\uFFE5\uFFE6\uFFE7\uFFE8\uFFE9\uFFEA\uFFEB'
                               r'\uFFEC\uFFED\uFFEE\uFFEF\uFFF0\uFFF1\uFFF2\uFFF3\uFFF4\uFFF5\uFFF6\uFFF7\uFFF8'
                               r'\uFFF9\uFFFA\uFFFB\uFFFC\uFFFD\uFFFE\uFFFF]')

        # 使用正则表达式查找所有标点符号
        matches = re.findall(punctuation_pattern, line)

        # 返回匹配的标点符号个数
        return len(matches)
    
    @staticmethod
    def count_chars(line : str):
        # 匹配中文字符
        chinese_pattern = r'[^\x00-\xff]'
        chinese_matches = re.findall(chinese_pattern, line)
        chinese_count = len(chinese_matches)

        # 匹配英文字符
        english_pattern = r'[a-zA-Z]'
        english_matches = re.findall(english_pattern, line)
        english_count = len(english_matches)

        other_count = len(line) - chinese_count - english_count
        return chinese_count, english_count, other_count

    def preprocess(self, lines_data: Optional[DataFrame] = None):
        # 提取标点个数
        lines_data["punctuation"] = lines_data["question"].apply(DecodeDataSet.count_punctuation_marks)
        lines_data[["en_chart_count", "zh_chart_count", "other_chart_count"]] = lines_data["question"].apply(
            DecodeDataSet.count_chars).apply(pd.Series)
        self.load_data = lines_data.drop(["question", "answer"], axis=1)
        self.labels = self.load_data[self.predict_field]
        self.features = self.load_data.drop(self.predict_field, axis=1)
        return self.features, self.labels
    
    def construct_data(self, lines_data: Optional[DataFrame] = None, plt_data: bool = True,
                    middle_save_path: Optional[Path] = None):
        features, labels = self.preprocess(lines_data)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(features, labels,
                                                                                test_size=self.test_size,
                                                                                shuffle=self.shuffle)
        if plt_data:
            self.plt_data(lines_data, middle_save_path)

    def plt_data(self, line_data: DataFrame, middle_save_path: Optional[Path] = None):
        p = sns.pairplot(self.load_data, corner=True)
        if middle_save_path:
            p.savefig(middle_save_path.joinpath("decore_num_feature.png"))
        else:
            plt.show()
        plt.close()
    
    def save(self, save_path: Path):
        self.features.to_csv(save_path.joinpath("features_preprocess.csv"), index=False)
        self.load_data.to_csv(save_path.joinpath("load_data.csv"), index=False)
        self.test_x.to_csv(save_path.joinpath("test_x.csv"), index=False)
        self.test_y.to_csv(save_path.joinpath("test_y.csv"), index=False)
        self.train_x.to_csv(save_path.joinpath("train_x.csv"), index=False)
        self.train_y.to_csv(save_path.joinpath("train_y.csv"), index=False)