# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# 使用多个模型进行预测，并比较不同模型的预测结果

import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Type

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from scipy import stats

from modelevalstate.data_feature.v1 import FileReader
from modelevalstate.inference.dataset import CustomLabelEncoder, preset_category_data
from modelevalstate.data_feature.dataset import MyDataSet, SimpleDataset
from modelevalstate.train.pretrain import NodeInfo, PretrainModel
from modelevalstate.analysis import AnalysisState
from modelevalstate.model.xgb_state_model import StateXgbModel

BATCH_PREFILL = "batch_prefill"
BATCH_DECODE = "batch_decode"


def get_direction(origin_value_list, predict_value_list):
    directions = 0
    for i, (origin, predict) in enumerate(zip(origin_value_list, predict_value_list)):
        if i == 0:
            continue
        if origin > origin_value_list[i - 1] and predict > predict_value_list[i - 1]:
            directions += 1
        elif origin < origin_value_list[i - 1] and predict < predict_value_list[i - 1]:
            directions += 1
    return directions


def predict_with_model(lines_data: DataFrame,
                       xgb_model_path: Optional[Path] = None,
                       ohe_path: Optional[Path] = None,
                       train_field="model_execute_time",
                       dataset_type: Type[MyDataSet] = MyDataSet):
    origin_data: List[NodeInfo] = []
    predict_data: List[NodeInfo] = []
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=ohe_path)
    custom_encoder.fit()
    data_processor = dataset_type(custom_encoder, predict_field=train_field)
    data_processor.construct_data(lines_data, plt_data=False)
    model = StateXgbModel(save_model_path=xgb_model_path, load_model_path=xgb_model_path, save_model=False)
    res = model.predict(data_processor.features)
    batch_stage_encoder = [
        custom_encoder.category_encoders[i]
        for i, v in enumerate(custom_encoder.category_info)
        if v.name == "batch_stage"
    ][0]
    data_processor.features["batch_stage"] = batch_stage_encoder.inverse_transform(
        data_processor.features["batch_stage"]
    )
    _origin_value_list = []
    _predict_value_list = []
    for ind, row in data_processor.features.iterrows():
        _cur_node = NodeInfo(row["batch_stage"], row["batch_size"])
        setattr(_cur_node, train_field, data_processor.labels.iloc[ind].item())
        _origin_value_list.append(data_processor.labels.iloc[ind].item())
        origin_data.append(_cur_node)
        _cur_node = deepcopy(_cur_node)
        setattr(_cur_node, train_field, res[ind])
        _predict_value_list.append(res[ind])
        predict_data.append(_cur_node)

    # 绘制结果图
    all_up, all_ud = PretrainModel.get_decode_and_prefill_time(tuple(predict_data), train_field)
    origin_up, origin_ud = PretrainModel.get_decode_and_prefill_time(tuple(origin_data), train_field)
    _x1, _predict_up_mean, _predict_up_positive_sigma, _predict_up_negative_sigma = AnalysisState.computer_mean_sigma(
        all_up, BATCH_PREFILL)
    _x2, _predict_ud_mean, _predict_ud_positive_sigma, _predict_ud_negative_sigma = AnalysisState.computer_mean_sigma(
        all_ud, BATCH_DECODE)
    _x3, _real_up_mean, _real_up_positive_sigma, _real_up_negative_sigma = \
        AnalysisState.computer_mean_sigma(origin_up, BATCH_PREFILL)
    _x4, _real_ud_mean, _real_ud_positive_sigma, _real_ud_negative_sigma = \
        AnalysisState.computer_mean_sigma(origin_ud, BATCH_DECODE)
    if _x1 != _x3:
        raise AssertionError("_x1 is not equal to _x3")
    if _x2 != _x4:
        raise AssertionError("_x2 is not equal to _x4")
    up_df = DataFrame({
        BATCH_PREFILL: _x1,
        "real_up_mean": _real_up_mean,
        "real_up_positive_sigma": _real_up_positive_sigma,
        "real_up_negative_sigma": _real_up_negative_sigma,
        "predict_up_mean": _predict_up_mean,
        "predict_up_positive_sigma": _predict_up_positive_sigma,
        "predict_up_negative_sigma": _predict_up_negative_sigma
    })
    ud_df = DataFrame({
        BATCH_DECODE: _x2,
        "real_ud_mean": _real_ud_mean,
        "real_ud_positive_sigma": _real_ud_positive_sigma,
        "real_ud_negative_sigma": _real_ud_negative_sigma,
        "predict_ud_mean": _predict_ud_mean,
        "predict_ud_positive_sigma": _predict_ud_positive_sigma,
        "predict_ud_negative_sigma": _predict_ud_negative_sigma
    })

    metric = {
        "mape": mean_absolute_percentage_error(_origin_value_list, _predict_value_list),
        "r2": r2_score(_origin_value_list, _predict_value_list),
        "rmse": np.sqrt(mean_squared_error(_origin_value_list, _predict_value_list)),
        "pearsonr": stats.pearsonr(_origin_value_list, _predict_value_list),
        "directions": get_direction(_origin_value_list, _predict_value_list) / len(_origin_value_list),
        "mean_ud_directions": get_direction(_real_ud_mean, _predict_ud_mean) / len(_real_ud_mean),
        "mean_up_directions": get_direction(_real_up_mean, _predict_up_mean) / len(_real_up_mean)
    }
    return up_df, ud_df, metric


def rename_column(prefix, columns):
    _new_columns = {}
    for _col in columns:
        if "predict" in _col:
            _new_columns[_col] = f"{prefix}__{_col}"
    return _new_columns


def manager():
    file_paths = [Path(r"PyProject\ModelEvalState\data\v1\llama3-8b1226-13\feature.csv")]
    base_path = Path(r"PyProject\state_eval\tmp\pd_content\train\117")
    xgb_model_path = base_path.joinpath("bak/base/xgb_model.ubj")
    ohe_path = base_path.joinpath("ohe")
    train_field = "model_execute_time"
    fl = FileReader(file_paths)
    test_input = fl.read_lines()
    simple_up, simple_ud, metric = predict_with_model(test_input, xgb_model_path=xgb_model_path, ohe_path=ohe_path,
                                                      train_field=train_field, dataset_type=SimpleDataset)
    simple_up = simple_up.rename(columns=rename_column("simple", simple_up.columns))
    simple_ud = simple_ud.rename(columns=rename_column("simple", simple_ud.columns))
    logging.info('simple metric: %s', metric)
    base_path = Path(r"PyProject\state_eval\tmp\pd_content\train\118")
    xgb_model_path = base_path.joinpath("bak/base/xgb_model.ubj")
    ohe_path = base_path.joinpath("ohe")
    data_up, data_ud, metric = predict_with_model(test_input, xgb_model_path=xgb_model_path, ohe_path=ohe_path,
                                                  train_field=train_field, dataset_type=MyDataSet)
    logging.info('data metric', metric)
    data_up = data_up.rename(columns=rename_column("batch_2_op", data_up.columns))
    data_ud = data_ud.rename(columns=rename_column("batch_2_op", data_ud.columns))

    base_path = Path(r"PyProject\state_eval\tmp\pd_content\train\125")
    xgb_model_path = base_path.joinpath("bak/base/xgb_model.ubj")
    ohe_path = base_path.joinpath("ohe")
    seq_data_up, seq_data_ud, metric = predict_with_model(test_input, xgb_model_path=xgb_model_path, ohe_path=ohe_path,
                                                          train_field=train_field, dataset_type=MyDataSet)
    logging.info('max seq len data metric', metric)
    seq_data_up = seq_data_up.rename(columns=rename_column("batch_seq_2_op", seq_data_up.columns))
    seq_data_ud = seq_data_ud.rename(columns=rename_column("batch_seq_2_op", seq_data_ud.columns))

    up = pd.merge(simple_up, data_up[[BATCH_PREFILL, *[k for k in data_up.columns if "predict" in k]]], \
                  on=BATCH_PREFILL, how="left", )
    up = pd.merge(up, seq_data_up[[BATCH_PREFILL, *[k for k in seq_data_up.columns if "predict" in k]]], \
                  on=BATCH_PREFILL, how="left", )
    ud = pd.merge(simple_ud, data_ud[[BATCH_DECODE, *[k for k in data_ud.columns if "predict" in k]]], \
                  on=BATCH_DECODE, how="left", )
    ud = pd.merge(ud, seq_data_ud[[BATCH_DECODE, *[k for k in seq_data_ud.columns if "predict" in k]]], \
                  on=BATCH_DECODE, how="left", )
    up_dfl = pd.melt(up, id_vars=[BATCH_PREFILL], value_name=train_field)
    sns.lineplot(up_dfl, x=BATCH_PREFILL, y=train_field, hue="variable", legend="brief")
    plt.savefig(base_path.joinpath("train_125_up.png"))
    plt.close()
    ud_dfl = pd.melt(ud, id_vars=[BATCH_DECODE], value_name=train_field, )
    sns.lineplot(ud_dfl, x=BATCH_DECODE, y=train_field, hue="variable", legend="brief")
    plt.savefig(base_path.joinpath("train_125_ud.png"))
    plt.close()


if __name__ == '__main__':
    manager()
