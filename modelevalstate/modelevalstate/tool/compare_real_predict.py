# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import glob
from pathlib import Path
from typing import Optional
import pandas as pd
from pandas import DataFrame


from matplotlib import pyplot as plt

from modelevalstate.train.pretrain import NodeInfo, PreTrainModel
from modelevalstate.analysis import AnalysisState


def get_nodes(df: DataFrame, field_name: str):
    target_data = []
    for _, row in df.iterrows():
        _cur_node = NodeInfo(row.batch_stage, row.batch_size)
        setattr(_cur_node, field_name, row[field_name])
        target_data.append(_cur_node)
    return tuple(target_data)


def main():
    real_dir = r"D:\下载D\deepseek\collect_forward_info"
    field_name = "model_execute_time"
    fields = glob.glob(f"{real_dir}/batch_need_*.csv")
    _real_nodes = []
    for f in fields:
        df = pd.read_csv(f)
        _tmp_nodes = get_nodes(df, field_name)
        _real_nodes.extend(_tmp_nodes)
    _real_up, real_ud = PreTrainModel.get_up_down(tuple(_real_nodes), field_name)
    train_sleep = r"D:\下载D\deepseek\train_sleep\41"
    field_name = "model_execute_time"
    fields = glob.glob(f"{train_sleep}/batch_need_*.csv")
    _train_sleep_nodes = []
    for f in fields:
        df = pd.read_csv(f, header=None, names=['batch_stage', 'batch_size', 'total_need_blocks', \
            'total_prell_token', 'max_seq_len', 'reqinfo', 'model_execute_time', 'execute_time', \
            'start_time', 'end_time'])
        _tmp_nodes = get_nodes(df, field_name)
        _train_sleep_nodes.extend(_tmp_nodes)
    _train_sleep_up, train_sleep_ud = PreTrainModel.get_up_down(tuple(_train_sleep_nodes), field_name)
    AnalysisState.plot_input_velocity_with_predict(_real_up, _train_sleep_up, "batch_predill", \
        "up_of_real_and_train_sleep", "batch_predill", "velocity", Path(train_sleep))
    AnalysisState.plot_input_velocity_with_predict(_real_up, _train_sleep_up, "batch_decode", \
        "up_of_real_and_train_sleep", "batch_decode", "velocity", Path(train_sleep))

    train = r"D:\下载D\deepseek\train\41_3-6"
    field_name = "model_execute_time"
    fields = glob.glob(f"{train}/batch_info_*.csv")
    _train_sleep_nodes = []
    for f in fields:
        df = pd.read_csv(f, header=None,
                          names=['batch_stage', 'batch_size', 'total_need_blocks', 'total_prell_token', 'max_seq_len',
                                  'reqinfo', 'model_execute_time', 'execute_time', 'start_time'])
        
        df_simulate = pd.read_csv(Path(train).joinpath(f'simulate_{f.split("_")[-1].split(".")[0]}.csv'), header=None,
                                  names=['model_execute_time'])
        df[field_name] = df_simulate / (10**6)
        _tmp_nodes = get_nodes(df, field_name)
        _train_sleep_nodes.extend(_tmp_nodes)
    _train_sleep_up, train_sleep_ud = PreTrainModel.get_up_down(tuple(_train_sleep_nodes), field_name)
    AnalysisState.plot_input_velocity_with_predict(_real_up, _train_sleep_up, "batch_predill",
                                                    "up_of_real_and_train", "batch_predill", "velocity", Path(train))
    AnalysisState.plot_input_velocity_with_predict(_real_up, _train_sleep_up, "batch_decode",
                                                    "up_of_real_and_train", "batch_decode", "velocity", Path(train))
    

if __main__ == '__main__':
    main()
