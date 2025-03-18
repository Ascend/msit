# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import json
import logging
import pandas as pd

has_orjson = False
try:
    import orjson
    has_orjson = True
except ImportError:
    pass

req_map = {}


def load_file(ff, auto_add_braces=False):
    if auto_add_braces:
        text = f"[{ff.read()}null]"
    else:
        text = ff.read()
    if has_orjson:
        return orjson.loads(text)
    else:
        return json.loads(text)


def log2csv(input_path_2):
    """
    解析所有log文件
    """
    small_file_list = []
    for _, _, files in os.walk(input_path_2):
        for f in files:
            if f.endswith(".log"):
                with open(file=os.path.join(input_path_2, f), mode="r", encoding="utf-8") as file:
                    data = load_file(file, auto_add_braces=True)
                    small_file_list.extend(data)  # yaml.load(stream=file, Loader=yaml.FullLoader)
    df_small_file = pd.json_normalize(small_file_list)
    df_small_file.to_csv(os.path.join(input_path_2, "all_data.csv"), index=0)


def concat_func(x):
    return pd.Series({
        'reqInfo': max(x['reqInfo'].unique(), key=len),
        'batch_used_block': x['batch_used_block'].max(),
        'total_token_num': x['total_token_num'].max(),
        'max_seq_len': x['max_seq_len'].max(),
        'queue_size': x['queue_size'].max(),
    })


def parse_batch_info(file1):
    batch_e = file1[file1["batch_id"].notnull()].dropna(axis=1, how="all").reset_index(drop=True).sort_values(
        by=["batch_id", "ts"])
    batch_execute = batch_e.groupby("batch_id").apply(concat_func).reset_index()

    model_e = file1[(file1["name"] == "model_execute") & (file1["timing"] == "E")].dropna(axis=1, how="all")
    model_e = model_e.reset_index(drop=True).rename(
        columns={"ts": "ts_model_execute_end", "sts": "ts_model_execute_begin"})
    model_execute = model_e

    batch_info = pd.merge(batch_execute, model_execute, on="reqInfo").reset_index()
    return batch_info


def determine_stage(row):
    if row["p_count"] > 0:
        return "prefill"
    else:
        return "decode"


def get_batch_need(batch_info):
    batch_info["req_ids"] = batch_info.apply(lambda x: x["reqInfo"][1:-1].split(", ")[0::2], axis=1)
    batch_info["execute_ids"] = batch_info.apply(lambda x: x["reqInfo"][1:-1].split(", ")[1::2], axis=1)
    batch_info["p_vector"] = batch_info.apply(
        lambda x: [int(a) for a, b in zip(x["req_ids"], x["execute_ids"]) if int(b) == 0], axis=1)
    batch_info["d_vector"] = batch_info.apply(
        lambda x: [int(a) for a, b in zip(x["req_ids"], x["execute_ids"]) if int(b) != 0], axis=1)
    batch_info["p_count"] = batch_info.apply(lambda x: len(x["p_vector"]), axis=1)
    batch_info["d_count"] = batch_info.apply(lambda x: len(x["d_vector"]), axis=1)
    batch_info["batch_stage"] = batch_info.apply(determine_stage, axis=1)
    batch_info["batch_size"] = batch_info.apply(lambda x: x["p_count"] + x["d_count"], axis=1)
    batch_info["model_execute_time"] = batch_info["ts_model_execute_end"] - batch_info["ts_model_execute_begin"]
    batch_info = batch_info.rename(columns={"batch_used_block": "total_need_blocks"})
    batch_info = batch_info.rename(columns={"total_token_num": "total_prefill_token"})
    batch_info = batch_info.rename(columns={"batch_id": "ibis_batchid"})
    batch_info = batch_info.rename(columns={"reqInfo": "req_info"})
    batch_info['batch_token'] = 0
    for index, row in batch_info.iterrows():
        if row['batch_stage'] == 'prefill':
            batch_info.at[index, 'batch_token'] = row['total_prefill_token']
        elif row['batch_stage'] == 'decode':
            execute_ids = [int(id) for id in row['execute_ids']]
            batch_token = sum(execute_ids)
            batch_info.at[index, 'batch_token'] = batch_token
    return batch_info


def analyze(input_path_1, input_path_2):
    # 1. parse log files to csv files using yaml
    log2csv(input_path_2)
    logging.info("log2csv end!")
    # 2. parse csv files for needed info
    ## read in all csv files
    file1 = pd.read_csv(os.path.join(input_path_2, "all_data.csv"), dtype={
        'event': str,
        'http_reqid': str,
        'func_name': str,
        'name': str,
        'free_block_list': str,
        'used_block_list': str,
    })

    ## get tmp csv files
    batch_info = parse_batch_info(file1)  # merge batch info (batch_execute and model_execute)
    batch_need = get_batch_need(batch_info)

    # 读取第二个CSV文件
    df2 = pd.read_csv(input_path_1, header=None)
    column_names = ['name', 'sts', 'ts', 'during_time', 'simulate_time', 'total_time']

    # 将指定的列名赋值给DataFrame
    df2.columns = column_names
    # 确认两文件的行数相同
    if len(batch_need) != len(df2):
        raise ValueError("两个CSV文件的行数必须相同")

    # 将第二个CSV文件中的`during_time`列添加到第一个CSV文件中，并修改列名
    batch_need['total_execute_time'] = batch_need['model_execute_time'] + df2['simulate_time']

    # 输出合并后的数据到新的CSV文件
    decode_token_num = 0
    decode_total_execute_time = 0
    prefill_token_num = 0
    prefill_total_execute_time = 0
    decode_total_request_time = 0
    # 遍历DataFrame
    for _, row in batch_need.iterrows():
        if row['batch_stage'] == 'decode':
            decode_token_num += row['batch_size']
            decode_total_execute_time += row['total_execute_time']
            decode_total_request_time += row['batch_size'] * row['total_execute_time']
        elif row['batch_stage'] == 'prefill':
            prefill_token_num += row['batch_size']
            prefill_total_execute_time += row['total_execute_time'] * row['batch_size']

    # 计算平均时间
    if decode_token_num != 0:
        avg_time = decode_token_num / decode_total_execute_time * 1000000
        avg_decode_time = decode_total_request_time / decode_token_num / 1000000
    else:
        avg_time = 0  # 或者其他你认为合适的值，例如np.inf，如果需要特殊处理
    if prefill_token_num != 0:
        avg_prefill_token = prefill_total_execute_time / prefill_token_num / 1000000
    return avg_time, avg_prefill_token, avg_decode_time
