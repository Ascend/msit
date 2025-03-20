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

import os
import json
import pandas as pd
import time
from glob import glob
import sqlite3
import re
# 定义一个函数来处理文件
def process_files(directory, out_path):
    # 获取所有的文件名
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in files:
        if file.startswith('batch_info_'):
            req_file = 'request_info' + file[10:]
            # 读取文件
            batch_df = pd.read_csv(os.path.join(directory, file), header=None)
            req_df = pd.read_csv(os.path.join(directory, req_file), header=None)
            column_names_1 = ['ibis_reqid', 'input_length', 'need_blocks', 'output_length']
            column_names_2 = ['batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'reqinfo', 'forward_time', 'execute_time', 'start_time']
            req_df.columns = column_names_1
            batch_df.columns = column_names_2
            req_df['execute_id'] = req_df['output_length']
            req_df = req_df.sort_values(by=['ibis_reqid', 'execute_id'], ascending=[True, True])

            # 重新排列列的顺序
            req_df = req_df[['ibis_reqid', 'execute_id', 'input_length', 'need_blocks', 'output_length']]
            # 将req_df的time列添加到batch_df中

            # 保存处理后的文件
            batch_df.to_csv(os.path.join(out_path, file), index=False)
            req_df.to_csv(os.path.join(out_path, req_file), index=False)



def filter_max_values_in_csv_files(path):
    # 获取指定目录下所有以'batch_info'开头的CSV文件
    files = [f for f in os.listdir(path) if f.startswith('batch_info')]

    # 初始化一个空的DataFrame，用于存储所有行
    _all_df = []
    for f in files:
        df = pd.read_csv(os.path.join(path, f))
        _all_df.append(df)
    _execute_time_df = pd.concat([_df["execute_time"] for _df in _all_df], axis=1)
    _execute_time_series = _execute_time_df.apply(lambda x: max(x), axis=1)
    _all_df[0]["model_execute_time"] = _execute_time_series

    return _all_df[0]


def find_first_file_starting_with(prefix, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                return os.path.join(root, file)
    return None

def analyze(input_path_1,input_path_2,input_path_3):
    process_files(input_path_1, input_path_2)

    db_path = glob(f"{input_path_3}/**/benchmark_data.db", recursive=True)[0]

    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)

    # 读取SQL查询结果并转换成DataFrame
    df3 = pd.read_sql("SELECT * FROM result", conn)

    # 关闭连接
    conn.close()

    # 对start_time列进行升序排序
    df3 = df3.sort_values(by='start_time')
    df3.reset_index(inplace=True)
    df3['request'] = df3.index + 1
    total_req = df3.shape[0]
    filtered_df = df3[df3['success'] == 1]
    success_req = filtered_df.shape[0]
    filtered_df.reset_index(inplace=True)

    df1 = filter_max_values_in_csv_files(input_path_2)
    prefix = 'simulate_'
    first_file = find_first_file_starting_with(prefix, input_path_1)
    if first_file:
        df2 = pd.read_csv(first_file, header=None)
    else:
        raise FileNotFoundError(f'在{input_path_1}目录下没有找到以{prefix}开头的文件')
    column_names = ['simulate_time']

    # 将指定的列名赋值给DataFrame
    df2.columns = column_names
    # 确认两文件的行数相同
    if len(df1) != len(df2):
        raise ValueError("两个CSV文件的行数必须相同")

    # 将第二个CSV文件中的`during_time`列添加到第一个CSV文件中，并修改列名
    df1['simulate_time'] = df2['simulate_time'] / 10**6
    second_prefill_row = df1[df1['batch_stage'] == 'prefill'].iloc[1]

    # 获取该行的索引
    index_to_delete = second_prefill_row.name
    total_simulate_time = 0

    # 删除该行之前的所有行
    df1 = df1.loc[df1.index >= index_to_delete]

    df_prefill = df1[df1['batch_stage'] == 'prefill']
    for index, row in df_prefill.iterrows():

        digits = re.findall(r'\d+', row['reqinfo'])

            # 将这些字符串转换为整数列表
        reqinfo_list = [int(num) for num in digits]

            # 使用列表推导式筛选偶数位的数字
        non_zero_values = [x for i, x in enumerate(reqinfo_list) if i % 2 == 0]
            # 遍历这些值
        for val in non_zero_values: 

            if val>success_req:
                break
            during_time = filtered_df.iloc[val-1]['first_chunk_latency'] 
            arrive_time = row['start_time'] - during_time
            filtered_df1 = df1[df1['start_time'] > arrive_time]

            # 进一步筛选出 start_time 小于等于 row['start_time'] 的行
            filtered_df1 = filtered_df1[filtered_df1['start_time'] <= row['start_time']]

            # 计算 simulate_time 的总和
            total_simulate_time += filtered_df1['simulate_time'].sum()
                    

    total_latency = filtered_df['first_chunk_latency'].sum() + total_simulate_time
    total_token = filtered_df['completion_tokens'].sum()
    avg_prefill_latency = total_latency / success_req
    total_time = filtered_df['completed_time'].max() - filtered_df['start_time'].min() + df1['simulate_time'].sum()
    throughput = total_token / total_time
    total_decode_time = filtered_df['latency'].sum() - filtered_df['first_chunk_latency'].sum() 
    average_decode_latency = total_decode_time / filtered_df['n_chunks'].sum()
    success_precent = success_req / total_req
    print(throughput, avg_prefill_latency, average_decode_latency, success_precent)
    return  throughput, avg_prefill_latency, average_decode_latency, success_precent