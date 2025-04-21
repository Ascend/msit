import os
import json
import pandas as pd
import time
import sqlite3
import re


# 定义一个函数来处理文件
def process_files(directory, out_path):
    # 获取所有的文件名
    files = [f for in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 对于每个function_execution_times_*.csv文件，找到对应的batch_*.csv文件
    for file in files:
        if file.startswith('batch_info_'):
            req_file = 'request_info_' + file[10:]
            batch_file = 'batch_need' + file[10:]
            req_out = 'request_need' + file[10:]
            forward_file = 'forward' + file[10:]
            forward_df = pd.read_csv(os.path.join(directory, forward_file), header=0)

            batch_df = pd.read_csv(os.path.join(directory, file), header=0)
            req_df = pd.read_csv(os.path.join(directory, req_file), header=0)
            column_names_1 = ['ibis_reqid', 'input_length', 'beed_blocks', 'output_length']
            column_names_2 = ['batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len',
                              'req_info', 'preprocess_time', 'forward_time', 'postprocess_time', 'forward_time',
                              'sample_time', 'postprocess_time', 'total_time', 'batch_id', 'batch_stage', 'batch_size',
                              'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'req_info', 'preprocess_time',
                              'forward_time', 'postprocess_time', 'forward_time', 'sample_time', 'postprocess',
                              'execute_time']
            column_names_3 = ['forward_time']

            req_df.columns = column_names_1
            forward_df.columns = column_names_3
            batch_df.columns = column_names_2
            req_df['execute_id'] = req_df['output_length']
            req_df = req_df.sort_values(by=['ibis_reqid', 'execute_id'], ascending=[True, True])
            batch_df['model_execute_time'] = forward_df['forward_time'] * 1000000
            # 重新排列列的顺序
            req_df = req_df[['ibis_reqid', 'execute_id', 'input_length', 'need_blocks', 'output_length']]
            # 将req_df的time列添加到batch_df中
            batch_df['ibis_reqid'] = batch_df.index
            batch_df = batch_df[
                ['ibis_batchid', 'batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len',
                 'model_execute_time', 'req_info']]
            # 保存处理后的文件
            batch_df.to_csv(os.path.join(out_path, batch_file), index=False)
            req_df.to_csv(os.path.join(out_path, req_out), index=False)

            print(f"Processed : {file}")


process_files('/data/deepseek/train', '/data/deepseek/output')
