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


import logging
import ast
import re

import ijson
import numpy as np
import pandas as pd

logger = logging.getLogger("infer_analyze")


class ProfileParser:
    def __init__(self, op_data, batch_data, json_path):
        self.op_data = op_data
        self.batch_data = batch_data
        self.json_path = json_path

    def run(self):
        self._pre_process()
        self._split_performance_time()
        self._determine_stage()
        self._count_batch()
        return self.op_data, self.batch_data

    def _pd_split(self):
        """
        查询json中是否有decode和prefill字段，如果只有一个说明是pd分离场景
        """
        decode_status = False
        prefill_status = False
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                for obj in ijson.items(f, 'item'):
                    if not decode_status and "Decode" in obj.get("name", ""):
                        decode_status = True
                    if not prefill_status and "Prefill" in obj.get("name", ""):
                        prefill_status = True
                if decode_status and not prefill_status:
                    return "decode"
                elif prefill_status and not decode_status:
                    return "prefill"
                else:
                    return "both"
        except Exception as e:

            return "both"

    def _pre_process(self):
        """
        对数据预处理，把不同格式的csv变成同一格式的dataframe。
        根据op name做正则判断是什么类型算子
        """
        self.op_data.rename(
            columns={
                'Op Name': 'Name',
                'OP Type': 'Type',
                'Task Type': 'Accelerator Core',
                'Task Start Time(us)': 'Start Time(us)',
                'Task Duration(us)': 'Duration(us)'
            },
            inplace=True
        )
        # 确保 Start Time(us) 列是数值类型
        self.op_data['Start Time(us)'] = pd.to_numeric(self.op_data['Start Time(us)'])
        self.op_data['Duration(us)'] = pd.to_numeric(self.op_data['Duration(us)'])
        self.op_data['time_diff'] = self.op_data['Start Time(us)'].shift(-1) - 
            self.op_data['Start Time(us)'] - self.op_data['Duration(us)']

        conditions = [
            self.op_data['Name'].str.contains(r"^(?:lcoc|lcom|lccl).*[_0-9]$", case=False, na=False),
            self.op_data['Name'].str.contains(r'matmul', case=False, na=False),
            self.op_data['Name'].str.contains(r'attention', case=False, na=False),
            self.op_data['Name'].str.contains(r'tensormove', case=False, na=False)]

        choices = ['Communication', 'Matmul', "attention", "Memory"]
        self.op_data['Core Type'] = np.select(conditions, choices, default='Vector')

        if self.op_data["Type"].isna().any():
            self.op_data["Type"] = self.op_data["Name"].str.replace(r'[V_0-9]+$', '', regex=True)

    def _split_performance_time(self):
        """
        确定两个stage的分界点在哪里。
        如果遇到算子名在mask内的，且结束时间距离下一个算子的开始时间大于200us，则标记为一个stage的结尾算子。
        """
        self.op_data['Split'] = False
        mask = (self.op_data['Name'] == 'aclnnArgMax_CastAiCore_Cast') | 
            (self.op_data['Name'] == 'aclnnTopk_CastAiCore_Cast')

        groups = (~mask).cumsum()
        mask_starts = self.op_data.index[mask]

        for start in mask_starts:
            subset = self.op_data.loc[start:]
            first_match = subset[(subset['time_diff'] > 200) | (subset['time_diff'].isna())].index.min()
            if first_match is not None:
                self.op_data.loc[first_match, 'Split'] = True

        self.op_data.at[self.op_data.index[-1], "Split"] = True

    def _determine_stage(self):
        """
        判断一个stage是decode还是prefill。
        在一个stage中，如果有flash attention算子，则把整个stage标记为prefill
        在一个stage中，如果有paged attention算子，则把整个stage标记为decode
        如果找不到则根据stage的算子密度区分
        """
        #如果是pd分离场景，则直接所有stage都相同
        pd_tmp = self._pd_split()
        if pd_tmp != "both":
            self.op_data["Stage"] = pd_tmp

        # 获取所有 Split=True 的行的索引
        split_indices = self.op_data[self.op_data['Split']].index.tolist()
        total_duration = self.op_data['Start Time(us)'].iloc[-1] + self.op_data['Duration(us)'].iloc[-1] - \
                         self.op_data['Start Time(us)'].iloc[0]
        start_row = 0

        for idx in split_indices:
            if pd_tmp == "both":
                has_prefill = self.op_data.loc[start_row:idx + 1, "Name"].str.contains(
                    r'(?i)flash.*attention', regex=True).any()
                has_decode = (self.op_data.loc[start_row:idx + 1, "Name"].str.contains(
                    r'(?i)paged.*attention', regex=True).any()
                              or self.op_data.loc[start_row:idx + 1, "Name"].str.contains("MLAKernel").any())

                if has_prefill and not has_decode:
                    self.op_data.loc[start_row:idx + 1, "Stage"] = "prefill"
                elif has_decode and not has_prefill:
                    self.op_data.loc[start_row:idx + 1, "Stage"] = "decode"
                else:
                    self.op_data.loc[start_row:idx + 1, "Stage"] = "other"

            time_diff = self.op_data.loc[idx, 'Start Time(us)'] + 
                self.op_data.loc[idx, 'Duration(us)'] - self.op_data.loc[start_row, 'Start Time(us)']

            if 'Task ID' in self.op_data.columns:
                task_id_diff = self.op_data.loc[idx, 'Task ID'] - self.op_data.loc[start_row, 'Task ID']
                density = task_id_diff / time_diff if task_id_diff != 0 else float('inf')
                self.op_data.loc[start_row:idx + 1, 'density'] = density
                
            self.op_data.loc[idx, "Duration(ms)"] = time_diff / 1000
            self.op_data.loc[idx, "Start Time(ms)"] = str(round(self.op_data.loc[start_row, 'Start Time(us)'] / 1000))
            self.op_data.loc[idx, "Duration Ratio"] = f"{round(time_diff / total_duration * 100, 3)}%"
            start_row = idx + 1

        if (self.op_data["Stage"].eq("other")).all() and "Task ID" in self.op_data.columns:
            density_mean = self.op_data["density"].mean()
            self.op_data.loc[self.op_data["density"] <= density_mean, "Stage"] = "prefill"
            self.op_data.loc[self.op_data["density"] > density_mean, "Stage"] = "decode"

        self.op_data['Stage_index'] = np.nan
        mask_decode = self.op_data['Split'] & (self.op_data["Stage"] == "decode")
        mask_prefill = self.op_data['Split'] & (self.op_data["Stage"] == "prefill")

        #获取每个stage的序号
        self.op_data.loc[mask_decode, 'Index'] = range(1, mask_decode.sum() + 1)
        self.op_data.loc[mask_prefill, 'Index'] = range(1, mask_prefill.sum() + 1)
        self.op_data["Index"] = self.op_data["Index"].fillna(-1).astype(int)

    def _count_batch(self):
        """
        如果有msprof_tx_xxxxx.json，则尝试用正则寻找batch size
        """
        if self.batch_data is None:
            return
        # 初始化结果列
        self.batch_data['span_number'] = None

        current_span = None
        i = 0
        while i < len(self.batch_data):
            message = self.batch_data.at[i, 'message']

            # 匹配主模式: span=数字*{...dpBatch...
            match = re.match(r'^span=(\d+)\*\{', message)
            if match and 'dpBatch' in message:
                current_span = int(match.group(1))
                self.batch_data.at[i, 'span_number'] = current_span

                # 向下查找连续的同 span 行
                j = i + 1
                while j < len(self.batch_data):
                    next_message = self.batch_data.at[j, 'message']
                    if next_message.startswith(f'span={current_span}*'):
                        self.batch_data.at[j, 'span_number'] = current_span
                        j += 1
                    else:
                        break
                i = j
            else:
                i += 1
                continue

        def process_group(group):
            # 去除每行 message 的 "span=数字*"
            cleaned = group['message'].str.replace(r'^span=\d+\*', '', regex=True)
            cleaned = cleaned.str.replace('^', '\"', regex=False)
            # 合并所有 message 内容为一个字符串
            messages = ''.join(cleaned)
            return messages

        # 按 span_number 分组聚合
        self.batch_data = self.batch_data.groupby('span_number', group_keys=False).apply(process_group)
        if not self.batch_data.empty:
            self.batch_data = self.batch_data.reset_index(name="message")
            self.batch_data["batch_size"] = self.batch_data["message"].apply(lambda x: len(ast.literal_eval(x)["rid"]))
        else:
            logger.info("Fail to get batch size info")
            self.batch_data = None
