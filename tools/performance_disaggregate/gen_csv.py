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


import sys
import os
import logging
import numpy as np
import pandas as pd
from load_file import FdOpen

logger = logging.getLogger("infer_analyze")


class Generator:
    def __init__(self, op_data, batch_data, output_path):
        self.sheet1 = None
        self.sheet2 = None
        self.sheet3 = None
        self.output_path = output_path
        self.op_data = op_data
        self.batch_data = batch_data

    def _prefill_and_decode_time(self):
        """
        生成第一个sheet
        """
        self.sheet1 = self.op_data[["Index", "Stage", "Start Time(ms)", "Duration(ms)", "Duration Ratio"]][
            self.op_data["Split"]]

        if self.batch_data is not None and len(self.sheet1) == len(self.batch_data):
            self.sheet1["Batch Size"] = self.batch_data["batch_size"].tolist()

    def _time_statistics(self, lst):
        """
        生成第二个sheet
        """
        def overlap(laped_df):
            """
            计算一个stage中的通信掩盖总时长
            """
            laped_df["end"] = laped_df["Start Time(us)"] + laped_df['Duration(us)']
            compute_inter = laped_df[laped_df["Core Type"] == "Communication"][["Start Time(us)", 'end']].to_numpy()
            commun_inter = laped_df[laped_df["Core Type"] != "Communication"][["Start Time(us)", 'end']].to_numpy()

            i = j = 0
            total = 0.0
            while i < len(compute_inter) and j < len(commun_inter):
                c_start, c_end = compute_inter[i]
                t_start, t_end = commun_inter[j]

                left = max(c_start, t_start)
                right = min(c_end, t_end)
                if left < right:
                    total += right - left
                if c_end < t_end:
                    i += 1
                else:
                    j += 1
            return total

        result = []
        for df, name in lst:
            #对list中的每个stage聚合信息，把聚合后的结果加入result中
            total_duration = df['Start Time(us)'].iloc[-1] + df['Duration(us)'].iloc[-1] - df['Start Time(us)'].iloc[0]
            df.loc[(df["Core Type"] == "Communication") & (df["Duration(us)"] > total_duration), "Duration(us)"] = 0
            grouped = df.groupby('Core Type').agg({
                'Duration(us)': ['sum', "count"],
            })
            grouped.columns = ["Duration(ms)", "Number"]
            grouped['Ratio(%)'] = (grouped['Duration(ms)'] / total_duration).apply(lambda x: f"{round(x * 100, 3)}%")
            grouped["Duration(ms)"] = grouped["Duration(ms)"].apply(lambda x: round(x / 1000, 2))
            row_idx = ["Matmul", "Vector", "Communication"]
            grouped = grouped[['Duration(ms)', 'Ratio(%)', 'Number']].reindex(row_idx).reset_index()

            uncovered_communication = df[df["Core Type"] == "Communication"]["Duration(us)"].sum() - overlap(df)
            free = total_duration - uncovered_communication - df[df["Core Type"] != "Communication"][
                "Duration(us)"].sum()

            for idx, data in zip(["Uncovered Communication", "Free", "Total Duration"],
                                 [uncovered_communication, free, total_duration]):
                new_row = [idx, round(data / 1000, 2), f"{round(data / total_duration * 100, 2)}%", "/"]
                grouped.loc[len(grouped)] = new_row
            grouped.loc[len(grouped)] = np.nan
            grouped.insert(0, "Index", None)
            grouped.loc[0, "Index"] = name
            result.append(grouped)
        self.sheet2 = pd.concat(result, ignore_index=True, copy=False)

    def _op_statistics(self, lst):
        """
        生成第三个sheet
        """

        def gen_sheet(df):
            #聚合stage中的信息
            total_duration = df['Start Time(us)'].iloc[-1] + df['Duration(us)'].iloc[-1] - df['Start Time(us)'].iloc[0]
            grouped = df.groupby(['Type', "Core Type"]).agg({
                'Type': 'count',
                'Duration(us)': ['sum', 'max', 'min', 'mean'],  # Duration 的和、最大值、最小值、平均值
            }).round(2)


            grouped.columns = ["Count", 'Total Time', 'Min Time(us)', "AVG Time(us)", 'Max Time(us)']
            grouped = grouped.reset_index()
            drop_idx = []
            for i in range(len(grouped) - 1):
                if grouped.loc[i, "Type"] == grouped.loc[i + 1, "Type"]:
                    grouped.loc[i, "Core Type"] = "Communication and Compute"
                    drop_idx.append(i + 1)

            grouped = grouped.drop(index=drop_idx).reset_index(drop=True)
            grouped['Ratio(%)'] = grouped['Total Time'] / grouped['Total Time'].sum()
            grouped = grouped.sort_values('Ratio(%)', ascending=False)

            other_row = ['Other', '/'] + grouped[grouped['Total Time'] / grouped['Total Time'].sum() <= 0.01].iloc[:,
                                         2:].sum().tolist()
            grouped = grouped[grouped['Total Time'] / grouped['Total Time'].sum() > 0.01].reset_index(drop=True)
            grouped.loc[len(grouped)] = other_row
            grouped['Ratio(%)'] = grouped['Ratio(%)'].apply(lambda x: f"{round(x * 100, 3)}%")

            return grouped

        result = []
        for df, name in lst:
            temp = gen_sheet(df)
            temp.loc[len(temp)] = np.nan
            temp.insert(0, "Index", None)
            temp.loc[0, "Index"] = name
            result.append(temp)

        self.sheet3 = pd.concat(result, ignore_index=True, copy=True)

    def generate_excel(self):
        """
        生成最终excel
        """
        self._prefill_and_decode_time()
        self._time_statistics(self._choose_stage())
        self._op_statistics(self._choose_stage())

        if not self.output_path:
            self.output_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(self.output_path):
            if os.path.isdir(self.output_path):
                self.output_path = os.path.join(self.output_path, "output.xlsx")
            else:
                logger.error("Output path must be direction")
        else:
            os.makedirs(self.output_path)
            self.output_path = os.path.join(self.output_path, "output.xlsx")
        if not FdOpen.check(self.output_path):
        try:
            with pd.ExcelWriter(self.output_path) as writer:
                self.sheet1.to_excel(writer, sheet_name='prefill and decode time', index=False)
                self.sheet2.to_excel(writer, sheet_name='time statistics', index=False)
                self.sheet3.to_excel(writer, sheet_name='op statistics', index=False)
        except Exception as e:
            logger.error(e)

    def _choose_stage(self):
        """
        根据所有stage的耗时，选出耗时最大/最小/中位数的代表
        """
        idx_list = []
        name_list = []
        for stage in ["Decode", "Prefill"]:
            stage_low = stage.lower()

            if not self.op_data[self.op_data["Stage"] == stage_low].empty:
                filtered = self.op_data[self.op_data["Stage"] == stage_low]['Duration(ms)'].dropna()

                sorted_filtered = filtered.sort_values().reset_index()
                filtered_length = len(sorted_filtered)

                min_pos = 0
                max_pos = filtered_length - 1
                mid_pos = int(filtered_length * 0.5)
                q25_pos = int(filtered_length * 0.25)
                q75_pos = int(filtered_length * 0.75)

                min_idx = sorted_filtered.iloc[min_pos]['index']
                max_idx = sorted_filtered.iloc[max_pos]['index']
                mid_idx = sorted_filtered.iloc[mid_pos]['index']
                q25_idx = sorted_filtered.iloc[q25_pos]['index']
                q75_idx = sorted_filtered.iloc[q75_pos]['index']

                max_num = self.op_data.loc[max_idx, "Index"]
                min_num = self.op_data.loc[min_idx, "Index"]
                mid_num = self.op_data.loc[mid_idx, "Index"]
                q25_num = self.op_data.loc[q25_idx, "Index"]
                q75_num = self.op_data.loc[q75_idx, "Index"]

                if filtered_length == 1:
                    idx_list.append(min_idx)
                    name_list.append(f"Only {stage}(No.{min_num})")
                elif filtered_length <= 5:
                    idx_list.extend([max_idx, mid_idx, min_idx])
                    name_list.extend([f"Maximum {stage}(No.{max_num})",
                                      f"Q2(Medium) {stage}(No.{mid_num})", f"Minium {stage}(No.{min_num})"])
                else:
                    idx_list.extend([max_idx, q75_idx, mid_idx, q25_idx, min_idx])
                    name_list.extend([f"Maximum {stage}(No.{max_num})", f"Q3 {stage}(No.{q75_num})",
                                      f"Q2(Medium) {stage}(No.{mid_num})", f"Q1 {stage}(No.{q25_num})",
                                      f"Minium {stage}(No.{min_num})"])


        split_indices = self.op_data[self.op_data['Split']].index.tolist()
        result = []

        for idx, name in zip(idx_list, name_list):
            for i, val in enumerate(split_indices):
                if val == idx:
                    if i == 0:
                        temp = self.op_data.iloc[0:val + 1].reset_index().copy()
                    else:
                        temp = self.op_data.iloc[split_indices[i - 1] + 1:val + 1].reset_index().copy()
                    result.append((temp, name))

        return result

