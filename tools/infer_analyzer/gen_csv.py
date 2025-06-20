import numpy as np
import pandas as pd
import logging
import sys
import os
from load_file import FdOpen

logger=logging.getLogger("infer_analyze")

class Generator:
    def __init__(self, op_data,batch_data, output_path):
        self.sheet1 = None
        self.sheet2 = None
        self.sheet3 = None
        self.output_path = output_path
        self.op_data = op_data
        self.batch_data = batch_data


    def _prefill_and_decode_time(self):
        self.sheet1 = self.op_data[["Index", "Stage", "Start Time(ms)", "Duration(ms)", "Duration Ratio"]][
            self.op_data["Split"]]

        if self.batch_data is not None and len(self.op_data) == len(self.batch_data):
            self.op_data["Batch Size"] = self.batch_data["batch_size"].tolist()

    def _time_statistics(self, lst):
        def overlap(laped_df):

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
            total_duration = df['Start Time(us)'].iloc[-1] + df['Duration(us)'].iloc[-1] - df['Start Time(us)'].iloc[0]
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

        def gen_sheet(df):
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
                sys.exit(0)
        else:
            os.makedirs(self.output_path)
            self.output_path = os.path.join(self.output_path, "output.xlsx")
        if not FdOpen.check(self.output_path):
            sys.exit(0)
        try:
            with pd.ExcelWriter(self.output_path) as writer:
                self.sheet1.to_excel(writer, sheet_name='prefill and decode time', index=False)
                self.sheet2.to_excel(writer, sheet_name='time statistics', index=False)
                self.sheet3.to_excel(writer, sheet_name='op statistics', index=False)
        except Exception as e:
            logger.error(e)
            sys.exit(0)


    def _choose_stage(self):
        idx_list = []
        name_list = []
        if not self.op_data[self.op_data["Stage"] == "decode"].empty:
            max_decode = self.op_data[self.op_data["Stage"] == "decode"]['Duration(ms)'].idxmax()  # 最大值的索引
            min_decode = self.op_data[self.op_data["Stage"] == "decode"]['Duration(ms)'].idxmin()  # 最小值的索引

            median_decode = self.op_data[self.op_data["Stage"] == "decode"]['Duration(ms)'].median()
            self.op_data['diff'] = np.abs(self.op_data['Duration(ms)'] - median_decode)
            mid_decode = self.op_data[self.op_data["Stage"] == "decode"]['diff'].idxmin()

            max_idx = self.op_data.loc[max_decode, "Index"]
            min_idx = self.op_data.loc[min_decode, "Index"]
            mid_idx = self.op_data.loc[mid_decode, "Index"]

            idx_list.extend([max_decode, min_decode, mid_decode])
            name_list.extend([f"Maximum Decode(No.{max_idx})", f"Minium Decode(No.{min_idx})",
                              f"Medium Decode(No.{mid_idx})"])

        if not self.op_data[self.op_data["Stage"] == "prefill"].empty:
            max_prefill = self.op_data[self.op_data["Stage"] == "prefill"]['Duration(ms)'].idxmax()  # 最大值的索引
            min_prefill = self.op_data[self.op_data["Stage"] == "prefill"]['Duration(ms)'].idxmin()  # 最小值的索引

            median_prefill = self.op_data[self.op_data["Stage"] == "prefill"]['Duration(ms)'].median()
            self.op_data['diff'] = np.abs(self.op_data['Duration(ms)'] - median_prefill)
            mid_prefill = self.op_data[self.op_data["Stage"] == "prefill"]['diff'].idxmin()
            idx_list.extend([max_prefill, min_prefill, mid_prefill])

            max_idx = self.op_data.loc[max_prefill, "Index"]
            min_idx = self.op_data.loc[min_prefill, "Index"]
            mid_idx = self.op_data.loc[mid_prefill, "Index"]

            idx_list.extend([max_prefill, min_prefill, mid_prefill])
            name_list.extend([f"Maximum Prefill(No.{max_idx})", f"Minium Prefill(No.{min_idx})",
                              f"Medium Prefill(No.{mid_idx})"])

        split_indices = self.op_data[self.op_data['Split']].index.tolist()

        result = []

        for idx, name in zip(idx_list, name_list):
            for i, val in enumerate(split_indices):
                if val == idx:
                    if i == 0:
                        temp = self.op_data.loc[0:val + 1].reset_index().copy()
                    else:
                        temp = self.op_data.loc[split_indices[i - 1]:val + 1].reset_index().copy()
                    result.append((temp, name))

        return result

