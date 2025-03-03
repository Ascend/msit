# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import os
import time
import copy
import logging
import sqlite3
from pathlib import Path
from concurrent import futures
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

DB_PATTERN = "ascend_pytorch_profiler_*.db"

QUERY_SQL = """
SELECT
    rdm.rankId,
    si.value AS groupName,
    co.endNs - co.startNs AS communication_time,
    sii.value AS opName,
    host.hostUid as host_id,
    op.value AS opType,
    et.name AS dataType,
    CASE
WHEN et.name = 'INT8' THEN 1 * co.count 
    WHEN et.name = 'INT16' THEN 2 * co.count
    WHEN et.name = 'INT32' THEN 4 * co.count
    WHEN et.name = 'INT64' THEN 8 * co.count
    WHEN et.name = 'UINT64' THEN 8 * co.count 
    WHEN et.name = 'UINT8' THEN 1 * co.count
    WHEN et.name = 'UINT16' THEN 2 * co.count
    WHEN et.name = 'UINT32' THEN 4 * co.count
    WHEN et.name = 'FP16' THEN 2 * co.count
    WHEN et.name = 'FP32' THEN 4 * co.count
    WHEN et.name = 'FP64' THEN 8 * co.count 
    WHEN et.name = 'BFP16' THEN 2 * co.count
    WHEN et.name = 'INT128' THEN 16 * co.count 
    END AS dataSize
FROM
    COMMUNICATION_OP co
CROSS
    JOIN RANK_DEVICE_MAP rdm
    JOIN STRING_IDS si ON co.groupName = si.id
    JOIN STRING_IDS sii ON co.opName = sii.id
    JOIN ENUM_HCCL_DATA_TYPE et ON co.dataType = et.id
    JOIN STRING_IDS op ON co.opType = op.id 
    JOIN HOST_INFO host
"""

DIXON_95_TABLE = {
    3: 0.941,
    4: 0.765,
    5: 0.642,
    6: 0.562,
    7: 0.507,
    8: 0.554,
    9: 0.512,
    10: 0.477,
    11: 0.575,
    12: 0.546,
    13: 0.521,
    14: 0.546,
    15: 0.524,
    16: 0.505,
    17: 0.489,
    18: 0.475,
    19: 0.462,
    20: 0.450,
    21: 0.440,
    22: 0.431,
    23: 0.422,
    24: 0.413,
    25: 0.406,
    26: 0.399,
    27: 0.393,
    28: 0.387,
    29: 0.381,
    30: 0.376
}


class MultiProcessor:
    def __init__(self):
        self._executor = futures.ProcessPoolExecutor(max_workers=os.cpu_count())

    def __enter__(self):
        if self._executor is None:
            raise RuntimeError("executor is None")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def launch(self, func, *args, **kwargs):
        return self._executor.submit(func, *args, **kwargs).result()

    def map(self, func, *iterables, **kwargs):
        partial_func = partial(func, **kwargs)
        return list(self._executor.map(partial_func, *iterables))


class DBLoader:
    def __init__(self, input_path):
        """
        递归查找指定目录下所有匹配的数据库文件 并将每个数据库文件的查询结果合并到一个DataFrame中
        输出一个包含所有数据的DataFrame
        """
        self.path = Path(input_path)
        self.all_json_objects = pd.DataFrame([])

        self.path_list = [db_file for db_file in self.path.rglob(DB_PATTERN) if db_file.is_file()]
        self.run()

    @staticmethod
    def mapper_func(path_map):
        json_obj = connect_and_process_sql(path_map, QUERY_SQL)
        json_obj = pd.DataFrame(json_obj)
        return json_obj

    def concat_db(self, db_list):
        df_list = [json_obj for json_obj in db_list if json_obj is not None]
        self.all_json_objects = pd.concat(df_list, ignore_index=True)

    def run(self):
        t1 = time.time()
        logging.info("loading data from db...")

        with MultiProcessor() as executor:
            mapper_res = executor.map(
                self.mapper_func,
                self.path_list,
            )

        self.concat_db(mapper_res)
        t2 = time.time()
        logging.info(f"loading cost time {t2 - t1}s")


class DBProcessor:
    def __init__(self, dfs):
        self.dfs = dfs
        self.rankid_arr = dfs["rankId"].values
        self.num_ranks = len(self.rankid_arr)
        self.group_name_arr = dfs["groupName"].values
        self.communication_time_arr = dfs["communication_time"].values
        self.op_name_arr = dfs["opName"].values
        self.host_id_arr = dfs["host_id"].values
        self.vote_result = defaultdict(lambda: [0, 0])
        self.slow_link_sum = []
        self.slow_link_ops = []

    def run(self):
        process_group = defaultdict(lambda: defaultdict(list))
        transmit_time_arr = np.zeros(self.num_ranks, dtype=np.int64)
        related_ranks_arr = np.zeros(self.num_ranks, dtype=np.int32)
        across_nodes_arr = np.zeros(self.num_ranks, dtype=np.bool_)

        for idx in range(self.num_ranks):
            if "send" in self.op_name_arr[idx] or "receive" in self.op_name_arr[idx]:
                continue
            process_group[self.group_name_arr[idx]][self.op_name_arr[idx]].append(idx)

        for _, ops_same_group in tqdm(process_group.items(), desc="Processing DB data..."):
            for _, ops in ops_same_group.items():
                communication_time_list = [self.communication_time_arr[op_idx] for op_idx in ops]
                transmit_time = min(communication_time_list)

                judge_flag, outlier_idx, vote_string = judge_vote(communication_time_list)

                if judge_flag:
                    op_idx = ops[outlier_idx]
                    self.vote_result[self.rankid_arr[op_idx]][0] += 1

                for op_idx in ops:
                    self.vote_result[self.rankid_arr[op_idx]][1] += 1

                across_nodes_flag = len(set([self.host_id_arr[idx] for idx in ops])) != 1

                related_ranks_num = len(ops)

                for op_idx in ops:
                    transmit_time_arr[op_idx] = transmit_time
                    related_ranks_arr[op_idx] = related_ranks_num
                    across_nodes_arr[op_idx] = across_nodes_flag

        self.dfs.insert(self.dfs.shape[1], 'transmit_time', transmit_time_arr)
        self.dfs.insert(self.dfs.shape[1], 'related_ranks', related_ranks_arr)

    def vote_result_to_df(self):
        res = pd.DataFrame(columns=["rankId", "perpetrator_times", "count_times"])
        for rank, vote_list in self.vote_result.items():
            perpetrator_times, count_times = vote_list[0], vote_list[1]
            res.loc[len(res.index)] = [rank, perpetrator_times, count_times]
        return res

    def parser_host_rank_map(self):
        map_dic = defaultdict(list)
        for i in range(self.num_ranks):
            if self.rankid_arr[i] not in map_dic[self.host_id_arr[i]]:
                map_dic[self.host_id_arr[i]].append(self.rankid_arr[i])

        res = pd.DataFrame(columns=["host", "rankId"])
        for key, value in map_dic.items():
            value.sort()
            res.loc[len(res.index)] = [key, str(value)]
        return res

    def parser_group_rank_map(self):
        map_dic = defaultdict(list)
        for i in range(self.num_ranks):
            if self.rankid_arr[i] not in map_dic[self.group_name_arr[i]]:
                map_dic[self.group_name_arr[i]].append(self.rankid_arr[i])
        res = pd.DataFrame(columns=["groupName", "rankId"])
        for key, value in map_dic.items():
            value.sort()
            res.loc[len(res.index)] = [key, str(value)]
        return res

    def sum_time_per_rank(self):
        transmit_time_sum = self.dfs.groupby('rankId')['transmit_time'].sum().reset_index()
        return transmit_time_sum

    def slow_link(self):
        """
        处理数据，分组并检测异常值。
        """
        mapper_res = self.dfs
        # 按 opType, dataSize, related_ranks 分组
        grouped = mapper_res.groupby(['opType', 'dataSize', 'related_ranks'])

        for _, group in grouped:
            # 提取分组数据中的 transmit_time 列
            transmit_time_data = group['transmit_time'].values

            # 检测异常值
            outliers = detect_outliers_z_score(transmit_time_data)

            if outliers:
                # 如果存在异常值，将整个分组数据存入 Slow_Link_Ops
                self.slow_link_ops.append(group)

        if self.slow_link_ops:
            self.slow_link_ops = pd.concat(self.slow_link_ops, ignore_index=True)
            # 重置索引并去掉多余的索引列
            data = pd.DataFrame(self.slow_link_ops)

            # 按 'opType', 'dataSize', 'related_ranks' 分组
            grouped = data.groupby(['opType', 'dataSize', 'related_ranks'])

            # 计算统计信息
            group_data = describe_duration(grouped['transmit_time'])

            # 找到每个组中 transmit_time 最小值和最大值对应的 rankId
            min_rank = grouped['transmit_time'].idxmin().map(data['rankId'])
            max_rank = grouped['transmit_time'].idxmax().map(data['rankId'])

            # 将最大值和最小值对应的 rankId 添加到 group_data
            group_data['max_rank'] = max_rank.values
            group_data['min_rank'] = min_rank.values

            # 构造 filteringName
            group_data['opType_relatedRanks_dataSize'] = group_data.index.map(lambda x: f"{x[0]}{x[2]}_{x[1]}")
            # 将 filteringName 移动到第一列
            cols = ['opType_relatedRanks_dataSize'] + [col for col in group_data.columns if
                                                       col != 'opType_relatedRanks_dataSize']
            group_data = group_data[cols]

            # 重置索引
            group_data = group_data.reset_index(drop=True)
            # 计算最大值和最小值与均值的绝对值
            group_data['abs_max_mean'] = abs(group_data['maxNs'] - group_data['meanNs'])
            group_data['abs_min_mean'] = abs(group_data['minNs'] - group_data['meanNs'])

            # 计算最大值和最小值与均值的绝对值中的较大值
            group_data['max_abs_mean'] = group_data[['abs_max_mean', 'abs_min_mean']].max(axis=1)

            # 计算偏移比值
            group_data['offset_ratio'] = group_data['max_abs_mean'] / group_data['stdNs']

            # 按偏移比值降序排序
            group_data = group_data.sort_values(by='offset_ratio', ascending=False)

            # 删除辅助列 'abs_max_mean', 'abs_min_mean', 'max_abs_mean'
            group_data = group_data.drop(columns=['abs_max_mean', 'abs_min_mean', 'max_abs_mean'])

            # 调整列的顺序，将 offsetRatio 移到 MinRank 和 MaxRank 之前
            columns = [col for col in group_data.columns if col not in ['max_rank', 'min_rank', 'offset_ratio']]
            columns.insert(len(columns), 'offset_ratio')  # 将 offsetRatio 插入到倒数第三的位置
            columns.extend(['max_rank', 'min_rank'])  # 添加 MaxRank 和 MinRank 到列的最后

            # 重新排列列的顺序
            group_data = group_data[columns]

            # 在处理 group_data 的最后部分并保存
            self.slow_link_sum = group_data
            return self.slow_link_sum, self.slow_link_ops
        else:
            # 如果没有异常值，返回空的 DataFrame
            return pd.DataFrame(), pd.DataFrame()

    def save_db(self, path="./"):
        default_name = "vote_result.db"
        logging.info(f"saving file to {path}")
        save_file = os.path.join(path, default_name)
        conn = sqlite3.connect(save_file)

        def df_to_db(sql_obj, df_obj, tabel_name):
            df_obj.to_sql(tabel_name, sql_obj, if_exists="replace", index=False)

        vote_res = self.vote_result_to_df()
        df_to_db(conn, vote_res, "vote_result")
        group_rank_map = self.parser_group_rank_map()
        df_to_db(conn, group_rank_map, "group_rank_map")
        host_rank_map = self.parser_host_rank_map()
        df_to_db(conn, host_rank_map, "host_rank_map")
        transmit_time_sum = self.sum_time_per_rank()
        df_to_db(conn, transmit_time_sum, "transmit_time_sum")
        slow_link_sum, slow_link_ops = self.slow_link()
        df_to_db(conn, slow_link_sum, "slow_link_sum")
        df_to_db(conn, slow_link_ops, "slow_link_ops")

        conn.close()


def format_columns(df: pd.DataFrame):
    formatted_df = df.rename(
        {
            "25%": "q1Ns",
            "50%": "medianNs",
            "75%": "q3Ns",
            0.25: "q1Ns",
            0.5: "medianNs",
            0.75: "q3Ns",
            "Q1": "q1Ns",
            "Q3": "q3Ns",
            "min": "minNs",
            "max": "maxNs",
            "median": "medianNs",
            "sum": "sumNs",
            "std": "stdNs",
            "mean": "meanNs",
            "count": "count"
        },
        axis="columns"
    )

    stats_cols = ["count", "meanNs", "stdNs", "minNs", "q1Ns", "medianNs", "q3Ns", "maxNs", "sumNs"]
    other_cols = [col for col in formatted_df.columns if col not in stats_cols]
    return formatted_df[stats_cols + other_cols]


def describe_duration(series_groupby):
    agg_df = series_groupby.agg(["min", "max", "count", "std", "mean", "sum"])
    quantile_df = series_groupby.quantile([0.25, 0.5, 0.75])

    quantile_df = quantile_df.unstack()
    quantile_df.columns = ["25%", "50%", "75%"]

    stats_df = pd.merge(agg_df, quantile_df, left_index=True, right_index=True)
    formated_df = format_columns(stats_df)
    formated_df.index.name = stats_df.index.name
    return formated_df


def detect_outliers_z_score(data, threshold=3):
    """
    使用 Z-Score 方法判断是否存在异常值。
    Z-Score 是一种统计方法，用于衡量数据点与均值的标准差距离。
    如果某个数据点的 Z-Score 超过阈值（默认为3），则认为它是异常值。

    返回值：
    - True：存在异常值
    - False：不存在异常值
    """
    # 计算数据的均值
    mean = np.mean(data)  # 均值表示数据的中心位置

    # 计算数据的标准差
    std = np.std(data)  # 标准差表示数据的离散程度

    # 如果标准差为0，直接返回 False（不存在异常值）
    if std == 0:
        return False

    # 计算 Z-Score 的上阈值和下阈值
    z_scores_upper_threshold = threshold * std + mean
    z_scores_lower_threshold = -threshold * std + mean

    # 判断是否存在 Z-Score 超过阈值的数据点
    has_outliers = any(x > z_scores_upper_threshold or x < z_scores_lower_threshold for x in data)

    # 返回是否存在异常值的布尔值
    return has_outliers


def judge_vote(time_list):
    n = len(time_list)
    if n in [1, 2, 3]:
        return False, None, None
    sorted_list = copy.deepcopy(time_list)
    sorted_list.sort()
    if n <= 30:
        if n <= 7:
            flag = (sorted_list[1] - sorted_list[0]) / (sorted_list[-1] - sorted_list[0])
        elif n <= 10:
            flag = (sorted_list[1] - sorted_list[0]) / (sorted_list[-2] - sorted_list[0])
        elif n <= 13:
            flag = (sorted_list[2] - sorted_list[0]) / (sorted_list[-2] - sorted_list[0])
        else:
            flag = (sorted_list[2] - sorted_list[0]) / (sorted_list[-3] - sorted_list[0])
        return flag, time_list.index(sorted_list[0]), None
    elif n > 30:
        flag_max = (sorted_list[-1] - sorted_list[-3]) / (sorted_list[-1] - sorted_list[2])
        flag_min = (sorted_list[2] - sorted_list[0]) / (sorted_list[-3] - sorted_list[0])
        return flag_min > flag_max, time_list.index(sorted_list[0]), None
    return None, None, None


def load_db(path):
    db_loader = DBLoader(path)
    return db_loader.all_json_objects


def connect_and_process_sql(database_path: Path, query_sql: str) -> object:
    """
    链接到指定的sql数据库路径 执行查询sql 返回查询结果
    """
    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query_sql, conn)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="freq analyze")
    parser.add_argument('-d', '--path', type=str, required=True, help="profiling data path")
    parser.add_argument('-o', '--output_path', type=str, required=True, help="output path")
    args = parser.parse_args()

    dfs = load_db(args.path)
    processor = DBProcessor(dfs)
    processor.run()
    processor.save_db(args.output_path)
