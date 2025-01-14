# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import os
import copy
from collections import defaultdict
import logging
from pathlib import Path
import sqlite3
import pandas as pd
from tqdm import tqdm
from concurrent import futures
from functools import partial
import time
import numpy as np


logging.basicConfig(level=logging.INFO)


DB_PARTTERN = "ascend_pytorch_profiler_*.db"

QUERY_SQL = """
SELECT
    rdm.rankId,
    si.value AS groupName,
    co.endNs - co.startNs AS communication_time,
    sii.value AS opName,
    host.hostUid as host_id,
    CASE
        WHEN et.name = 'INT8' THEN 1 * co.count
        WHEN et.name = 'FP16' THEN 2 * co.count
        WHEN et.name = 'FP32' THEN 4 * co.count
        WHEN et.name = 'BF16' THEN 2 * co.count
    END AS dataSize
FROM
    COMMUNICATION_OP co
CROSS
    JOIN RANK_DEVICE_MAP rdm
    JOIN STRING_IDS si ON co.groupName = si.id
    JOIN STRING_IDS sii ON co.opName = sii.id
    JOIN ENUM_HCCL_DATA_TYPE et ON co.dataType = et.id
    JOIN HOST_INFO host
"""


DIXTON_95_TABLE = {
    3:0.941,
    4:0.765,
    5:0.642,
    6:0.562,
    7:0.507,
    8:0.554,
    9:0.512,
    10:0.477,
    11:0.575,
    12:0.546,
    13:0.521,
    14:0.546,
    15:0.524,
    16:0.505,
    17:0.489,
    18:0.475,
    19:0.462,
    20:0.450,
    21:0.440,
    22:0.431,
    23:0.422,
    24:0.413,
    25:0.406,
    26:0.399,
    27:0.393,
    28:0.387,
    29:0.381,
    30:0.376
}


class MulitProcessor:
    def __init__(self):
        self._executor = futures.ProcessPoolExecutor(max_workers=os.cpu_count())

    def __enter__(self):
        if self._executor is None:
            raise RuntimeError("executor is None")
        return self
    
    def close(self):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def launch(self, func, *args, **kwargs):
        return self._executor.submit(func, *args, **kwargs).result()
    
    def map(self, func, *iterables, **kwargs):
        partial_func = partial(func, **kwargs)
        return list(self._executor.map(partial_func, *iterables))
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

        with MulitProcessor() as executor:
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
        self.n = len(self.rankid_arr)
        self.group_name_arr = dfs["groupName"].values
        self.communication_time_arr = dfs["communication_time"].values
        self.op_name_arr = dfs["opName"].values
        self.host_id_arr = dfs["host_id"].values
        self.vote_result = defaultdict(lambda: [0, 0])
    
    def run(self):
        process_group = defaultdict(lambda: defaultdict(list))
        transmit_time_arr = np.zeros((self.n), dtype=np.int64)
        related_ranks_arr = np.zeros((self.n), dtype=np.int32)
        across_nodes_arr = np.zeros((self.n), dtype=np.bool_)

        for idx in range(self.n):
            if "send" in self.op_name_arr[idx] or "receive" in self.op_name_arr[idx]:
                continue
            process_group[self.group_name_arr[idx]][self.op_name_arr[idx]].append(idx)
        
        for group_name, ops_same_group in tqdm(process_group.items(), desc="Processing DB data..."):
            for op_name, ops in ops_same_group.items():
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

    def vote_result_to_df(self):
        res = pd.DataFrame(columns=["rankId", "perpetrator_times", "count_times"])
        for rank, vote_list in self.vote_result.items():
            perpetrator_times, count_times = vote_list[0], vote_list[1]
            res.loc[len(res.index)] = [rank, perpetrator_times, count_times]
        return res
    
    def parser_host_rank_map(self):
        map_dic = defaultdict(list)
        for i in range(self.n):
            if self.rankid_arr[i] not in map_dic[self.host_id_arr[i]]:
                map_dic[self.host_id_arr[i]].append(self.rankid_arr[i])

        res = pd.DataFrame(columns=["host", "rankId"])
        for key, value in map_dic.items():
            value.sort()
            res.loc[len(res.index)] = [key, str(value)]
        return res
    
    def parser_group_rank_map(self):
        map_dic = defaultdict(list)
        for i in range(self.n):
            if self.rankid_arr[i] not in map_dic[self.group_name_arr[i]]:
                map_dic[self.group_name_arr[i]].append(self.rankid_arr[i])
        res = pd.DataFrame(columns=["groupName", "rankId"])
        for key, value in map_dic.items():
            value.sort()
            res.loc[len(res.index)] = [key, str(value)]
        return res
    
    def save_db(self, path="./"):
        default_name = "vote_result.db"
        logging.info(f"saving file to {path}")
        save_file = os.path.join(path, default_name)
        conn = sqlite3.connect(save_file)

        def df_to_db(sql_obj, df_obj, tabel_name):
            df_obj.to_sql(tabel_name, sql_obj, if_exists="replace")
        
        vote_res = self.vote_result_to_df()
        df_to_db(conn, vote_res, "vote_result")
        group_rank_map = self.parser_group_rank_map()
        df_to_db(conn, group_rank_map, "group_rank_map")
        host_rank_map = self.parser_host_rank_map()
        df_to_db(conn, host_rank_map, "host_rank_map")

        conn.close()


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
