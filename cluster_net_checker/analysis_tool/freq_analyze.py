# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import os
import time
from pathlib import Path
import sqlite3
from concurrent import futures
from functools import partial
from collections import defaultdict
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO)


QUERY_SQL = """
SELECT
    rdm.rankId,
    freq.timestampNs,
    freq.freq
FROM
    AICORE_FREQ freq
CROSS
    JOIN
    RANK_DEVICE_MAP rdm
"""

DB_PATTERN = "ascend_pytorch_profiler_*.db"

COMMON_FREQ = 1800

FREE_FREQ = 800


def connect_and_process_sql(database_path: Path, query_sql: str) -> object:
    """
    链接到指定的sql数据库路径 执行查询sql 返回查询结果
    """
    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query_sql, conn)
    return df


class MulitProcessor:
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

        with MulitProcessor() as executor:
            mapper_res = executor.map(
                self.mapper_func,
                self.path_list,
            )
        
        self.concat_db(mapper_res)
        t2 = time.time()
        logging.info(f"loading cost time {t2 - t1}s")


def load_db(path):
    db_loader = DBLoader(path)
    return db_loader.all_json_objects


class FreqProcessor:
    def __init__(self, dfs):
        self.rankid_arr = dfs["rankId"].values
        self.time_arr = dfs["timestampNs"].values
        self.freq_arr = dfs["freq"].values
        self.run()

    def parser_freq(self):
        n = len(self.freq_arr)
        dic = defaultdict(list)

        for idx in range(n):
            freq = self.freq_arr[idx]
            if freq not in dic[self.rankid_arr[idx]]:
                dic[self.rankid_arr[idx]].append(freq)
        
        return dic
    
    def run(self):
        dic = self.parser_freq()
        error_ranks = []
        free_ranks = []
        for rank, value in dic.items():
            if value == [COMMON_FREQ]:
                pass
            elif set(value) == set([COMMON_FREQ, FREE_FREQ]):
                free_ranks.append(rank)
            else:
                error_ranks.append(rank)

        logging.info("=" * 50)
        if len(free_ranks) > 0:
            logging.info(f"find ranks with free time, aic freq in {[COMMON_FREQ, FREE_FREQ]}: {free_ranks}")
        else:
            logging.info("no rank found with free time")

        logging.info("=" * 50)
        if len(error_ranks) > 0:    
            logging.info("find ranks with abnormal aic freq:")
            for rank in error_ranks:
                logging.info(f"rank: {rank}, abnormal_freq: {dic[rank]}")
        else:
            logging.info("no rank found with abnormal aic freq")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="freq analyze")
    parser.add_argument('-d', '--path', type=str, required=True, help="profiling data path")
    args = parser.parse_args()
    
    dfs = load_db(args.path)
    FreqProcessor(dfs)

