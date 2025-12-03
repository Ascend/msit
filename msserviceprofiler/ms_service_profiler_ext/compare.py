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
import argparse

import pandas as pd

from ms_service_profiler.data_source.db_data_source import DBDataSource
from ms_service_profiler.exporters.utils import save_dataframe_to_csv
from ms_service_profiler.utils.log import set_log_level, logger
from ms_service_profiler.exporters.utils import check_input_dir_valid, check_output_path_valid


def read_sql_from_given_path(given_path):
    profiler_files_input = os.listdir(given_path)

    profiler_inputs = []
    for file in profiler_files_input:
        if file.endswith('.db'):
            data_dict = DBDataSource.process(os.path.join(given_path, file))
            if not data_dict.get('tx_data_df').empty:
                df = data_dict.get('tx_data_df')
                profiler_inputs.append(df)

    if profiler_inputs:
        sql_df_result = pd.concat(profiler_inputs, axis=0, ignore_index=True)
        return sql_df_result
    else:
        logger.error(f'the data from {given_path} is empty.')
        return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS Server Profiler Compare Tool")

    parser.add_argument("input_path", type=check_input_dir_valid, help="Directory containing analyzed results")
    parser.add_argument("golden_path", type=check_input_dir_valid, help="Directory containing analyzed results")
    parser.add_argument(
        "--output-path",
        type=check_output_path_valid,
        default=os.path.join(os.getcwd(), 'compare_result'),
        help="Output Directory after comparing."
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'fatal', 'critical'],
        help='Log level to print.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_log_level(args.log_level)

    profiler_input_df = read_sql_from_given_path(args.input_path)

    profiler_input_df.reset_index(drop=True, inplace=True)

    profiler_input_df['name'] = profiler_input_df['message'].apply(lambda x: x.get('name'))
    profiler_input_result = profiler_input_df.groupby('name')['during_time'].agg([
        'mean',
        ('A-P50', lambda x: x.quantile(0.50)),
        ('A-P90', lambda x: x.quantile(0.90))
    ])
    profiler_input_result.rename(columns={'mean': 'A-AVG'}, inplace=True)

    profiler_golden_df = read_sql_from_given_path(args.golden_path)

    profiler_golden_df.reset_index(drop=True, inplace=True)

    profiler_golden_df['name'] = profiler_golden_df['message'].apply(lambda x: x.get('name'))
    profiler_golden_result = profiler_golden_df.groupby('name')['during_time'].agg([
        'mean',
        ('B-P50', lambda x: x.quantile(0.50)),
        ('B-P90', lambda x: x.quantile(0.90))
    ])
    profiler_golden_result.rename(columns={'mean': 'B-AVG'}, inplace=True)

    result_concat = pd.concat([profiler_input_result, profiler_golden_result], axis=1)

    result_concat['DIFF-AVG'] = result_concat['A-AVG'] - result_concat['B-AVG']
    result_concat['DIFF-P50'] = result_concat['A-P50'] - result_concat['B-P50']
    result_concat['DIFF-P90'] = result_concat['A-P90'] - result_concat['B-P90']
    result_concat['RDIFF-AVG'] = result_concat['DIFF-AVG'] / result_concat['A-AVG']
    result_concat['RDIFF-P50'] = result_concat['DIFF-P50'] / result_concat['A-P50']
    result_concat['RDIFF-P90'] = result_concat['DIFF-P90'] / result_concat['A-P90']
    result_concat = result_concat.reset_index()
    result_concat = result_concat.rename(columns={'index': 'name'})

    save_dataframe_to_csv(result_concat, args.output_path, 'span_comparation_result.csv')

    logger.info("Comparing finished successfully, the results stored under %r", args.output_path)


if __name__ == '__main__':
    main()