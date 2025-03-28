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
from pathlib import Path

from ms_service_profiler.parse import parse, preprocess_prof_folders
from ms_service_profiler.exporters.factory import ExporterFactory
from ms_service_profiler.exporters.utils import check_input_path_valid, check_output_path_valid, create_sqlite_db
from ms_service_profiler.plugins import custom_plugins
from ms_service_profiler.utils.log import set_log_level

from ms_service_profiler_ext.exporters.exporter_summary import ExporterSummary


def add_summary_exporter(func):
    def wrapper(args):
        default_exporters = func(args)

        summary_exporter = ExporterSummary()
        summary_exporter.initialize(args)

        return [summary_exporter] + default_exporters
    return wrapper


def main():
    parser = argparse.ArgumentParser(description='MS Server Profiler Analyze')
    parser.add_argument(
        '--input-path',
        required=True,
        type=check_input_path_valid,
        help='Path to the folder containing profile data.')
    parser.add_argument(
        '--output-path',
        type=check_output_path_valid,
        default=os.path.join(os.getcwd(), 'output'),
        help='Output file path to save results.')
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'fatal', 'critical'],
        help='Log level to print.')

    args = parser.parse_args()

    # 初始化日志等级
    set_log_level(args.log_level)

    # msprof预处理
    preprocess_prof_folders(args.input_path)

    # 初始化Exporter
    wrapped_create_exporters = add_summary_exporter(ExporterFactory.create_exporters)
    exporters = wrapped_create_exporters(args)

    # 创建output目录
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    create_sqlite_db(args.output_path)

    # 解析数据并导出
    parse(args.input_path, custom_plugins, exporters)


if __name__ == '__main__':
    main()