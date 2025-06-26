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
import pandas as pd

from ms_service_profiler.exporters.utils import check_input_path_valid, check_output_path_valid


def add_exporters(args):
    from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_prefill import ExporterPrefill
    from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_decode import ExporterDecode

    if not hasattr(args, 'format'):
        args.format = 'csv'
    exporter_cls = []
    exporters = []
    if args.prefill_batch_size > 0 or args.prefill_rid != '-1':
        exporter_cls.append(ExporterPrefill)
    if args.decode_batch_size > 0 or args.decode_rid != '-1':
        exporter_cls.append(ExporterDecode)
    for cls in exporter_cls:
        exporter = cls()
        exporter.initialize(args)
        exporters.append(exporter)
    return exporters


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "split", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="MS Server Profiler Split"
    )
    parser.add_argument(
        '--input-path', required=True, type=check_input_path_valid, help='Path to the folder containing profile data.'
    )
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
        help='Log level to print')
    parser.add_argument('--prefill-batch-size', type=int, default=0, help='Batch size for Prefill data.')
    parser.add_argument('--decode-batch-size', type=int, default=0, help='Batch size for Decode data.')
    parser.add_argument(
        '--prefill-number', type=int, default=1, help='The number of Prefill batch to calc statistical data'
    )
    parser.add_argument(
        '--decode-number', type=int, default=1, help='The number of Decode batch to calc statistical data'
    )
    parser.add_argument('--prefill-rid', type=str, default='-1', help='The rid for Prefill batch to split')
    parser.add_argument('--decode-rid', type=str, default='-1', help='The rid for Decode batch to split')
    parser.set_defaults(func=main)


def main(args):
    from ms_service_profiler.parse import parse
    from ms_service_profiler.utils.log import set_log_level
    from ms_service_profiler.plugins import custom_plugins

    # 初始化日志等级
    set_log_level(args.log_level)

    # 初始化Exporter
    exporters = add_exporters(args)

    # 创建output目录
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # 解析数据并导出
    parse(args.input_path, custom_plugins, exporters, args=args)
