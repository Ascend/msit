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


import argparse
import logging

import logger

from gen_csv import Generator
from load_file import FileReader
from parse_profile import ProfileParser

logger = logger.create_logger(name="infer_analyze", level=logging.INFO)


class PerformanceAnalyzer:
    def __init__(self, file_path, output_path=None):
        self.file_path = file_path
        self.output_path = output_path

    def run(self):
        logger.info("Start analyze")
        data = FileReader(self.file_path).load_data()
        logger.info("File read successfully")
        processed_csv = ProfileParser(data[0], data[1], data[2]).run()
        logger.info("Analyze successfully")
        Generator(processed_csv[0], processed_csv[1], output_path=self.output_path).generate_excel()
        logger.info("Save successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool for splitting and analysis inference')
    parser.add_argument('--data-path', required=True, help='Prof data path')
    parser.add_argument('--output-path', help='Output data path')

    args = parser.parse_args()

    # 运行分析器
    PerformanceAnalyzer(args.data_path, args.output_path).run()

    # python infer_analyze.py --data-path="" --output-path=""
