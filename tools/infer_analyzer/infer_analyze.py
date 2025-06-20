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
    analyzer = PerformanceAnalyzer(args.data_path, args.output_path).run()

    # python infer_analyze.py --data-path="" --output-path=""
