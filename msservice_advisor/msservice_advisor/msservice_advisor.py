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
import json
import csv
from collections import namedtuple
from glob import glob

from msservice_advisor.profiling_analyze.utils import TARGETS, str_ignore_case


# {"21559056a7ff44c88a891ecbb537c431": "0", ...}
REQ_TO_DATA_MAP_PATTERN = "req_to_data_map.json"

# FirstTokenTime,DecodeTime,LastDecodeTime,...
# 213.2031 ms,228.3775 ms,88.327 ms,...
# -> average, max, min, P75, P90, SLO_P90, P99, N
RESULT_PERF_PATTERN = "result_perf_*.csv"
PERF_METRICS = ["average", "max", "min", "P75", "P90", "SLO_P90", "P99", "N"]
PERF_METRICS_MAP = {str_ignore_case(ii): ii for ii in PERF_METRICS}

# ...,Concurrency,ModelName,lpct,Throughput,GenerateSpeed,...
# ...,50,DeepSeek-R1,0.9336 ms,2.789 req/s,...
RESULT_COMMON_PATTERN = "result_common_*.csv"

# {"7": {"input_len": 213, "output_len": 12, "prefill_bsz": 15, "decode_bsz": [20, ...],
#        "req_latency": 2348332643508911, "latency": [798.7475395202637, ...], "queue_latency": [445314, ...], ... }
RESULTS_PER_REQUEST_PATTERN = "results_per_request_*.json"

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"

TARGETS_MAP = {
    "ttft": TARGETS.FirstTokenTime,
    "firsttokentime": TARGETS.FirstTokenTime,
    "throughput": TARGETS.Throughput,
}


""" parse_benchmark_instance """


def get_latest_matching_file(instance_path, pattern):
    files = glob(os.path.join(instance_path, pattern))
    return max(files, key=os.path.getmtime) if files else None


def read_csv(file_path):
    result = {}
    with open(file_path, mode="r", newline="", encoding="utf-8") as ff:
        for row in csv.DictReader(ff):
            for kk, vv in row.items():
                result.setdefault(kk, []).append(vv)
    return result


def read_json(file_path):
    with open(file_path) as ff:
        result = json.load(ff)
    return result


def read_csv_or_json(file_path):
    print(f">>>> {file_path = }")
    if not file_path or not os.path.exists(file_path):
        return None
    if file_path.endswith(".json"):
        return read_json(file_path)
    if file_path.endswith(".csv"):
        return read_csv(file_path)
    return None


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


def parse_benchmark_instance(instance_path):
    print("\n>>>> req_to_data_map:")
    req_to_data_map = read_csv_or_json(get_latest_matching_file(instance_path, REQ_TO_DATA_MAP_PATTERN))
    print(f">>>> req_to_data_map: {get_next_dict_item(req_to_data_map) if req_to_data_map else None}")

    print("\n>>>> result_perf:")
    result_perf = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_PERF_PATTERN))
    result_perf = {kk: dict(zip(PERF_METRICS, vv)) for kk, vv in result_perf.items()}
    print(f">>>> result_perf: {get_next_dict_item(result_perf) if result_perf else None}")

    print("\n>>>> result_common:")
    result_common = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_COMMON_PATTERN))
    print(f">>>> result_common: {result_common if result_common else None}")

    print("\n>>>> results_per_request:")
    results_per_request = read_csv_or_json(get_latest_matching_file(instance_path, RESULTS_PER_REQUEST_PATTERN))
    print(f">>>> results_per_request: {get_next_dict_item(results_per_request) if results_per_request else None}")

    return dict(
        req_to_data_map=req_to_data_map,
        result_perf=result_perf,
        result_common=result_common,
        results_per_request=results_per_request,
    )


""" parse_mindie_server_config """


def parse_mindie_server_config():
    print("\n>>>> mindie_service_config:")
    mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
    print(f">>>> mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}")

    if mindie_service_config:
        mindie_server_log_path = mindie_service_config.get("LogConfig", {}).get("logPath", "logs/mindie-server.log")
        mindie_server_log_path = os.path.join(mindie_service_path, mindie_server_log_path)
    else:
        mindie_server_log_path = None
    return mindie_service_config, mindie_server_log_path


""" analyze """


def analyze(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    from msservice_advisor.profiling_analyze import base_analyze
    from msservice_advisor.profiling_analyze import batch_analyze
    from msservice_advisor.profiling_analyze.register import REGISTRY, print_answer

    print("\n<think>")
    for name, analyzer in REGISTRY.items():
        print(name)
        analyzer(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics)
    print("</think>")

    print_answer()


""" arg_parse """


def arg_parse(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--instance_path",
        type=str,
        default="benchamrk instance output directory",
        help="instance",
        required=True
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str_ignore_case,
        default="ttft",
        choices=list(TARGETS_MAP.keys()),
        help="profiling key target",
    )
    parser.add_argument(
        "-m",
        "--target_metrics",
        type=lambda xx: PERF_METRICS_MAP.get(str_ignore_case(xx), None),
        default="average",
        choices=PERF_METRICS,
        help="profiling key target metrics",
    )
    return parser.parse_known_args(argv)[0]


def main():
    import sys

    args = arg_parse(sys.argv)
    benchmark_instance = parse_benchmark_instance(args.instance_path)
    mindie_service_config, mindie_server_log_path = parse_mindie_server_config()
    analyze(mindie_service_config, benchmark_instance, mindie_server_log_path, args.target, args.target_metrics)


if __name__ == "__main__":
    main()
