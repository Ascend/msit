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

from msservice_advisor.profiling_analyze.utils import TARGETS, LOG_LEVELS, SUGGESTION_TYPES
from msservice_advisor.profiling_analyze.utils import str_ignore_case, logger, set_log_level

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

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


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
    logger.debug(f"read_csv_or_json {file_path = }")
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
    logger.debug("\nreq_to_data_map:")
    req_to_data_map = read_csv_or_json(get_latest_matching_file(instance_path, REQ_TO_DATA_MAP_PATTERN))
    logger.debug(f"req_to_data_map: {get_next_dict_item(req_to_data_map) if req_to_data_map else None}")

    logger.debug("\nresult_perf:")
    result_perf = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_PERF_PATTERN))
    result_perf = {kk: dict(zip(PERF_METRICS, vv)) for kk, vv in result_perf.items()} if result_perf else {}
    logger.debug(f"result_perf: {get_next_dict_item(result_perf) if result_perf else None}")

    logger.debug("\nresult_common:")
    result_common = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_COMMON_PATTERN))
    logger.debug(f"result_common: {result_common if result_common else None}")

    logger.debug("\nresults_per_request:")
    results_per_request = read_csv_or_json(get_latest_matching_file(instance_path, RESULTS_PER_REQUEST_PATTERN))
    logger.debug(f"results_per_request: {get_next_dict_item(results_per_request) if results_per_request else None}")

    return dict(
        req_to_data_map=req_to_data_map if req_to_data_map else {},
        result_perf=result_perf if result_perf else {},
        result_common=result_common if result_common else {},
        results_per_request=results_per_request if results_per_request else {},
    )


""" parse_mindie_server_config """


def parse_mindie_server_config(service_config_path):
    logger.debug("\nmindie_service_config:")
    if service_config_path.endswith(".json"):  # config.json directly
        mindie_service_path = os.path.dirname(os.path.dirname(service_config_path))
    else:  # mindie service path
        mindie_service_path = service_config_path
        service_config_path = os.path.join(service_config_path, "conf", "config.json")
    logger.info(f"mindie_service_path: {mindie_service_path}, service_config_path: {service_config_path}")
    mindie_service_config = read_csv_or_json(service_config_path)
    
    logger.debug(
        f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
    )

    if mindie_service_config:
        mindie_server_log_path = mindie_service_config.get("LogConfig", {}).get("logPath", "logs/mindie-server.log")
        mindie_server_log_path = os.path.join(mindie_service_path, mindie_server_log_path)
    else:
        mindie_server_log_path = None
    return mindie_service_config, mindie_server_log_path


""" analyze """


def analyze(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    import msservice_advisor.profiling_analyze
    from msservice_advisor.profiling_analyze.register import REGISTRY, ANSWERS

    logger.info("")
    logger.info("<think>")
    for name, analyzer in REGISTRY.items():
        logger.info(name)
        analyzer(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics)
    logger.info("</think>")

    logger.info("")
    logger.info("<answer>")
    for suggesion_type in SUGGESTION_TYPES:
        for name, items in ANSWERS.get(suggesion_type, dict()).items():
            for action, reason in items:
                logger.info(f"[{suggesion_type}] {name}")
                logger.info(f"[action] {action}")
                logger.info(f"[reason] {reason}")
                logger.info("")
    logger.info("</answer>")


""" arg_parse """


def arg_parse(argv):
    import argparse

    mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--instance_path", type=str, default="instance", help="benchamrk instance output directory", required=True
    )
    parser.add_argument(
        "-s", "--service_config_path", type=str, default=mindie_service_path, help="service config json path"
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
    parser.add_argument("--log_level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")

    parser.add_argument(
        "--show",
        action='store_true',
        help="control to show the plot",
    )
    return parser.parse_known_args(argv)[0]


def main():
    import sys
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"Failed to import matplotlib.pyplot, cannot create a fit curve plot: {e}")
        plt = None
        
    args = arg_parse(sys.argv)
    set_log_level(args.log_level)
    benchmark_instance = parse_benchmark_instance(args.instance_path)
    mindie_service_config, mindie_server_log_path = parse_mindie_server_config(args.service_config_path)
    analyze(mindie_service_config, benchmark_instance, mindie_server_log_path, args.target, args.target_metrics)
    
    if not args.show:
        return
    if plt is None:
        logger.error("Failed to import matplotlib.pyplot, can not show the plot.")
        return
    plt.show()
    plt.close()
        

if __name__ == "__main__":
    main()
