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

import random
import numpy as np

from msservice_advisor.profiling_analyze.register import register_analyze, cached, answer
from msservice_advisor.profiling_analyze.utils import TARGETS, SUGGESTION_TYPES, logger
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warning(f"Failed to import matplotlib.pyplot, cannot create a fit curve plot: {e}")
    plt = None
    

def summary_batch_info(batch_info):
    summary = {}
    for batchsize, latency_list in batch_info.items():
        summary[batchsize] = {}
        latency_list.sort()

        summary[batchsize]["BSZ"] = batchsize
        summary[batchsize]["MIN"] = latency_list[0]
        summary[batchsize]["P10"] = latency_list[int(len(latency_list) * 0.1)]
        summary[batchsize]["P30"] = latency_list[int(len(latency_list) * 0.3)]
        summary[batchsize]["P50"] = latency_list[int(len(latency_list) * 0.5)]
        summary[batchsize]["P70"] = latency_list[int(len(latency_list) * 0.7)]
        summary[batchsize]["P90"] = latency_list[int(len(latency_list) * 0.9)]
        summary[batchsize]["MAX"] = latency_list[-1]
        summary[batchsize]["FIT_DATA"] = latency_list[int(len(latency_list) * 0.3): int(len(latency_list) * 0.7)]
    return summary


def print_list(array):
    for item in array:
        logger.info(item)


@cached()
def read_batch_and_latency(pre_request):
    # 1. 获取所有的batchsize 对应的 latency
    # 2. 计算P50
    # 3. 打印出来
    prefill_batch_info = {}  # batchsize => list of latency
    decode_batch_info = {}  # batchsize => list of latency
    for request in pre_request.values():
        prefill_batch_size = request.get("prefill_bsz")
        decode_batch_size_list = request.get("decode_bsz")
        latency_list = request.get("latency")
        for index, latency in enumerate(latency_list):
            if index == 0:
                prefill_batch_info.setdefault(prefill_batch_size, [])
                prefill_batch_info[prefill_batch_size].append(latency)
            elif len(decode_batch_size_list) < index:
                logger.debug("Check the result_perf_*.csv data, decode_bsz not matching with latency")
                break
            else:
                decode_batch_size = decode_batch_size_list[index - 1]
                decode_batch_info.setdefault(decode_batch_size, [])
                decode_batch_info[decode_batch_size].append(latency)

    prefill_summary = summary_batch_info(prefill_batch_info)  # batchsize => P50
    decode_summary = summary_batch_info(decode_batch_info)  # batchsize => P50

    return prefill_summary, decode_summary


def find_best_by_curve_fit(summary_fit_data, process_name):
    from scipy.optimize import curve_fit, minimize

    max_batch_size = summary_fit_data[-1]["BSZ"]
    logger.info(f"{process_name} 上次运行组的最大的 batch size 为 {max_batch_size}")

    if len(summary_fit_data) > 2:

        def func_curv(x, a, b, c):
            return a * x**b * np.exp(-c / x)

    else:

        def func_curv(x, a, b):
            return a * x + b

    points = []
    targets = []

    for x in summary_fit_data:
        bsz = x["BSZ"]
        for latency in x["FIT_DATA"]:
            points.append(bsz)
            targets.append(bsz * 1000 / latency)

    points.append(max_batch_size * 10)
    targets.append(0.00001)

    popt, pcov = curve_fit(func_curv, points, targets, maxfev=10000)
    logger.info(f"{process_name} 函数拟合后参数：{popt}")

    # 或者使用数值优化（通用方法，适用于任何模型）
    def negative_func(x):
        return -func_curv(x, *popt)  # 最小化负函数即最大化原函数

    best_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 2)])
    aggressive_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 5)])
    logger.info(f"{process_name} 搜索范围 2 倍当前最大batchsize. 结果是: {best_predicted.x[0]} {best_predicted}")
    logger.info(
        f"{process_name} 搜索范围 5 倍当前最大batchsize. 结果是:  {aggressive_predicted.x[0]} {aggressive_predicted}"
    )

    result = {
        "best_batch_size": int(best_predicted.x[0]),
        "max_batch_size": max_batch_size,
        "points": points,
        "targets": targets,
        "popt": popt,
        "process_name": process_name,
        "func_curv": func_curv
    }

    return result


def get_predict_image(results):
    import datetime
    # 获取当前时间戳
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H%M%S")
    
    if plt is None:
        return

    len_result = len(results)
    if len_result <= 3:
        fig, axes = plt.subplots(1, len_result, figsize=(10 * len_result, 6))
    else:
        num_rows = len_result // 3 + (len_result % 3 > 0)
        fig, axes = plt.subplots(num_rows, 3, figsize=(10 * 3, 6 * num_rows))
        axes = axes.flatten()
    for result, ax in zip(results, axes):
        max_batch_size = result.get('max_batch_size', 0)
        points = result.get('points', [])
        targets = result.get('targets', [])
        popt = result.get('popt', None)
        process_name = result.get('process_name', '')
        func_curv = result.get('func_curv', None)
        if func_curv is None or max_batch_size == 0:
            logger.info("func_curv is None")
            return
        # 开始画图
        x_values = np.linspace(0, max_batch_size * 5, 300)

        # 绘制模型预测均值和置信区间
        ax.plot(x_values, func_curv(x_values, *popt), label=f"Model Prediction", color="blue")
        ax.scatter(points, targets, c="green", s=100, edgecolor="black", label="Existing Data Points")

        ax.set_title(f"Curve Fit Optimization({process_name})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speed")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()

    png_name = f"func_curv_{timestamp}.png"
    logger.info(f"拟合画图路径：{png_name}")
    plt.savefig(png_name)


@register_analyze()
def find_best_batch_size(config, benchmark, output_log, limit, target_metrics):
    if "results_per_request" not in benchmark:
        return

    results = []
    prefill_summary, decode_summary = read_batch_and_latency(benchmark.get("results_per_request", {}))

    def divide_fit_and_print(summary):
        summary.sort(key=lambda x: x["BSZ"])
        to_fit = [dict(BSZ=x["BSZ"], FIT_DATA=x.pop("FIT_DATA")) for x in summary]
        return to_fit, summary

    prefill_to_fit, prefill_to_print = divide_fit_and_print(list(prefill_summary.values()))
    decode_to_fit, decode_to_print = divide_fit_and_print(list(decode_summary.values()))

    logger.info("==decode==")
    print_list(decode_to_print)
    logger.info("==prefill==")
    print_list(prefill_to_print)

    if len(decode_to_fit) <= 1:
        answer(
            suggesion_type=SUGGESTION_TYPES.config,
            suggesion_item="maxBatchSize",
            action="set bigger",
            reason="目前batch样本太小，建议调大点试试",
        )

    if len(prefill_to_fit) <= 1:
        answer(
            suggesion_type=SUGGESTION_TYPES.config,
            suggesion_item="maxPrefillBatchSize",
            action="set bigger",
            reason="目前batch样本太小，建议调大点试试",
        )
    if len(prefill_to_fit) > 1:
        best_prefill_result = find_best_by_curve_fit(prefill_to_fit, "prefill")
        results.append(best_prefill_result)
        answer(
            suggesion_type=SUGGESTION_TYPES.config,
            suggesion_item="maxPrefillBatchSize",
            action=f"set to {best_prefill_result['best_batch_size']}",
            reason="经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize",
        )
    if len(decode_to_fit) > 1:
        best_decode_result = find_best_by_curve_fit(decode_to_fit, "decode")
        results.append(best_decode_result)
        answer(
            suggesion_type=SUGGESTION_TYPES.config,
            suggesion_item="maxBatchSize",
            action=f"set to {max(best_decode_result['best_batch_size'], best_prefill_result['best_batch_size'])}",
            reason="经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize",
        )
    get_predict_image(results)
