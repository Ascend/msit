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

import numpy as np
from msservice_advisor.profiling_analyze.register import register_analyze, cached, answer
from msservice_advisor.profiling_analyze.utils import TARGETS


def get_dict_value_by_pos(dict_value, target_pos):
    cur = dict_value
    for kk in target_pos.split(":"):
        if not cur:
            cur = None
            break
        if isinstance(cur, list) and str.isdigit(kk):
            cur = cur[int(kk)]
        elif kk in cur:
            cur = cur[kk]
        else:
            cur = None
            break
    return cur


@register_analyze()
def num_mem_size_checker(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    npu_mem_size_pos = "BackendConfig:ModelDeployConfig:ModelConfig:0:npuMemSize"

    npu_mem_size = get_dict_value_by_pos(mindie_service_config, npu_mem_size_pos)
    if npu_mem_size is not None and npu_mem_size != -1:
        print(f"获取目前 numMemSize 的值为 {npu_mem_size}, 并不是 -1")
        answer(config="npuMemSize", action="set to -1", reason="设置为-1，将由服务化自动根据剩余的显存数量，配置block数量，会尽量用满显存空间")


@register_analyze()
def check_input_tokens(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    max_prefill_tokens = get_dict_value_by_pos(mindie_service_config, "BackendConfig:ScheduleConfig:maxPrefillTokens")
    print(f"maxPrefillTokens: {max_prefill_tokens}")

    max_input_tokens = benchmark_instance.get("result_perf", {}).get("InputTokens", {}).get("max", "0").split(" ")[0]
    max_input_tokens_float = float(max_input_tokens) if max_input_tokens.replace(".", "", 1).isdigit() else 0
    print(f"Max InputTokens: {max_input_tokens_float}")

    if max_prefill_tokens is not None and max_input_tokens_float < max_prefill_tokens:
        answer(config="maxPrefillTokens", action=f"set to {max_input_tokens}", reason="设置为数据集的最大输入长度")


@register_analyze()
def check_output_tokens(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    max_iter_times = get_dict_value_by_pos(mindie_service_config, "BackendConfig:ScheduleConfig:maxIterTimes")
    print(f"maxIterTimes: {max_iter_times}")

    max_output_tokens = (
        benchmark_instance.get("result_perf", {}).get("GeneratedTokens", {}).get("max", "0").split(" ")[0]
    )
    max_output_tokens_float = float(max_output_tokens) if max_output_tokens.replace(".", "", 1).isdigit() else 0
    print(f"Max GeneratedTokens: {max_output_tokens_float}")

    if max_iter_times is not None and max_output_tokens_float < max_iter_times:
        answer(config="max_iter_times", action=f"set to {max_output_tokens_float}", reason="设置为数据集的最大输入长度")


@register_analyze()
def check_prefill_latency(mindie_service_config, benchmark_instance, mindie_server_log_path, target, target_metrics):
    results_per_request = benchmark_instance.get("results_per_request").values()
    prefill_latencies = np.array([ii["latency"][0] for ii in results_per_request if len(ii.get("latency", [])) > 0])

    counts, buckets = np.histogram(prefill_latencies)
    bucket_keys = ["{:.2f}-{:.2f}".format(ii, jj) for ii, jj in zip(buckets[:-1], buckets[1:])]
    bucket_keys_max_len = len(max(bucket_keys, key=lambda ii: len(ii)))
    print("First token latency:")
    print(" " * (4 + bucket_keys_max_len - 6) + "Bucket: Count")
    print(" " * 4 + "-" * bucket_keys_max_len + ": ------")
    print(
        "\n".join(
            [" " * (4 + bucket_keys_max_len - len(ii)) + "{}: {}".format(ii, jj) for ii, jj in zip(bucket_keys, counts)]
        )
    )

    support_select_batch = get_dict_value_by_pos(
        mindie_service_config, "BackendConfig:ScheduleConfig:supportSelectBatch"
    )
    if target == TARGETS.FirstTokenTime and support_select_batch:
        answer(config="support_select_batch", action="set to False", reason="设置为数据集的最大输入长度")
