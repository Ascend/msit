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

import json
import csv
import threading
from warnings import warn
from queue import Queue
from pathlib import Path
from collections import namedtuple
from typing import List, Optional
from dataclasses import dataclass, field

from modelevalstate.common import _PREFILL, _DECODE

STOP_SUB_PROCESS = "state::process::break"

batch_need_fields = (
    "batch_index", "reqInfo", "ts_batch_execute_begin", "ts_batch_execute_end", "batch_execute_delta",
    "ts_model_execute_begin",
    "ts_model_execute_end", "model_execute_delta", "req_ids", "execute_ids", "p_vector", "d_vector", "end_vector",
    "p_count", "d_count", "end_count")

BatchNeed = namedtuple("BatchNeed", batch_need_fields)

request_need_fields = "ibis_reqid", "http_reqid", "req_token_size", "res_token_size", "ts_RecvHttpReq", \
    "ts_ReturnHttpRes", "HttpReq_delta", "P_count", "P_queue_latency", "P_batch_execute_delta", "P_e2e_latency", \
        "P_model_execute_delta", "D_count", "D_queue_latency_mean", "D_queue_latency_min", "D_queue_latency_max", \
            "D_batch_execute_delta_mean", "D_batch_execute_delta_min", "D_batch_execute_delta_max", \
                "D_e2e_latency_mean", "D_e2e_latency_min", "D_e2e_latency_max", \
                    "D_model_execute_delta_mean", "D_model_execute_delta_min", "D_model_execute_delta_max"

RequestNeed = namedtuple("RequestNeed", request_need_fields)

batch2req_need_fields = "ibis_reqid", "execute_id", "stage", "ts_AddToQueue", "ts_RemoveFromQueue", \
    "ts_batch_execute_begin", "ts_batch_execute_end", "ts_StateChangeToStart", "ts_StateChangeToEnd", \
        "ts_model_execute_begin", "ts_model_execute_end", "reqInfo"

Batch2RequestNeed = namedtuple("Batch2RequestNeed", batch2req_need_fields)

train_batch_data_fields = (
    "stage", "batch_num", "model_time", "batch_execute_delta", "total_time", "end_count", "request_info")
TrainBatchData = namedtuple("TrainBatchData", train_batch_data_fields)

train_request_info_fields = ("id", "execute_id", "input_len", "output_len", "httpreq_delta", 
                             "p_queue_latency", "p_batch_execute_delta", "d_count", "d_queue_latency_mean", 
                             "d_queue_latency_min", "d_queue_latency_max", "d_batch_execute_delta_mean", 
                             "d_batch_execute_delta_min", "d_batch_execute_delta_max", 
                             "state_change_delta", "queue_wait_time")
TrainRequestInfo = namedtuple("TrainRequestInfo", train_request_info_fields, \
                              defaults=[0 for _ in range(len(train_request_info_fields))])



@dataclass
class NodeInfo:
    stage: str  # 当前模型状态的类型 Prefill/Decode
    batch_num: int  # 当前状态处理的请求个数
    model_time: float  # 这个model （prefill、decode）执行完成的时间
    batch_execute_delta: float  # 组batch的时间
    total_time: float  # 从开始处理到现在的时间
    end_count: int  = 0 # batch中执行完成后有多少个请求结束
    request_info: List = field(default_factory=list)  # 当前batch处理包含的请求信息


def load_line_data(line, version=2):
    cur_row = [row for row in csv.reader([line])][0]
    _req_info = []
    for _req_row in eval(cur_row[-1]):
        _req_info.append(TrainRequestInfo(*[float(i) for i in _req_row]))
    if version == 1:
        # 最初的版本
        _node_info = NodeInfo(stage=cur_row[0],
                          batch_num=int(cur_row[1]),
                          model_time=float(cur_row[2]),
                          batch_execute_delta=float(cur_row[3]),
                          total_time=float(cur_row[4]),
                          request_info=_req_info)
    else:
        _node_info = NodeInfo(stage=cur_row[0],
                          batch_num=int(cur_row[1]),
                          model_time=float(cur_row[2]),
                          batch_execute_delta=float(cur_row[3]),
                          total_time=float(cur_row[4]),
                          end_count=int(cur_row[5]),
                          request_info=_req_info)
    return _node_info


def convert_to_config_param_json(file_path: Path, new_file_path: Path, request_rate: int = 20):
    """
    加载config param 信息，转为元组
    :param file_path:
    :param new_file_path:
    :param request_rate:
    :return:
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    new_data = {
        "model": data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelName"],
        "world_size": data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["worldSize"],
        "request_rate": request_rate,
        "concurrency": data["ServerConfig"]["maxLinkNum"],
        "prefill_batchsize": data["BackendConfig"]["ScheduleConfig"]["maxPrefillBatchSize"],
        "decode_batchsize": data["BackendConfig"]["ScheduleConfig"]["maxBatchSize"],
        "select_batch": data["BackendConfig"]["ScheduleConfig"]["supportSelectBatch"],
        "prefillTimeMsPerReq": data["BackendConfig"]["ScheduleConfig"]["prefillTimeMsPerReq"],
        "decodeTimeMsPerReq": data["BackendConfig"]["ScheduleConfig"]["decodeTimeMsPerReq"]
    }
    with open(new_file_path, 'w') as f:
        json.dump(new_data, f)


def get_all_req_info(request_need_csv: Path, req_queue: Optional[Queue] = None):
    res = {}
    with open(request_need_csv, newline='') as request_files:
        request_reader = csv.reader(request_files)
        for _, row in enumerate(request_reader):
            cur_row_info = RequestNeed(*row)
            res[cur_row_info.ibis_reqid] = (
                cur_row_info.req_token_size, cur_row_info.res_token_size, cur_row_info.HttpReq_delta,
                cur_row_info.P_queue_latency, cur_row_info.P_batch_execute_delta,
                cur_row_info.D_count, cur_row_info.D_queue_latency_mean, cur_row_info.D_queue_latency_min,
                cur_row_info.D_queue_latency_max, cur_row_info.D_batch_execute_delta_mean,
                cur_row_info.D_batch_execute_delta_min,
                cur_row_info.D_queue_latency_max)
    if req_queue:
        req_queue.put(res)
    return res


def get_all_batch2req_info(bahtch2req_final: Path, batch2req_queue: Optional[Queue] = None):
    res = {}
    with open(bahtch2req_final, newline='') as batch2req_files:
        batch2req_reader = csv.reader(batch2req_files)
        for _, row in enumerate(batch2req_reader):
            cur_row_info = Batch2RequestNeed(*row)
            res[(cur_row_info.ibis_reqid, cur_row_info.execute_id)] = (int(cur_row_info.ts_RemoveFromQueue) - int(
                cur_row_info.ts_AddToQueue), int(cur_row_info.ts_StateChangeToEnd) - int(
                cur_row_info.ts_StateChangeToStart))
    if batch2req_queue:
        batch2req_queue.put(res)
    return res


def writer_to_file(file_path: Path, data_queue: Optional[Queue] = None):
    _mode = "w+"
    if Path(file_path).exists():
        _mode = "a+"
    with open(file_path, _mode, newline='') as all_info_file:
        all_info_writer = csv.writer(all_info_file)
        while True:
            data = data_queue.get()
            if data == STOP_SUB_PROCESS:
                break
            all_info_writer.writerow(data)


def convert_to_all_info_with_optim(request_need_csv: Path, batch_need_csv: Path, batch2req_final: Path,
                                   all_info_csv: Path):
    batch2req_queue = Queue()
    req_queue = Queue()
    t = threading.Thread(target=get_all_batch2req_info, args=(batch2req_final, batch2req_queue))
    t.start()
    t1 = threading.Thread(target=get_all_req_info, args=(request_need_csv, req_queue))
    t1.start()
    t.join()
    t1.join()
    all_batch2req = batch2req_queue.get()
    all_request_need = req_queue.get()
    all_info_queue = Queue()
    t2 = threading.Thread(target=writer_to_file, args=(all_info_csv, all_info_queue))
    t2.start()

    with open(batch_need_csv, newline='') as batch_files:
        batch_reader = csv.reader(batch_files)
        for _, row in enumerate(batch_reader):
            cur_row_info = BatchNeed(*row)
            _batch_num = int(cur_row_info.p_count) + int(cur_row_info.d_count)
            if eval(cur_row_info.d_vector):
                _batch_stage = _DECODE
            elif eval(cur_row_info.p_vector):
                _batch_stage = _PREFILL
            else:
                warn("Abnormal data")
                raise ValueError("Abnormal data")
            _model_time = cur_row_info.model_execute_delta
            _batch_execute_delta = cur_row_info.batch_execute_delta
            _total_time = cur_row_info.ts_model_execute_end
            _request_info = []
            _cur_execute_ids = eval(cur_row_info.execute_ids)
            for i, _req_id in enumerate(eval(cur_row_info.req_ids)):
                if _req_id not in all_request_need:
                    warn(f"Not found req_id. r{_req_id} in {request_need_csv}")
                    raise ValueError(f"Not found req_id. r{_req_id} in {request_need_csv}")
                _run_num = _cur_execute_ids[i]
                if (_req_id, _run_num) not in all_batch2req:
                    warn(f"Not found req id ,execute id. r{(_req_id, _run_num)} in {batch2req_final}")
                    raise ValueError(f"Not found req id ,execute id. r{(_req_id, _run_num)} in {batch2req_final}")
                _tmp_req_data = [_req_id, _run_num, *all_request_need[_req_id], *all_batch2req[(_req_id, _run_num)]]
                _request_info.append(_tmp_req_data)
            _new_data = [_batch_stage, _batch_num, _model_time, _batch_execute_delta, _total_time,
                         int(cur_row_info.end_count), _request_info]
            all_info_queue.put(_new_data)
    all_info_queue.put(STOP_SUB_PROCESS)
    t2.join()