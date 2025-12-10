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

from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput, TokenizedEmbeddingReqInput
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode
)

from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook


def prof_get_batch_rids(batch):
    prof_rid_list = []
    for req in batch.reqs:
        prof_rid_list.append(str(req.rid))
    return prof_rid_list


def get_batch_type(batch):
    if batch.forward_mode.is_decode():
        return "decode"
    elif batch.forward_mode.is_extend():
        return "prefill"
    return "unknow"


def prof_kvcache_info(scheduler, name="allocate"):
    if scheduler.is_hybrid:
        (
            _,
            _,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = scheduler._get_swa_token_info()
        Profiler(Level.INFO).domain("KVCache")\
            .metric("deviceBlock", full_available_size)\
            .metric("fullEvictableSize", full_evictable_size)\
            .metric("swaAvailableSize", swa_available_size)\
            .metric("swaEvictableSize", swa_evictable_size)\
            .event(name)
    else:
        _, _, available_size, evictable_size = scheduler._get_token_info()
        Profiler(Level.INFO).domain("KVCache").metric("deviceBlock", available_size)\
            .metric("fullEvictableSize", evictable_size)\
            .event(name)


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler.recv_requests"),
    min_version="0.5.4"
)
def recv_requests(original_func, this, *args, **kwargs):
    recv_reqs = original_func(this, *args, **kwargs)

    for req in recv_reqs:
        if isinstance(
            req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            Profiler(Level.INFO).domain("Schedule").res(str(req.rid)).event("recvReq")

    return recv_reqs


@vllm_hook(
    hook_points=[
        ("sglang.srt.managers.scheduler", "Scheduler.handle_generate_request"),
        ("sglang.srt.managers.scheduler", "Scheduler.handle_embedding_request"),
    ],
    min_version="0.5.4"
)
def request_dispatcher(original_func, this, recv_req, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("Schedule").span_start("processReq").\
        res(str(recv_req.rid))

    output = original_func(this, recv_req, *args, **kwargs)

    prof.span_end()

    return output


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler.get_next_batch_to_run"),
    min_version="0.5.4"
)
def get_next_batch_to_run(original_func, this, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("Schedule").span_start("batchFrameworkProcessing")

    batch = original_func(this, *args, **kwargs)

    if batch:
        prof_kvcache_info(this, "allocate")
        prof.attr("batch_type", get_batch_type(batch)).res(prof_get_batch_rids(batch))
        prof.span_end()

    return batch


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler.run_batch"),
    min_version="0.5.4"
)
def run_batch(original_func, this, batch, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("modelExec").\
        res(prof_get_batch_rids(batch)).attr("batch_type", get_batch_type(batch))

    result = original_func(this, batch, *args, **kwargs)

    prof.span_end()

    return result


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler.process_batch_result"),
    min_version="0.5.4"
)
def process_batch_result(original_func, this, batch, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("postprocess").\
        res(prof_get_batch_rids(batch)).attr("batch_type", get_batch_type(batch))

    result = original_func(this, batch, *args, **kwargs)

    prof.span_end()

    return result


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler._add_request_to_queue"),
    min_version="0.5.4"
)
def add_request_to_queue(original_func, this, req, is_retracted: bool = False, *args, **kwargs):
    if this.disaggregation_mode == DisaggregationMode.NULL:
        if not this._abort_on_queued_limit(req):
            Profiler(Level.INFO).domain("Schedule").res(str(req.rid)).\
                metric_scope("QueueName", "WAITING").event("Enqueue")
            Profiler(Level.INFO).domain("Schedule").metric("QueueSize", len(this.waiting_queue)).\
                metric_scope("QueueName", "WAITING").event("Queue")
        elif this.disaggregation_mode == DisaggregationMode.PREFILL:
            Profiler(Level.INFO).domain("Schedule").res(str(req.rid)).\
                metric_scope("QueueName", "PrefillBootstrap").event("Enqueue")
            Profiler(Level.INFO).domain("Schedule").metric("QueueSize", len(this.disagg_prefill_bootstrap_queue)).\
                metric_scope("QueueName", "PrefillBootstrap").event("Queue")
        elif this.disaggregation_mode == DisaggregationMode.DECODE:
            this.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                Profiler(Level.INFO).domain("Schedule").res(str(req.rid)).\
                    metric_scope("QueueName", "DecodePrealloc").event("Enqueue")
                Profiler(Level.INFO).domain("Schedule").metric("QueueSize", len(this.disagg_decode_prealloc_queue)).\
                    metric_scope("QueueName", "DecodePrealloc").event("Queue")

    result = original_func(this, req, is_retracted, *args, **kwargs)

    return result


@vllm_hook(
    ("sglang.srt.managers.scheduler", "Scheduler.get_new_batch_prefill"),
    min_version="0.5.4"
)
def get_new_batch_prefill(original_func, this, *args, **kwargs):
    new_batch = original_func(this, *args, **kwargs)
    if new_batch is None:
        return new_batch

    Profiler(Level.INFO).domain("Schedule").metric("QueueSize", len(this.waiting_queue)).\
        metric_scope("QueueName", "WAITING").event("Queue")
    
    for req in new_batch.reqs:
        Profiler(Level.INFO).domain("Schedule").res(str(req.rid)).\
            metric_scope("QueueName", "WAITING").event("Dequeue")

    return new_batch


@vllm_hook(
    ("sglang.srt.managers.schedule_batch", "Req.init_next_round_input"),
    min_version="0.5.4"
)
def init_next_round_input(original_func, this, *args, **kwargs):
    ret = original_func(this, *args, **kwargs)

    if this.origin_input_ids != 0:
        Profiler(Level.INFO).domain("Schedule.KVCache").\
            metric("hitRate", len(this.prefix_indices) / len(this.origin_input_ids)).\
            res(str(this.rid)).event("HitCache")

    return ret


@vllm_hook(
    ("sglang.srt.managers.scheduler_output_processor_mixin",
     "SchedulerOutputProcessorMixin.process_batch_result_prefill"),
    min_version="0.5.4"
)
def process_batch_result_prefill(original_func, this, batch, *args, **kwargs):
    if this.is_generation:
        for req in batch.reqs:
            if this.enable_overlap and req.is_retracted and len(req.output_ids) > 0:
                continue

            not_finished = this.is_mixed_chunk and this.enable_overlap and \
                (req.finished() or req.is_retracted)
            if not_finished:
                continue

            if req.is_retracted:
                continue

            if req.is_chunked <= 0 and req.finished():
                Profiler(Level.INFO).domain("Request").res(str(req.rid)).\
                    metric("recvTokenSize", len(req.origin_input_ids)).\
                    metric("replyTokenSize", len(req.output_ids)).\
                    event("PrefillEnd")

    ret = original_func(this, batch, *args, **kwargs)

    prof_kvcache_info(this, "free")

    return ret


@vllm_hook(
    ("sglang.srt.managers.scheduler_output_processor_mixin",
     "SchedulerOutputProcessorMixin.process_batch_result_decode"),
    min_version="0.5.4"
)
def process_batch_result_decode(original_func, this, batch, *args, **kwargs):
    for req in batch.reqs:
        if req.is_retracted:
            continue

        if req.finished():
            Profiler(Level.INFO).domain("Request").res(str(req.rid)).\
                metric("recvTokenSize", len(req.origin_input_ids)).\
                metric("replyTokenSize", len(req.output_ids)).\
                event("DecodeEnd")

    ret = original_func(this, batch, *args, **kwargs)

    prof_kvcache_info(this, "free")

    return ret