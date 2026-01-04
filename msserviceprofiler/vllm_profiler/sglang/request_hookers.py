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

from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook


@vllm_hook(
    ("sglang.srt.managers.detokenizer_manager", "DetokenizerManager.handle_batch_token_id_out"),
    min_version="0.5.4"
)
def handle_batch_token_id_out(original_func, this, recv_obj, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("Request").\
            res(recv_obj.rids).span_start("detokenize")

    ret = original_func(this, recv_obj, *args, **kwargs)

    prof.span_end()

    return ret


@vllm_hook(
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager._tokenize_one_request"),
    min_version="0.5.4"
)
async def tokenize_one_request(original_func, this, obj, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("Request").res(str(obj.rid)).span_start("tokenize")
        
    ret = await original_func(this, obj, *args, **kwargs)

    prof.span_end()

    return ret


@vllm_hook(
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager._batch_tokenize_and_process"),
    min_version="0.5.4"
)
async def batch_tokenize_and_process(original_func, this, batch_size, obj, *args, **kwargs):
    prof_list = []
    rid_list = [obj[i].rid for i in range(batch_size)]
    for rid in rid_list:
        prof_list.append(
            Profiler(Level.INFO).domain("Request").span_start("tokenize").res(str(rid))
        )

    ret = await original_func(this, batch_size, obj, *args, **kwargs)

    for prof in prof_list:
        prof.span_end()

    return ret


@vllm_hook(
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager._send_one_request"),
    min_version="0.5.4"
)
def send_one_request(original_func, this, obj, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("Request").span_start("send_to_scheduler.dispatch").\
        res(str(obj.rid))

    state = original_func(this, obj, *args, **kwargs)

    prof.span_end()

    return state


@vllm_hook(
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager._send_batch_request"),
    min_version="0.5.4"
)
def send_batch_request(original_func, this, obj, tokenized_objs, *args, **kwargs):
    prof_rid_list = []
    for tokenized_obj in tokenized_objs:
        prof_rid_list.append(tokenized_obj.rid)
    prof = Profiler(Level.INFO).domain("Request").\
        res(prof_rid_list).span_start("send_to_scheduler.dispatch")

    ret = original_func(this, obj, tokenized_objs, *args, **kwargs)

    prof.span_end()

    return ret


@vllm_hook(
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager._wait_one_response"),
    min_version="0.5.4"
)
async def wait_one_response(original_func, this, obj, *args, **kwargs):
    is_stream = obj.stream

    async for response in original_func(this, obj, *args, **kwargs):
        Profiler(Level.INFO).domain("Request").attr("stream", is_stream)\
            .res(str(obj.rid)).event("httpRes")
        yield response


@vllm_hook(
    hook_points=[
        ("sglang.srt.managers.io_struct", "GenerateReqInput.normalize_batch_and_arguments"),
        ("sglang.srt.managers.io_struct", "EmbeddingReqInput.normalize_batch_and_arguments")
    ],
    min_version="0.5.4"
)
def normalize_batch_and_arguments(original_func, this, *args, **kwargs):
    ret = original_func(this, *args, **kwargs)

    if this.is_single:
        bootstrap_room = (
            this.bootstrap_room if hasattr(this, "bootstrap_room") else None
        )
        Profiler(Level.INFO).domain("Request").res(str(this.rid)).\
            attr("bootstrap_room", bootstrap_room).event("httpReq")
    else:
        for i in range(len(this.rid)):
            bootstrap_room = (
                this.bootstrap_room[i]
                if hasattr(this, "bootstrap_room") and this.bootstrap_room
                else None
            )
            Profiler(Level.INFO).domain("Request").res(str(this.rid)).\
                attr("bootstrap_room", bootstrap_room).event("httpReq")

    return ret


