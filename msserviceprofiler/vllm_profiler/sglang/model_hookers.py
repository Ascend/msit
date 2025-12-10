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
    ("sglang.srt.model_executor.forward_batch_info", "ForwardBatch.init_new"),
    min_version="0.5.4"
)
def init_new(original_func, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("preprocess")

    ret = original_func(*args, **kwargs)

    prof.span_end()

    return ret


@vllm_hook(
    ("sglang.srt.model_executor.model_runner", "ModelRunner.forward"),
    min_version="0.5.4"
)
def forward(original_func, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("forward")

    output = original_func(*args, **kwargs)

    prof.span_end()

    return output


@vllm_hook(
    ("sglang.srt.model_executor.model_runner", "ModelRunner.sample"),
    min_version="0.5.4"
)
def sample(original_func, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("sample")

    next_token_ids = original_func(*args, **kwargs)

    prof.span_end()

    return next_token_ids