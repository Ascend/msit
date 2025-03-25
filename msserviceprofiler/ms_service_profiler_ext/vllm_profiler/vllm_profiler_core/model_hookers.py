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
from .vllm_hooker_base import VLLMHookerBase

is_model_first_run = True


class ModelRunnerExecuteHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.worker.model_runner import ModelRunner

        def execute_model_maker(ori_func):
            def execute_model(this, model_input, kv_caches, *args, **kwargs):
                global is_model_first_run
                if is_model_first_run:
                    is_model_first_run = False
                    return ori_func(this, model_input, kv_caches, *args, **kwargs)

                prof = Profiler(Level.INFO)
                prof.span_start("modelExec")
                ret = ori_func(this, model_input, kv_caches, *args, **kwargs)

                is_prefill = model_input.attn_metadata.prefill_metadata

                request_id_list = []

                for request_id, _ in model_input.request_ids_to_seq_ids.items():
                    request_id_list.append({"rid": request_id})

                prof.res(request_id_list)

                if is_prefill:
                    prof.attr("batch_type", "prefill")
                else:
                    prof.attr("batch_type", "decode")

                batch_size = model_input.input_tokens.shape[0]
                prof.attr("batch_size", batch_size)

                prof.span_end()
                return ret

            return execute_model

        self.do_hook([ModelRunner.execute_model], execute_model_maker)


is_forward_first_run = True


class ModelForwardHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.attention.backends.utils import CommonAttentionState

        def begin_forward_maker(ori_func):
            def begin_forward(this, model_input):
                global is_forward_first_run
                ret = ori_func(this, model_input)
                if is_forward_first_run:
                    is_forward_first_run = False
                    return ret

                request_id_list = []

                for request_id, _ in model_input.request_ids_to_seq_ids.items():
                    request_id_list.append(request_id)
                prof = Profiler(Level.INFO)
                prof.res(request_id_list).event("Forward")
                return ret

            return begin_forward

        self.do_hook([CommonAttentionState.begin_forward], begin_forward_maker)


model_hookers = [ModelRunnerExecuteHook, ModelForwardHook]
