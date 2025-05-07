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
import os
import stat
import time
from pathlib import Path
from loguru import logger
import numpy as np
import torch

from modelevalstate.config.config import settings
from modelevalstate.inference.constant import IS_SLEEP_FLAG, BatchStage
from modelevalstate.inference.data_format_v1 import BatchField, RequestField, ConfigPath
from modelevalstate.inference.dataset import CustomLabelEncoder, preset_category_data, DataProcessor
from modelevalstate.inference.file_reader import FileHanlder, StaticFile
from modelevalstate.inference.state_eval_v1 import predict_v1_with_cache


class ServiceField:
    batch_field = None
    request_field = None
    next_tokens = None
    config_path = None
    fh = None
    data_processor = None
    req_id_and_max_decode_length = None


ServiceField.config_path = ConfigPath(settings.latency_model.model_path, settings.latency_model.ohe_path,
                                      settings.latency_model.static_file_dir)


class Simulate:
    first = True

    @staticmethod
    def init(plugin_object):
        if Simulate.first:
            if isinstance(plugin_object.input_manager.cache_config.eos_token_id, int):
                plugin_object.eos_token_id = plugin_object.input_manager.cache_config.eos_token_id
            else:
                plugin_object.eos_token_id = plugin_object.input_manager.cache_config.eos_token_id[0]
            if settings.latency_model.req_and_decode_file.exists():
                with open(settings.latency_model.req_and_decode_file, 'r') as f:
                    ServiceField.req_id_and_max_decode_length = {int(k): int(v) for k, v in json.load(f).items()}
            else:
                ServiceField.req_id_and_max_decode_length = {}
            if not Path(ServiceField.config_path.static_file_dir).exists():
                Path(ServiceField.config_path.static_file_dir).mkdir(parents=True)
            static_file = StaticFile(base_path=settings.latency_model.static_file_dir)
            ServiceField.fh = FileHanlder(static_file)
            ServiceField.fh.load_static_data()
            custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=settings.latency_model.ohe_path)
            custom_encoder.fit(load=True)
            ServiceField.data_processor = DataProcessor(custom_encoder)
            Simulate.first = False

    @staticmethod
    def generate_random_token(plugin_object, shape, max_value=32000):
        # max_value 是vacab size，就是词表的范围
        array = np.random.choice(np.arange(0, max_value + 1), size=np.prod(shape), replace=False)
        array = np.reshape(array, shape)
        array = np.where(array == plugin_object.eos_token_id, np.random.randint(0, max_value + 1), array)
        return array

    @staticmethod
    def generate_logits(batch_size, vocab_size: int = 129280, device="npu:0", dtype="float16"):
        dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float: "float",
            torch.int8: "int8"
        }
        _cur_dtype = torch.float16
        for k, v in dtype_map.items():
            if v == dtype:
                _cur_dtype = k
                break
        tensor = torch.randn((batch_size, vocab_size), dtype=_cur_dtype, device=device)
        return tensor

    @staticmethod
    def generate_features(plugin_object, input_metadata, cached_idx):
        output_len_count = plugin_object.input_manager.cache.output_len_count[cached_idx]
        if input_metadata.is_prefill:
            batch_stage = BatchStage.PREFILL
        else:
            batch_stage = BatchStage.DECODE
        all_input_length = input_metadata.batch_seq_len
        all_need_blocks = np.count_nonzero(input_metadata.block_tables > -1, axis=-1)
        request_field = []
        _total_req_input_len = []
        for _cache_id_index, _cache_id in enumerate(cached_idx):
            _req_input_len = all_input_length[_cache_id_index] - output_len_count[_cache_id_index]
            _total_req_input_len.append(_req_input_len)
            request_field.append(RequestField(_req_input_len,
                                              all_need_blocks[_cache_id_index],
                                              output_len_count[_cache_id_index]))
        batch_field = BatchField(batch_stage, input_metadata.batch_size,
                                 np.count_nonzero(input_metadata.block_tables > -1, axis=-1).sum(),
                                 sum(_total_req_input_len), max(_total_req_input_len))

        request_field = tuple(request_field)
        ServiceField.batch_field = batch_field
        ServiceField.request_field = request_field
        return batch_field, request_field

    @staticmethod
    def generate_token(plugin_object, input_metadata, cached_idx):
        next_tokens = Simulate.generate_random_token(plugin_object, (input_metadata.batch_size,),
                                                     plugin_object.model_wrapper.config.vocab_size)

        batch_field, request_field = Simulate.generate_features(plugin_object, input_metadata, cached_idx)
        output_len_count = plugin_object.input_manager.cache.output_len_count[cached_idx]
        ServiceField.batch_field = batch_field
        ServiceField.request_field = request_field
        new_next_tokens = next_tokens.copy()
        for i, _ in enumerate(next_tokens):
            _cur_out_len = output_len_count[i]
            if input_metadata.batch_request_ids[i] not in ServiceField.req_id_and_max_decode_length:
                continue
            _max_out_len = ServiceField.req_id_and_max_decode_length[input_metadata.batch_request_ids[i]]
            if _cur_out_len > _max_out_len:
                new_next_tokens[i] = plugin_object.eos_token_id
        ServiceField.next_tokens = new_next_tokens

    @staticmethod
    def predict(time_sleep: bool = True):
        """

        return: time ms
        """
        time_sleep = os.getenv(IS_SLEEP_FLAG, str(time_sleep)).lower().strip() == "true"
        predict_res = predict_v1_with_cache(ServiceField.batch_field, ServiceField.request_field,
                                            ServiceField.config_path, ServiceField.fh, ServiceField.data_processor)

        for _pre_v in predict_res:
            if _pre_v == -1:
                continue
            if time_sleep:
                time.sleep(_pre_v / 10 ** 6)
                return 0
            else:
                return _pre_v

    @staticmethod
    def predict_and_save():
        res = Simulate.predict(False)
        file_path = Path(settings.benchmark.custom_collect_output_path).joinpath(f"simulate_{os.getpid()}.csv")
        logger.debug(f"file path {file_path}")
        if file_path.exists():
            with open(file_path, "a+") as f:
                f.write(str(res))
                f.write("\n")
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(file_path, flags, modes), 'w') as fout:
                fout.write(str(res))
                fout.write("\n")
