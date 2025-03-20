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
import os
import time
import json
from pathlib import Path

import numpy as np

from modelevalstate.inference.data_format_v1 import BatchField, RequestField, ConfigPath
from modelevalstate.inference.state_eval_v1 import predict_v1_with_cache
from modelevalstate.inference.file_reader import FileHanlder, StaticFile
from modelevalstate.inference.constant import IS_SLEEP_FLAG, BatchStage, DataProcessType
from modelevalstate.inference.dataset import CustomLabelEncoder, preset_category_data, SimpleDataProcessor, \
    DataProcessor


class ServiceField:
    batch_field = None
    request_field = None
    next_tokens = None
    config_path = None
    fh = None
    data_processor = None
    req_id_and_max_decode_length = None


class Simulate:
    @staticmethod
    def init(plugin_object):
        self = plugin_object
        if isinstance(self.input_manager.cache_config.eos_token_id, int):
            self.eos_token_id = self.input_manager.cache_config.eos_token_id
        else:
            self.eos_token_id = self.input_manager.cache_config.eos_token_id[0]

    @staticmethod
    def generate_random_token(plugin_object, shape, max_value=32000):
        # max_value 是vacab size，就是词表的范围
        array = np.random.choice(np.arange(0, max_value + 1), size=np.prod(shape), replace=False)
        array = np.reshape(array, shape)
        array = np.where(array == plugin_object.eos_token_id, np.random.randint(0, max_value + 1), array)
        return array

    @staticmethod
    def generate_token(plugin_object, input_metadata, cached_idx):
        self = plugin_object
        next_tokens = Simulate.generate_random_token(plugin_object, (input_metadata.batch_size,))

        output_len_count = self.input_manager.cache.output_len_count[cached_idx]
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
        new_next_tokens = next_tokens.copy()
        for i, _ in enumerate(next_tokens):
            _cur_out_len = output_len_count[i]
            if input_metadata.batch_request_ids[i] not in ServiceField.req_id_and_max_decode_length:
                continue
            _max_out_len = ServiceField.req_id_and_max_decode_length[input_metadata.batch_request_ids[i]]
            if _cur_out_len > _max_out_len:
                new_next_tokens[i] = self.eos_token_id
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


def init(req_and_decode_file: Path, data_processor_type: DataProcessType = DataProcessType.DATA_PROCESS):
    with open(req_and_decode_file, 'r') as f:
        ServiceField.req_id_and_max_decode_length = {int(k): int(v) for k, v in json.load(f).items()}
    static_file = StaticFile(base_path=ServiceField.config_path.static_file_dir)
    ServiceField.fh = FileHanlder(static_file)
    ServiceField.fh.load_static_data()
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=ServiceField.config_path.ohe_path)
    custom_encoder.fit(load=True)
    if data_processor_type == DataProcessType.DATA_PROCESS:
        ServiceField.data_processor = DataProcessor(custom_encoder)
    else:
        ServiceField.data_processor = SimpleDataProcessor(custom_encoder)


ServiceField.config_path = ConfigPath(Path(r"./predict_model/bak/base/xgb_model.ubj"),
                                      Path(r"./predict_model/ohe"),
                                      Path(r"./predict_model/llama3-8b"))

init(Path(r"./predict_model/req_id_and_decode_num.json"))