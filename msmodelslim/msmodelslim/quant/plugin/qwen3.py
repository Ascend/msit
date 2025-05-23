#  -*- coding: utf-8 -*-
#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import argparse
import json
import os
from typing import List, Dict

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, W8A8DynamicProcessorConfig, W8A8DynamicQuantConfig
from msmodelslim.quant.session.plugin import QuantPlugin


class Qwen3Plugin(QuantPlugin):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.config.num_hidden_layers = args.layer_count if args.layer_count > 0 else self.config.num_hidden_layers
        self.is_moe = "moe" in self.config.model_type
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, config=self.config, torch_dtype="auto")
        self.model.model.embed_tokens.to(torch.get_default_device())
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.default_calib = self.load_jsonl(args.calib_file, args.calib_key)
    
    @staticmethod
    def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
        dataset = []
        with os.fdopen(os.open(dataset_path, os.O_RDONLY, 0o600),
                    'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line)
                text = data.get(key_name, line)
                dataset.append(text)
        return dataset

    def load_model(self) -> PreTrainedModel:
        return self.model

    def load_calib_data(self) -> List[torch.Tensor]:
        return [self.tokenizer(prompt, return_tensors="pt").data for prompt in self.default_calib]

    def load_default_quant_cfg(self) -> Dict[str, BaseModel]:
        if self.is_moe:
            return {
                "w8a8": W8A8ProcessorConfig(
                    cfg_map={
                        "model.layers.*.self_attn.*": W8A8QuantConfig()
                    },
                ),
                "w8a8_dynamic": W8A8DynamicProcessorConfig(
                    cfg_map={
                        "model.layers.*.mlp.*": W8A8DynamicQuantConfig()
                    },
                    disable_names=[
                        "model.layers.*.mlp.gate"
                    ]
                )
            }
        else:
            return {
                "w8a8": W8A8ProcessorConfig(
                    cfg_map={
                        "*": W8A8QuantConfig()
                    },
                    disable_names=["model.layers.*.mlp.down_proj"]
                )
            }
