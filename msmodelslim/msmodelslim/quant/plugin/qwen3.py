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
from typing import List

import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from msmodelslim.quant.session.plugin import QuantPlugin


class Qwen3Plugin(QuantPlugin):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.config.num_hidden_layers = 1
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def load_model(self) -> PreTrainedModel:
        return self.model

    def load_calib_data(self) -> List[torch.Tensor]:
        return [self.tokenizer(self.args.calib_path, return_tensors="pt")]
