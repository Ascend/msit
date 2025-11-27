# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from pathlib import Path

import torch
from regex import F
from torch import nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.security.model import SafeGenerator
from .base import BaseModelAdapter


class VLMBaseModelAdapter(BaseModelAdapter):
    """
    VLM base model adapter providing basic attributes and methods for VLM models.
    To use, subclass and implement the required methods for your specific model.
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.config = self._load_config(trust_remote_code=trust_remote_code)
        self.model_pedigree = self._get_model_pedigree(self.model_type)
        self.model_type = self._get_model_type(self.model_type)

    def _enable_kv_cache(self, model: nn.Module, enable: bool):
        model.model.config.use_cache = enable

    def _load_config(self, trust_remote_code=False) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(model_path=str(self.model_path),
                                                        trust_remote_code=trust_remote_code)

    def _get_model_type(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type
        return model_type

    def _get_model_pedigree(self, model_type: str) -> str:
        if model_type is None:
            return self.config.model_type

        model_type = re.match(r'^[a-zA-Z]+', model_type)
        if model_type is None:
            raise SchemaValidateError(f"Invalid model_name: {model_type}.",
                                      action='Please check the model type')
        return model_type.group().lower()
