# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from dataclasses import dataclass, field
from typing import List

from msmodelslim.utils.exception import SchemaValidateError


@dataclass
class Metadata:
    # ID of the quantization config, e.g., 'Qwen3-32B W8A8'
    config_id: str
    # score of the quantization config, used to sort the quantization configs
    score: float
    # label of the quantization config, used to filter the quantization configs.
    # e.g., # {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}
    label: dict
    # verified model types, e.g., ['LLaMa3.1-70B', 'Qwen2.5-72B']
    verified_model_types: List[str] = field(default_factory=list)


@dataclass
class BaseQuantConfig:
    apiversion: str  # API version
    metadata: Metadata  # metadata of the quantization config
    spec: object  # spec of the quantization config

    @staticmethod
    def from_dict(d: object) -> "BaseQuantConfig":
        if not isinstance(d, dict):
            raise SchemaValidateError(f'quant config must be a dict',
                                      action='Please make sure the quant config is a dictionary')
        metadata = d['metadata'] if 'metadata' in d else {'config_id': "Unknown", 'score': '100', 'label': {}}
        return BaseQuantConfig(
            apiversion=d.get('apiversion', 'Unknown'),
            metadata=Metadata(**metadata),
            spec=d['spec']
        )
