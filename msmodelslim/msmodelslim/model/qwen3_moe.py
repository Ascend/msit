# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Any, Optional, Generator, Tuple

from torch import nn

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.utils.logging import logger_setter
from .common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from .factory import ModelFactory
from .interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV0, ModelSlimPipelineInterfaceV1, \
    IterSmoothInterface, FlexSmoothQuantInterface
from .transformers import TransformersModel


@ModelFactory.register("Qwen3-30B")
@ModelFactory.register("Qwen3-235B")
@logger_setter()
class Qwen3MoeModelAdapter(TransformersModel,
                           ModelInfoInterface,
                           ModelSlimPipelineInterfaceV0,
                           ModelSlimPipelineInterfaceV1,
                           IterSmoothInterface,
                           FlexSmoothQuantInterface,
                           ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3_moe'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self,
                                dataset: Any,
                                batch_size: int,
                                device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset,
                                              batch_size=batch_size,
                                              device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module, transformer_blocks: Optional[List[Tuple[str, nn.Module]]] = None,
                             ) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model, transformer_blocks)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear融合的映射配置：输入层归一化到QKV投影
            norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                         f"model.layers.{layer_idx}.self_attn.q_proj",
                         f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
            )

            # OV融合的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # 为当前layer添加2个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping_config
                ),
            ])
        return adapter_config
