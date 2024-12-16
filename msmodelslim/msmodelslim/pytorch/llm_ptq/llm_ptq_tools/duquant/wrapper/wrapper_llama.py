# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.duquant_alg import DuQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_base import DuQuantWrapperBase


class DuQuantLlamaMLPWrapper(DuQuantWrapperBase):

    def __init__(self, module: torch.nn.Module, name, config: DuQuantConfig):
        super(DuQuantLlamaMLPWrapper, self).__init__(module, name, config)

    def get_weight_for_duquant(self):
        concatenate_weight = torch.cat(
            [self.module.gate_proj.weight.data.t(), self.module.up_proj.weight.data.t()], dim=1)

        return concatenate_weight

    def set_weight_for_duquant(self, w):
        w_split = w.split(
                [self.module.gate_proj.out_features, self.module.up_proj.out_features],
                dim=1)
        self.module.gate_proj.weight.data = w_split[0].t()
        self.module.up_proj.weight.data = w_split[1].t()

    def forward(self, x):
        return self.tensor_forward(x)


class DuQuantLlamaAttentionWrapper(DuQuantWrapperBase):

    def __init__(self, module: torch.nn.Module, name, config: DuQuantConfig):
        super(DuQuantLlamaAttentionWrapper, self).__init__(module, name, config)

    def get_weight_for_duquant(self):
        concatenate_weight = torch.cat(
            [self.module.q_proj.weight.data.t(),
             self.module.k_proj.weight.data.t(),
             self.module.v_proj.weight.data.t()],
            dim=1)

        return concatenate_weight

    def set_weight_for_duquant(self, w):
        w_split = w.split(
                [self.module.q_proj.out_features, self.module.k_proj.out_features, self.module.v_proj.out_features],
                dim=1)
        self.module.q_proj.weight.data = w_split[0].t()
        self.module.k_proj.weight.data = w_split[1].t()
        self.module.v_proj.weight.data = w_split[2].t()

    def forward(self, *args, **kwargs):
        return self.attention_forward(*args, **kwargs)
