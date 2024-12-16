# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.duquant_alg import DuQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_base import DuQuantWrapperBase


class DuQuantBaichuanMLPWrapper(DuQuantWrapperBase):

    def __init__(self, module: torch.nn.Module, name, config: DuQuantConfig):
        super(DuQuantBaichuanMLPWrapper, self).__init__(module, name, config)

    def get_weight_for_duquant(self):
        concatenate_weight = torch.cat(
            [self.module.gate_proj.weight.data.t(), self.module.up_proj.weight.data.t()], dim=1)

        return concatenate_weight

    def set_weight_for_duquant(self, w):
        [self.module.gate_proj.weight.data, self.module.up_proj.weight.data] = \
            [w_.t() for w_ in w.split([self.module.gate_proj.out_features, self.module.up_proj.out_features], dim=1)]

    def forward(self, x):
        return self.tensor_forward(x)


class DuQuantBaichuanAttentionWrapper(DuQuantWrapperBase):

    def __init__(self, module: torch.nn.Module, name, config: DuQuantConfig):
        super(DuQuantBaichuanAttentionWrapper, self).__init__(module, name, config)

    def get_weight_for_duquant(self):
        return self.module.W_pack.weight.data.t()

    def set_weight_for_duquant(self, w):
        self.module.W_pack.weight.data = w.t()

    def forward(self, *args, **kwargs):
        return self.attention_forward(*args, **kwargs)
