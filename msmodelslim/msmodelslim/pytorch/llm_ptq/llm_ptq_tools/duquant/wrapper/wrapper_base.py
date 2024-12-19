# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import functools
import gc
from abc import abstractmethod

import torch

from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.duquant_alg import DuQuantConfig, DuQuant


@functools.lru_cache(maxsize=1)
def is_npu_available():
    npu_available = False

    try:
        import torch_npu
    except ImportError:
        npu_available = False
    else:
        npu_available = torch.npu.is_available()

    return npu_available


@functools.lru_cache(maxsize=1)
def is_cuda_available():
    cuda_available = False

    try:
        import torch_npu
    except ImportError:
        cuda_available = False
    else:
        cuda_available = torch.cuda.is_available()

    return cuda_available


class DuQuantWrapperBase(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, config: DuQuantConfig):
        super(DuQuantWrapperBase, self).__init__()
        self.module = module
        self.name = name
        self.delta_vector = None
        self.delta_vector_two = None
        self.rotations = None
        self.permutations = None
        self.permutations_size = config.total_rotate_step - 1
        self.config = config
        self.duquant = DuQuant(config)
        self.find_params = False

    @abstractmethod
    def get_weight_for_duquant(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_weight_for_duquant(self, w) -> torch.Tensor:
        raise NotImplementedError

    def print_with_name(self, *args, **kwargs):
        logger.info(f"{self.name}", *args, **kwargs)

    def get_delta_inverse_matrix(self):
        return torch.diag(self.delta_vector.reciprocal())

    def get_delta_inverse_matrix_two(self):
        return torch.diag(self.delta_vector_two.reciprocal())

    def apply_rotations_and_permutations(self, x) -> torch.Tensor:

        input_shape = x.shape
        x = x.view(-1, x.shape[-1])

        token_size = x.shape[0]
        feature_size = x.shape[-1]
        block_size = self.config.split_size

        for i in range(0, self.permutations_size):
            x = x.reshape(-1, feature_size // block_size, block_size).transpose(0, 1)
            x = (x @ self.rotations[i]).transpose(0, 1).reshape(token_size, feature_size)

            zigzag_index, sorted_index = self.permutations[i][0], self.permutations[i][1]
            x = x[:, sorted_index[zigzag_index]]

        x = x.reshape(-1, feature_size // block_size, block_size).transpose(0, 1)
        x = (x @ self.rotations[-1]).transpose(0, 1).reshape(token_size, feature_size)

        return x

    def rotate_activation(self, x: torch.Tensor):
        input_shape = x.shape
        input_dtype = x.dtype
        x = x.view(-1, x.shape[-1])

        if self.delta_vector is not None:
            x = x.mul(self.delta_vector.to(device=x.device)).to(dtype=input_dtype)

        if self.rotations is not None:
            x = self.apply_rotations_and_permutations(x)

        if self.delta_vector_two is not None:
            x = x.mul(self.delta_vector_two.to(device=x.device)).to(dtype=input_dtype)

        return x.view(input_shape).to(dtype=input_dtype)

    def find_delta_vector(self, x: torch.Tensor, w: torch.Tensor):
        return self.duquant.construct_delta_vector(x, w, delta_min=1e-5, delta_max=1e5)

    def find_rotations_and_permutations(self, x: torch.Tensor):
        return self.duquant.construct_rotations_and_permutations(x)

    def online_rotate_activation(self, x):

        if not self.config.enable:
            return x

        x = self.rotate_activation(x)

        return x

    @torch.no_grad()
    def find_duquant_params(self, x):

        logger.info(f"find duquant paramters for {self.name}")

        x_dtype = x.dtype
        x_ = x.clone().detach().double().view(-1, x.shape[-1])

        w_ = self.get_weight_for_duquant().detach()
        w_dtype = w_.dtype
        w_ = w_.double()

        if self.delta_vector is None and not self.config.disable_delta1:
            self.delta_vector = self.find_delta_vector(x_, w_)
            x_ = x_.mul(self.delta_vector)
            w_ = w_.t().div(self.delta_vector).t()

        if self.rotations is None:
            self.rotations, self.permutations = self.find_rotations_and_permutations(x_)
            x_ = self.apply_rotations_and_permutations(x_)
            w_ = self.apply_rotations_and_permutations(w_.t()).t()

        if self.delta_vector_two is None and not self.config.disable_delta2:
            self.delta_vector_two = self.find_delta_vector(x_, w_)
            x_ = x_.mul(self.delta_vector_two)
            w_ = w_.t().div(self.delta_vector_two).t()

        x_ = x_.to(dtype=x_dtype)
        w_ = w_.to(dtype=w_dtype)

        self.rotations = self.rotations.to(dtype=x_dtype)
        self.set_weight_for_duquant(w_)

        gc.collect()

        if is_npu_available():
            torch.npu.empty_cache()
        elif is_cuda_available():
            torch.cuda.empty_cache()

        self.find_params = True

    def tensor_forward(self, x):
        if self.config.enable and not self.find_params:
            self.find_duquant_params(x)

        x = self.online_rotate_activation(x)

        return self.module(x)

    def attention_forward(self, *args, **kwargs):
        x = kwargs['hidden_states']

        if self.config.enable and not self.find_params:
            self.find_duquant_params(x)

        x = self.online_rotate_activation(x)

        kwargs['hidden_states'] = x

        return self.module(*args, **kwargs)


class DuQuantLinearWrapper(DuQuantWrapperBase):

    def __init__(self, module: torch.nn.Linear, name, config: DuQuantConfig):
        super(DuQuantLinearWrapper, self).__init__(module, name, config)

    def get_weight_for_duquant(self):
        return self.module.weight.data.t()

    def set_weight_for_duquant(self, w):
        self.module.weight.data = w.t()

    def forward(self, x):
        return self.tensor_forward(x)
