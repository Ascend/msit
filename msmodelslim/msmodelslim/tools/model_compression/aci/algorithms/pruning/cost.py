from abc import ABC, abstractmethod
import numpy as np

from torch import nn


class Coster(ABC):
    @abstractmethod
    def get_cost(self, layer, input_costs, in_channels=None, out_channels=None) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class ParamCoster(Coster):
    def get_cost(self, layer, input_costs, in_channels=None, out_channels=None) -> float:
        cost = 0
        if isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels if out_channels is None else out_channels
            in_channels = layer.in_channels if in_channels is None else in_channels
            cost = (out_channels // layer.groups) * in_channels * np.prod(layer.kernel_size)
        elif isinstance(layer, nn.Linear):
            out_channels = layer.out_features if out_channels is None else out_channels
            in_channels = layer.in_features if in_channels is None else in_channels

            cost = out_channels * in_channels
        else:
            raise NotImplementedError(f"Layer type {layer} is not suppported")

        return cost / 1e6  # Returns in unit of millions

    def get_name(self) -> str:
        return "param"


class FlopCoster(ParamCoster):
    def get_cost(self, layer, input_costs, in_channels=None, out_channels=None) -> float:
        original_flops, original_params = input_costs
        new_num_params = super().get_cost(layer, input_costs, in_channels, out_channels)

        cost = 0
        if isinstance(layer, nn.Conv2d):
            original_num_patches = original_flops // original_params
            cost = original_num_patches * new_num_params
        elif isinstance(layer, nn.Linear):
            original_num_patches = original_flops // layer.out_features // layer.in_features
            cost = original_num_patches * in_channels * out_channels
            cost = cost / 1e6

        else:
            raise NotImplementedError(f"Layer type {layer} is not suppported")

        return cost / 1e3  # Returns in unit of Gigaflops

    def get_name(self) -> str:
        return "flop"