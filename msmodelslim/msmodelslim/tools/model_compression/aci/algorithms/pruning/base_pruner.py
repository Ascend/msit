from abc import ABC, abstractmethod
import typing
import torch
import torch.nn as nn

from torch_pruning import ops
from .masker import MaskingType, add_mask, set_mask, has_mask, remove_mask


class BasePruner(ABC):

    def __init__(self,
                 importance: typing.Callable,
                 masking_type: str = "soft",
                 ):
        """
        Base class for structural pruning.

        Args:
            importance: criteria to estimate channels importance
            masking_type: type of mask that is applied to model
            bn_rescaling: whether add BatchNorm scaling option or not
        """
        self.importance = importance
        self.masking_type = MaskingType(masking_type)

    def create_mask(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, tuple(self.root_module_types)):
                add_mask(module, self.masking_type, mask_name='weight')

    def remove_mask(self, model: nn.Module) -> None:
        for module in model.modules():
            if has_mask(module, name='weight'):
                remove_mask(module, mask_name='weight')

    def set_channel_pruning_mask(self, layer: nn.Module, channel_mask: torch.Tensor) -> None:
        # For output masks
        if isinstance(layer, ops.TORCH_CONV):
            set_mask(layer, "weight", channel_mask.view(-1, 1, 1, 1))
        elif isinstance(layer, ops.TORCH_LINEAR):
            set_mask(layer, "weight", channel_mask.view(-1, 1))
        else:
            raise NotImplementedError(f"Cannot apply channel mask to layer of type {type(layer)}")

    def estimate_importance(self, group, ch_groups=1, input_masks=None) -> torch.Tensor:
        return self.importance(group, ch_groups=ch_groups, input_masks=input_masks)

    @abstractmethod
    def prune(self, interactive: bool = False):
        pass