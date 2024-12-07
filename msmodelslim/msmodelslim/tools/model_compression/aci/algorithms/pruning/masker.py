import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Tuple
from torch import autograd, nn


class MaskingType(Enum):
    Soft = "soft"
    Hard = "hard"
    Trainable = "trainable"


def mask_smoother(x, eps=1e-3):
    x1 = torch.pow(x, 2)
    x2 = torch.add(x1, eps)
    return torch.div(x1, x2)


class HardParameterMasker(autograd.Function):
    """Hard channel masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        mask, = ctx.saved_tensors
        return grad_output * mask, None


class SoftParameterMasker(autograd.Function):
    """STE (straight-through estimator) channel masker"""

    @staticmethod
    def forward(_, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


class TrainableParameterMasker(SoftParameterMasker):
    """STE (straight-through estimator) channel masker"""
    pass


def get_orig(module: nn.Module, name: str) -> nn.Parameter:
    if has_mask(module, name):
        return getattr(module, f"{name}_orig")
    else:
        return getattr(module, name)


def has_mask(module: nn.Module, name: str) -> bool:
    return hasattr(module, f"{name}_mask")


def get_mask(module: nn.Module, name: str) -> torch.Tensor:
    return getattr(module, f"{name}_mask")


def set_mask(module: nn.Module, name: str, new_mask: torch.Tensor) -> None:
    """
    Can pass any mask that is expandable to the actual mask size
    """
    if not has_mask(module, name) and not new_mask.all():
        "Cannot set a non-True mask to an unmasked layer"
        return

    masking_type: MaskingType = getattr(module, f"{name}_masking_type")

    mask = get_mask(module, name)
    if masking_type == MaskingType.Trainable:
        mask.data.copy_(torch.mul(mask, new_mask.expand(*mask.shape)))
    else:
        mask.data[:] = new_mask.expand(*mask.shape)


def _create_masking_pre_hook(name: str, masking_type: MaskingType) -> Callable[[nn.Module, Any], None]:
    if masking_type == MaskingType.Soft:
        masker = SoftParameterMasker
    elif masking_type == MaskingType.Hard:
        masker = HardParameterMasker
    elif masking_type == MaskingType.Trainable:
        masker = TrainableParameterMasker
    else:
        raise NotImplementedError(f"Forward pre hook for masking type '{masking_type}' not implemented")

    def _masking_pre_hook(module: nn.Module, _: Any) -> None:
        if not hasattr(module, f"{name}_orig"):
            raise Exception("Module %s param/buffer %s wasn't setup to be pruned" % module, name)

        mask = get_mask(module, name)
        orig = get_orig(module, name)

        setattr(module, name, masker.apply(orig, mask))

    return _masking_pre_hook


def _create_masking_cleanup_hook(name: str) -> Callable[[nn.Module, Any], None]:
    def _masking_cleanup_hook(module: nn.Module, input: Any, output: Any) -> None:
        if not hasattr(module, f"{name}_orig"):
            raise Exception("Module %s param/buffer %s wasn't setup to be pruned" % module, name)

        setattr(module, name, None)

    return _masking_cleanup_hook


def _add_attr_mask(
        module: nn.Module, name: str, masking_type: MaskingType, mask_zeros: bool
) -> None:
    orig_attr = getattr(module, name)

    if orig_attr is None:
        return
    if name not in module._parameters.keys():
        return

    # Move original parameter to new name
    module.register_parameter(f"{name}_orig", orig_attr)
    del module._parameters[name]

    # Store the masking type
    setattr(module, f"{name}_masking_type", masking_type)

    # Register the masking buffer
    mask = (orig_attr != 0.) if mask_zeros else torch.ones_like(orig_attr, dtype=torch.bool)

    if masking_type == MaskingType.Trainable:
        module.register_parameter(f"{name}_mask", nn.Parameter(mask.type(torch.float32), requires_grad=True))
    else:
        module.register_buffer(f"{name}_mask", mask)

    # Register forward pre hook that will recompute the plain attribute before each forward
    remove_prehook = module.register_forward_pre_hook(_create_masking_pre_hook(name, masking_type))
    setattr(module, f"{name}_remove_prehook", remove_prehook)

    # Register forward hook that will cleanup the plain attribute after each forward
    # Note: without this, the model cannot be deepcopied (due to non-leaf tensors)

    remove_posthook = module.register_forward_hook(_create_masking_cleanup_hook(name))
    setattr(module, f"{name}_remove_posthook", remove_posthook)


def add_mask(
        module: nn.Module, masking_type: MaskingType = MaskingType.Soft, mask_zeros: bool = True,
        mask_name: str = "weight"
) -> None:
    _add_attr_mask(module, mask_name, masking_type, mask_zeros)


def _remove_attr_mask(module: nn.Module, name: str) -> None:
    orig = module._parameters.get(f"{name}_orig", None)

    if orig is None:
        return

    # update the orig with the current mask
    mask = module._buffers[f"{name}_mask"]
    orig.data = (orig * mask).detach()

    # delete
    remove_prehook = getattr(module, f"{name}_remove_prehook")
    remove_prehook.remove()
    delattr(module, f"{name}_remove_prehook")

    remove_posthook = getattr(module, f"{name}_remove_posthook")
    remove_posthook.remove()
    delattr(module, f"{name}_remove_posthook")

    masking_type: MaskingType = getattr(module, f"{name}_masking_type")

    if masking_type == MaskingType.Trainable:
        del module._parameters[f"{name}_mask"]
    else:
        del module._buffers[f"{name}_mask"]
    del module._parameters[f"{name}_orig"]

    # Remove the masking type
    delattr(module, f"{name}_masking_type")

    setattr(module, name, orig)


def remove_mask(module: nn.Module, mask_name: str = "weight") -> None:
    _remove_attr_mask(module, mask_name)