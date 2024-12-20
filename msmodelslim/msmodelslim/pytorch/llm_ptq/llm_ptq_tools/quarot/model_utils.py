# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import torch


def get_embeddings(model) -> list[torch.nn.Module]:
    embedding_list = []
    for name, module in model.named_modules():
        lowest_level = name.split(".")[-1]
        if "embed" in lowest_level:
            embedding_list.append(module)
    if len(embedding_list) > 1:
        embedding_list = embedding_list[:1]
    elif len(embedding_list) <= 0:
        raise ValueError("Cannot recognize model's embedding layers!")
    return embedding_list


def get_transformer_layers(model) -> list[torch.nn.Module]:
    for name, module in model.named_modules():
        lowest_level = name.split(".")[-1]
        if lowest_level == "layers":
            return [layer for layer in module]
    raise ValueError(f'Cannot recognize model layers!')


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, device, eps=1e-5, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(mean_dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)
