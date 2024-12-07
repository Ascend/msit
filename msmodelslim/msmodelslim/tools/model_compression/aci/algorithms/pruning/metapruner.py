import typing, warnings
import torch
import torch.nn as nn


from torch_pruning import ops, function, dependency
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler


class MetaPruner:
    """
        Meta pruner for structural pruning.

        Args:

            # Basic
            * model (nn.Module): A to-be-pruned model
            * example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            * importance (Callable): importance estimator.
            * global_pruning (bool): enable global pruning. Default: False.
            * pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
            * pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio. Will cover pruning_ratio if specified. Default: None.
            * max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
            * iterative_steps (int): number of steps for iterative pruning. Default: 1.
            * iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
            * ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
            * round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.

            # Adavanced
            * in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
            * out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
            * num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
            * prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
            * prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
            * head_pruning_ratio (float): head pruning ratio. Default: 0.0.
            * customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
            * unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
            * root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
            * forward_fn (Callable): A function to execute model.forward. Default: None.
            * output_transform (Callable): A function to transform network outputs. Default: None.

            # Deprecated
            * channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
            * ch_sparsity (float): the same as pruning_ratio. Default: None.
            * ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
        """

    def __init__(
            self,
            # Basic
            model: nn.Module,  # a simple pytorch model
            example_inputs: torch.Tensor,  # a dummy input for graph tracing. Should be on the same
            importance: typing.Callable,  # tp.importance.Importance for group importance estimation
            global_pruning: bool = False,
            pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
            pruning_ratio_dict: typing.Dict[nn.Module, float] = None,
            # layer-specific pruning ratio, will cover pruning_ratio if specified
            max_pruning_ratio: float = 1.0,  # maximum pruning ratio. useful if over-pruning happens.
            iterative_steps: int = 1,  # for iterative pruning
            iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler,  # scheduler for iterative pruning.
            ignored_layers: typing.List[nn.Module] = None,  # ignored layers
            round_to: int = None,  # round channels to the nearest multiple of round_to

            # Advanced
            in_channel_groups=None,  # The number of channel groups for layer input
            out_channel_groups=None,  # The number of channel groups for layer output
            num_heads=None,  # The number of heads for multi-head attention
            prune_num_heads: bool = False,  # remove entire heads in multi-head attention
            prune_head_dims: bool = True,  # remove head dimensions in multi-head attention
            head_pruning_ratio: float = 0.0,  # head pruning ratio
            head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None,  # layer-specific head pruning ratio
            customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
            # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
            unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
            # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
            root_module_types=None,
            # root module for each group
            forward_fn: typing.Callable = None,  # a function to execute model.forward
            output_transform: typing.Callable = None,  # a function to transform network outputs

            # deprecated
            channel_groups=None,  # channel grouping
            ch_sparsity: float = None,
            ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
    ):
        self.model = model
        self.importance = importance

        if ch_sparsity is not None:
            warnings.warn("ch_sparsity is deprecated in v1.3.0. Please use pruning_ratio.")
            pruning_ratio = ch_sparsity
        if ch_sparsity_dict is not None:
            warnings.warn("ch_sparsity_dict is deprecated in v1.3.0. Please use pruning_ratio_dict instead.")
            pruning_ratio_dict = ch_sparsity_dict

        self.pruning_ratio = pruning_ratio
        self.pruning_ratio_dict = pruning_ratio_dict if pruning_ratio_dict is not None else {}
        self.max_pruning_ratio = max_pruning_ratio
        self.global_pruning = global_pruning

        channel_groups = channel_groups or []

        if len(channel_groups) > 0:
            warnings.warn("channel_groups is deprecated. Please use in_channel_groups and out_channel_groups instead.")
            out_channel_groups.update(channel_groups)

        if len(num_heads) > 0:
            out_channel_groups.update(num_heads)

        self.in_channel_groups = in_channel_groups or {}
        self.out_channel_groups = out_channel_groups or {}
        self.root_module_types = root_module_types
        self.round_to = round_to

        # MHA
        self.num_heads = num_heads or {}
        self.prune_num_heads = prune_num_heads
        self.prune_head_dims = prune_head_dims
        self.head_pruning_ratio = head_pruning_ratio

        self.find_ignored_layers(ignored_layers)

        ###############################################
        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
            ignored_params=self.ignored_params,
        )

        ###############################################
        # Iterative pruning
        # The pruner will prune the model iteratively for several steps to achieve the target pruning ratio
        # E.g., if iterative_steps=5, pruning_ratio=0.5, the pruning ratio of each step will be [0.1, 0.2, 0.3, 0.4, 0.5]
        self.setup_iterative_pruning(iterative_steps,
                                     iterative_pruning_ratio_scheduler,
                                     pruning_ratio_dict,
                                     head_pruning_ratio_dict)

    def find_ignored_layers(self, ignored_layers):
        self.ignored_layers = []
        self.ignored_params = []
        if ignored_layers is not None:
            for layer in ignored_layers:
                if isinstance(layer, nn.Module):
                    self.ignored_layers.extend(list(layer.modules()))
                elif isinstance(layer, nn.Parameter):
                    self.ignored_params.append(layer)

    def compute_layer_specific_pruning_ratio(self, pruning_ratio_dict):
        for module in pruning_ratio_dict:
            ratio = pruning_ratio_dict[module]
            for submodule in module.modules():
                prunable_types = tuple([ops.type2class(
                    prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                if isinstance(submodule, prunable_types):
                    self.pruning_ratio_dict[submodule] = self.iterative_pruning_ratio_scheduler(
                        ratio, self.iterative_steps
                    )