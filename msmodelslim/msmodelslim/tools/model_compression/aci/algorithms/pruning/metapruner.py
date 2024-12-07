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
            channel_groups = None,  # channel grouping
            ch_sparsity: float = None,
            ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
            initial_total_channels=0,
            initial_total_heads = 0
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

        self.channel_groups = channel_groups or {}
        self.out_channel_groups = out_channel_groups or {}
        if len(self.channel_groups) > 0:
            warnings.warn("channel_groups is deprecated. Please use in_channel_groups and out_channel_groups instead.")
            self.out_channel_groups.update(self.channel_groups)

        self.num_heads = num_heads or {}
        if len(self.num_heads) > 0:
            self.out_channel_groups.update(self.num_heads)

        self.in_channel_groups = in_channel_groups or {}
        self.root_module_types = root_module_types or []
        self.round_to = round_to

        # MHA
        self.prune_num_heads = prune_num_heads
        self.prune_head_dims = prune_head_dims
        self.head_pruning_ratio = head_pruning_ratio

        self.find_ignored_layers(ignored_layers)
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        self.init_num_heads = {}
        self.initial_total_channels = initial_total_channels
        self.initial_total_heads = initial_total_heads



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

    def compute_head_pruning_ratio(self, head_pruning_ratio_dict):
        for module in head_pruning_ratio_dict:
            ratio = head_pruning_ratio_dict[module]
            for submodule in module.modules():
                prunable_types = tuple([ops.type2class(
                    prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                if isinstance(submodule, prunable_types):
                    self.head_pruning_ratio_dict[submodule] = self.iterative_pruning_ratio_scheduler(
                        ratio, self.iterative_steps
                    )

    def detect_group_convs_norms(self):
        for m in self.model.modules():
            layer_pruner = self.DG.get_pruner_of_module(m)
            in_ch_group = layer_pruner.get_in_channel_groups(m)
            out_ch_group = layer_pruner.get_out_channel_groups(m)
            if isinstance(m, ops.TORCH_CONV) and m.groups == m.out_channels:
                continue
            if in_ch_group > 1:
                self.in_channel_groups[m] = in_ch_group
            if out_ch_group > 1:
                self.out_channel_groups[m] = out_ch_group

    def initial_channels(self):
        for m in self.DG.module2node.keys():
            if (ops.module2type(m) in self.DG.REGISTERED_PRUNERS) or (
            isinstance(m, tuple(self.DG.CUSTOMIZED_PRUNERS.keys()))):
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)
                if m in self.num_heads:
                    self.init_num_heads[m] = self.num_heads[m]

    def count_total_initial_channels(self):
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            group = self._downstream_node_as_root_if_attention(group)
            self.initial_total_channels += (
                        (self.DG.get_out_channels(group[0][0].target.module)) // self._get_channel_groups(group))
            for dep, _ in group:
                if dep.target.module in self.num_heads and self.DG.is_out_channel_pruning_fn(dep.handler):
                    self.initial_total_heads += self.num_heads[dep.target.module]
                    break  # only count heads once


    def setup_iterative_pruning(self,
                                iterative_steps,
                                iterative_pruning_ratio_scheduler,
                                pruning_ratio_dict,
                                head_pruning_ratio_dict):
        self.iterative_steps = iterative_steps
        self.iterative_pruning_ratio_scheduler = iterative_pruning_ratio_scheduler
        self.current_step = 0
        # channel pruning ratio for each iterative step
        self.per_step_pruning_ratio = self.iterative_pruning_ratio_scheduler(
            self.pruning_ratio, self.iterative_steps
        )
        self.per_step_head_pruning_ratio = self.iterative_pruning_ratio_scheduler(
            self.head_pruning_ratio, self.iterative_steps
        )

        self.pruning_ratio_dict = {}
        if pruning_ratio_dict is not None:
            self.compute_layer_specific_pruning_ratio(pruning_ratio_dict)

        self.head_pruning_ratio_dict = {}
        if head_pruning_ratio_dict is not None:
            self.compute_head_pruning_ratio(head_pruning_ratio_dict)

        self.detect_group_convs_norms()
        self.initial_channels()

        if self.global_pruning:
            self.count_total_initial_channels()

        return

    def step(self, interactive=False) -> typing.Union[typing.Generator, None]:
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        if interactive:  # yield groups for interactive pruning
            return pruning_method()
        else:
            for group in pruning_method():
                group.prune()

    def estimate_importance(self, group, ch_groups=1) -> torch.Tensor:
        return self.importance(group, ch_groups=ch_groups)

    def pruning_history(self) -> typing.List[typing.Tuple[str, bool, typing.Union[list, tuple]]]:
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history) -> None:
        self.DG.load_pruning_history(pruning_history)

    def get_target_pruning_ratio(self, module) -> float:
        s = self.pruning_ratio_dict.get(module, self.per_step_pruning_ratio)[self.current_step]
        return min(s, self.max_pruning_ratio)

    def get_target_head_pruning_ratio(self, module) -> float:
        s = self.head_pruning_ratio_dict.get(module, self.per_step_head_pruning_ratio)[self.current_step]
        return min(s, 1)

    def reset(self) -> None:
        self.current_step = 0

    def regularize(self, model, loss) -> typing.Any:
        """ Model regularizor for sparse training
        """
        pass

    def _check_pruning_ratio(self, group) -> bool:
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                        1 - self.max_pruning_ratio
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                        1 - self.max_pruning_ratio
                ) or layer_in_ch == 1:
                    return False
        return True

    def _is_attn_group(self, group) -> bool:
        is_attn = False
        qkv_layers = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if self.DG.is_out_channel_pruning_fn(pruning_fn) and module in self.num_heads:
                qkv_layers.append(module)
                is_attn = True
        return is_attn, qkv_layers

    def _get_channel_groups(self, group) -> int:
        ch_groups = 1

        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            channel_groups = self.out_channel_groups if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.in_channel_groups

            if module in channel_groups:
                ch_groups = channel_groups[module]
        return ch_groups  # no channel grouping

    def _downstream_node_as_root_if_attention(self, group):
        # Use a downstream node as the root if torch.unbind exists. TODO: find a general way to handle torch.unbind in timm
        is_attention = False
        downstream_dep = None
        for _dep, _idxs in group:
            if _dep.source.module in self.num_heads and self.DG.is_out_channel_pruning_fn(_dep.handler):
                is_attention = True
            if isinstance(_dep.target.module, tuple(self.root_module_types)) and self.DG.is_in_channel_pruning_fn(
                    _dep.handler):
                downstream_dep = _dep
        if is_attention and downstream_dep is not None:  # use a downstream node as the root node for attention layers
            group = self.DG.get_pruning_group(downstream_dep.target.module, downstream_dep.handler, _idxs)
        return group

    def _round_to(self, n_pruned, current_channels, round_to):
        rounded_channels = current_channels - n_pruned
        rounded_channels = rounded_channels - rounded_channels % round_to
        n_pruned = current_channels - rounded_channels
        return max(n_pruned, 0)

    def compute_target_prune(self, pruning_fn, module):
        if self.DG.is_out_channel_pruning_fn(pruning_fn):
            current_channels = self.DG.get_out_channels(module)
            target_pruning_ratio = self.get_target_pruning_ratio(module)
            target_pruned = current_channels - int(
                self.layer_init_out_ch[module] *
                (1 - target_pruning_ratio)
            )
        else:
            current_channels = self.DG.get_in_channels(module)
            target_pruning_ratio = self.get_target_pruning_ratio(module)
            target_pruned = current_channels - int(
                self.layer_init_in_ch[module] *
                (1 - target_pruning_ratio)
            )
        return current_channels, target_pruned

    def recalculate_num_heads(self, group, n_heads_removed):
        for dep, _ in group:
            if dep.target.module in self.num_heads:
                self.num_heads[dep.target.module] -= n_heads_removed
                self.out_channel_groups[dep.target.module] = self.num_heads[dep.target.module]

    def compute_indices_ch_groups(self, pruning_indices,
                                  imp, ch_groups, n_pruned_per_group, group_size):
        for chg in range(ch_groups):
            sub_group_imp = imp[chg * group_size: (chg + 1) * group_size]
            sub_imp_argsort = torch.argsort(sub_group_imp)
            sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size  # offset
            pruning_indices.append(sub_pruning_idxs)
        pruning_indices = torch.unique(torch.cat(pruning_indices, 0))

    def compute_prune_indices_linear(self, imp, pruning_indices, ch_groups, group_size,
                                     current_channels, target_pruned, round_function=None):
        n_pruned = round_function(target_pruned, current_channels, self.round_to)
        if ch_groups > 1:  # independent pruning for each group
            n_pruned_per_group, remainder = divmod(n_pruned, ch_groups)
            if remainder > 0:
                n_pruned_per_group = round_function(n_pruned_per_group, group_size, self.round_to)
            if n_pruned_per_group > 0:
                self.compute_indices_ch_groups(pruning_indices, imp,
                                               ch_groups, n_pruned_per_group, group_size)
        else:
            pruning_indices = torch.argsort(imp)[:n_pruned]

    def compute_prune_indices_attention_heads(self, imp, pruning_indices, module,
                                              qkv_layers, ch_groups, group_size, current_channels,
                                              round_to_power_of_2=False):
        target_head_pruning_ratio = self.get_target_head_pruning_ratio(qkv_layers[0])
        target_pruned = current_channels - int(
            self.layer_init_out_ch[module] *
            (1 - target_head_pruning_ratio)
        )
        n_heads_removed = self.num_heads[qkv_layers[0]] - int(
            self.init_num_heads[qkv_layers[0]] * (1 - target_head_pruning_ratio))
        if round_to_power_of_2:
            n_heads_removed = self._round_to_power_of_2(n_heads_removed, self.num_heads[qkv_layers[0]])
        if (n_heads_removed > 0) and (n_heads_removed < self.num_heads[qkv_layers[0]]):
            head_imp = imp.view(ch_groups, -1).mean(1)
            for head_id in torch.argsort(head_imp)[:n_heads_removed]:
                pruning_indices.append(
                    torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))
        return n_heads_removed

    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):  # check pruning ratio
                ##################################
                # Compute raw importance score
                ##################################
                group = self._downstream_node_as_root_if_attention(group)
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue

                ##################################
                # Compute the number of dims/channels to prune
                ##################################
                current_channels, n_pruned = self.compute_target_prune(pruning_fn, module)
                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                ##################################
                # collect pruning idxs
                ##################################
                pruning_idxs = []
                _is_attn, qkv_layers = self._is_attn_group(group)
                group_size = current_channels // ch_groups
                # dims/channels
                if n_pruned > 0:
                    self.compute_prune_indices_linear(imp, pruning_indices, ch_groups, group_size,
                                                      current_channels, target_pruned, self._round_to)

                else:  # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs.append(imp_argsort[:n_pruned])

                # num heads
                n_heads_removed = 0
                if _is_attn and self.prune_num_heads:  # Prune entire attn heads
                    n_heads_removed = self.compute_prune_indices_attention_heads(imp, pruning_indices, module,
                                                                                 qkv_layers, ch_groups, group_size,
                                                                                 current_channels, False)
                if len(pruning_idxs) == 0: continue
                pruning_idxs = torch.unique(torch.cat(pruning_idxs, 0)).tolist()
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs)

                con1 = self.DG.check_pruning_group(group) and _is_attn
                con2 = self.prune_num_heads and n_heads_removed

                if con1 and con2 > 0:
                    self.recalculate_num_heads(group, n_heads_removed)
                if self.DG.check_pruning_group(group):
                    yield group

    def global_compute_prune_indices_linear(self, pruning_indices, imp, module, group,
                                            thres, ch_groups, group_size,
                                            get_channel_fn):
        if ch_groups > 1:  # re-compute importance for each channel group if channel grouping is enabled
            n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
            if n_pruned_per_group > 0:
                if self.round_to:
                    n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                _is_attn, _ = self._is_attn_group(group)
                if not _is_attn or self.prune_head_dims:
                    raw_imp = self.estimate_importance(group, ch_groups=ch_groups)  # re-compute importance
                    self.compute_indices_ch_groups(pruning_indices, raw_imp,
                                                   ch_groups, n_pruned_per_group, group_size)
        else:
            _pruning_indices = (imp <= thres).nonzero().view(-1)
            imp_argsort = torch.argsort(imp)
            if len(_pruning_indices) > 0 and self.round_to:
                n_pruned = len(_pruning_indices)
                current_channels = get_channel_fn(module)
                n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                _pruning_indices = imp_argsort[:n_pruned]
            pruning_indices.append(_pruning_indices)

    def global_compute_prune_indices_attn_heads(self, pruning_indices, qkv_layers,
                                                head_imp, head_thres,
                                                group_size):
        head_pruning_indices = (head_imp <= head_thres).nonzero().view(-1)
        if len(head_pruning_indices) > 0:
            for head_id in head_pruning_indices:
                pruning_indices.append(
                    torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))
        for qkv_layer in qkv_layers:
            self.num_heads[qkv_layer] -= len(head_pruning_indices)  # update num heads after pruning

    def global_calculate_threshol(self, global_importance):
        concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        target_pruning_ratio = self.per_step_pruning_ratio[self.current_step]
        n_pruned = len(concat_imp) - int(
            self.initial_total_channels *
            (1 - target_pruning_ratio)
        )
        if n_pruned > 0:
            topk_imp, _ = torch.topk(concat_imp, k=n_pruned, largest=False)
            thres = topk_imp[-1]
            return (n_pruned, topk_imp, thres)
        else:
            return (n_pruned, None, None)

    def global_calculate_threshol_head_pruning(self, global_head_importance):
        concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
        target_head_pruning_ratio = self.per_step_head_pruning_ratio[self.current_step]
        n_heads_removed = len(concat_head_imp) - int(
            self.initial_total_heads *
            (1 - target_head_pruning_ratio)
        )
        if n_heads_removed > 0:
            topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_removed, largest=False)
            head_thres = topk_head_imp[-1]
            return (n_heads_removed, topk_head_imp, head_thres)
        else:
            return (n_heads_removed, None, None)

    def global_calculate_importance(self, group, ch_groups):
        group_size = len(imp) // ch_groups
        if ch_groups > 1:
            dim_imp = imp.view(ch_groups, -1).mean(dim=0)
        else:
            dim_imp = imp
        global_importance.append((group, ch_groups, group_size, dim_imp))

        # pre-compute head importance for attn heads
        _is_attn, qkv_layers = self._is_attn_group(group)
        if _is_attn and self.prune_num_heads and self.get_target_head_pruning_ratio(qkv_layers[0]) > 0:
            head_imp = imp.view(ch_groups, -1).mean(1)  # average importance by head.
            global_head_importance[group] = (qkv_layers, head_imp)
        return (group_size, global_importance, global_head_importance, _is_attn, qkv_layers)

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        global_head_importance = {}  # for attn head pruning
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):
                group = self._downstream_node_as_root_if_attention(
                    group)  # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)  # raw importance score

                if imp is None: continue
                group_size, global_importance, global_head_importance, _is_attn, qkv_layers = self.global_calculate_importance(
                    group, ch_groups)

        if len(global_importance) == 0 and len(global_head_importance) == 0:
            return

        #######################################################
        # 2. Thresholding by concatenating all importance scores.
        #######################################################

        if len(global_importance) > 0:  # Find the threshold for global pruning
            n_pruned, topk_imp, thres = self.global_calculate_threshol(global_importance)

        # Find the threshold for head pruning
        if len(global_head_importance) > 0:
            n_heads_removed, topk_head_imp, head_thres = self.global_calculate_threshol_head_pruning(
                global_head_importance)

        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.DG.get_in_channels

            # Prune feature dims/channels
            pruning_indices = []
            if len(global_importance) > 0 and n_pruned > 0:
                self.global_compute_prune_indices_linear(pruning_indices, imp, module, group,
                                                         thres, ch_groups, group_size,
                                                         get_channel_fn)
                # Prune heads
            if len(global_head_importance) > 0 and n_heads_removed > 0 and (group in global_head_importance):
                qkv_layers, head_imp = global_head_importance[group]
                self.global_compute_prune_indices_attn_heads(pruning_indices, qkv_layers,
                                                             head_imp, head_thres,
                                                             group_size)

            if len(pruning_indices) == 0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self.DG.check_pruning_group(group):
                yield group