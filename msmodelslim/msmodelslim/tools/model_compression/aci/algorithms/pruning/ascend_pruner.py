import torch
import numpy as np
import math
import typing, warnings

from .metapruner import MetaPruner as BasePruner
from torch_pruning import ops, function


def min_cost_knapsack(weights, prices, C):
    n = len(weights)

    if sum(weights) <= C:
        return sum(prices), C, list(range(n))

    W = np.cumsum(weights)
    P = np.cumsum(prices)

    i_c = n - 1
    for i in range(len(W)):
        if W[i] > C:
            i_c = i
            break

    W = np.roll(W, 1)
    W[0] = 0
    P = np.roll(P, 1)
    P[0] = 0

    opt = np.inf
    C_star = -1
    i_star = -1

    dp = [[None] * (C + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(C + 1):
            if j == 0:
                dp[i][j] = 0
            else:
                dp[i][j] = np.inf

    for i in range(n - 1, -1, -1):
        if i <= i_c:
            C_max = max(0, C - W[i])
            C_min = max(0, C - W[i] - weights[i] + 1)

            tmp = min([dp[i + 1][k] for k in range(C_min, C_max + 1)]) + P[i]
            if tmp < opt:
                opt = tmp
                i_star = i
                C_star = np.argmin([dp[i + 1][k] + P[i_star] for k in range(C_min, C_max + 1)]) + W[i_star] + C_min

        for k in range(C, weights[i] - 1, -1):
            dp[i][k] = min(dp[i + 1][k], dp[i + 1][k - weights[i]] + prices[i])

    result = list(range(0, i_star))
    k = C_star - W[i_star]
    for i in range(i_star, n):
        if k <= 0:
            break
        if np.isclose(dp[i][k], dp[i + 1][k - weights[i]] + prices[i]):
            result.append(i)
            k -= weights[i]

    return opt, C_star, result


class AscendPruner(BasePruner):
    """
        Pruner which produces pruned model with number of channels optimized for Ascend310P (powers of 2)
    """

    def __init__(self,
                 model,
                 example_inputs,
                 importance,
                 global_pruning=False,
                 pruning_ratio=0.5,
                 pruning_steps=5,
                 local_power_of_two_pruning=True,
                 unwrapped_parameters=None,
                 ignored_layers=None,
                 channel_groups={},
                 output_transform=None,
                 num_heads={},
                 prune_head_dims=True,
                 prune_num_heads=False,
                 head_pruning_ratio=0.5,  # disabled when prune_num_heads=False
                 customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
                 root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM]
                 # root module for each group
                 ):

        super(AscendPruner, self).__init__(model,
                                           example_inputs,
                                           importance,
                                           global_pruning=global_pruning,
                                           pruning_ratio=pruning_ratio,
                                           iterative_steps=pruning_steps,
                                           unwrapped_parameters=unwrapped_parameters,
                                           ignored_layers=ignored_layers,
                                           channel_groups=channel_groups,
                                           output_transform=output_transform,
                                           num_heads=num_heads,
                                           prune_head_dims=prune_head_dims,
                                           prune_num_heads=prune_num_heads,
                                           head_pruning_ratio=head_pruning_ratio,
                                           customized_pruners=customized_pruners,
                                           root_module_types=root_module_types
                                           )

        self.local_power_of_two_pruning = local_power_of_two_pruning

    def step(self, interactive=False) -> typing.Union[typing.Generator, None]:
        self.current_step += 1
        if self.global_pruning:
            pruning_method = self.prune_global
        elif self.local_power_of_two_pruning:
            pruning_method = self.prune_local_power_of_two
        else:
            pruning_method = self.prune_local

        if interactive:  # yield groups for interactive pruning
            return pruning_method()
        else:
            for group in pruning_method():
                group.prune()

    def _round_to_power_of_2(self, n_pruned, current_channels, round_to=None):
        rounded_channels = current_channels - n_pruned
        possible_power_floor = math.floor(math.log(rounded_channels, 2))
        possible_power_ceil = math.ceil(math.log(rounded_channels, 2))

        if current_channels <= (2 ** possible_power_ceil):
            power = possible_power_floor
        else:
            power = possible_power_ceil  # power = min([possible_power_floor, possible_power_floor], key= lambda z: abs(rounded_channels - 2**z))

        n_pruned = current_channels - (2 ** power)
        return n_pruned

    def solve_knapsack(self, groups, capacity, costs, weights):
        capacity = int(capacity)
        costs = np.array(costs)
        weights = np.array(weights)

        idx = np.argsort(weights)
        weights = weights[idx]
        costs = costs[idx]
        groups = [groups[i] for i in idx]

        _, C_star, result = min_cost_knapsack(weights, costs, capacity)

        if C_star <= 0:
            return

        pruning_groups = [groups[i] for i in result]
        return pruning_groups

    def prune_local_power_of_two(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        capacity = 0.0
        costs = []
        weights = []
        groups = []

        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):  # check pruning ratio

                group = self._downstream_node_as_root_if_attention(group)

                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)

                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue

                current_channels, target_pruned = self.compute_target_prune(pruning_fn, module)

                if target_pruned <= 0:
                    continue

                group_size = current_channels // ch_groups
                _is_attn, qkv_layers = self._is_attn_group(group)

                pruning_indices = []

                if (self.prune_head_dims and _is_attn) or (not _is_attn):
                    self.compute_prune_indices_linear(imp, pruning_indices, ch_groups, group_size,
                                                      current_channels, target_pruned, self._round_to_power_of_2)

                n_heads_removed = 0
                if _is_attn and self.prune_num_heads:  # Prune entire attn heads
                    n_heads_removed = self.compute_prune_indices_attention_heads(imp, pruning_indices, module,
                                                                                 qkv_layers, ch_groups, group_size,
                                                                                 current_channels, True)
                    pruning_indices = torch.unique(torch.cat(pruning_indices, 0))

                if len(pruning_indices) == 0: continue
                capacity += target_pruned
                imp_pruning = imp[pruning_indices]
                cost = imp_pruning.mean().item()
                weight = len(pruning_indices)

                costs.append(cost)
                weights.append(weight)
                groups.append((module, pruning_fn, pruning_indices, n_heads_removed))

        pruning_groups = self.solve_knapsack(groups, capacity, costs, weights)

        for group in pruning_groups:
            module, pruning_fn, pruning_indices, n_heads_removed = group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            _is_attn, _ = self._is_attn_group(group)

            if self.DG.check_pruning_group(group) and _is_attn and self.prune_num_heads and n_heads_removed > 0:
                self.recalculate_num_heads(group, n_heads_removed)

            if self.DG.check_pruning_group(group):
                yield group 