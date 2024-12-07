from typing import List, Tuple
import torch


KnapsackGroup = Tuple[torch.Tensor, torch.Tensor]


def is_power_two(n):
    return (n & (n-1) == 0) and n != 0


def _condense_tradeoff(group: KnapsackGroup, capacity: float) -> Tuple[KnapsackGroup, torch.Tensor]:
    """
    Condenses the given tradeoff group. Assumes group is total/cumulative value-cost pairs (not individual value-cost).
    """
    value, cost = group
    del group

    # Sort the value descending
    idxs = value.argsort(descending=True)
    value = value[idxs]

    mask = torch.ones_like(idxs).type(torch.bool)
    cost = cost[idxs]

    # Keep value only if equal to smallest cost seen so far
    # Otherwise, there is a bigger value with lower cost
    cum_mincost, _ = cost.cummin(0)
    mask.logical_and_((cum_mincost == cost))

    # Keep only the largest value for each cost
    mask[1:].logical_and_(cum_mincost[1:] != cum_mincost[:-1])

    # Keep only values meeting the capacity
    mask.logical_and_(cost <= capacity) # mask.logical_and_(cost <= capacity.item())

    # Apply current mask
    value = value[mask]
    cost = cost[mask]
    idxs = idxs[mask]
    mask = mask[mask]

    # Keep only the lowest cost for each value (in case of duplicates)
    mask[:-1].logical_and_(value[1:] != value[:-1])
    return (value[mask], cost[mask]), idxs[mask]


# @torch.jit.script
def _merge_costs_sets(
    left: KnapsackGroup, right: KnapsackGroup,
    capacity: float
) -> Tuple[KnapsackGroup, torch.Tensor, torch.Tensor]:
    """
    Returns the merged tradeoff tensor, of size R <= N*M.
    Space is O(N*M). Runtime is O(N*M log(N*M)).

    Args:
        left: Tuple of values and corresponding costs. Each has shape (N,)
        right: Tuple of values and corresponding costs. Each has shape (M,)

    Returns:
        value: Merged tradeoff value-cost tuple. Each has shape (R,)
        left_keep: Index from the left tradeoff used in the merge. Shape (R,)
        right_keep: Index from the right tradeoff used in the merge. Shape (R,)
    """

    left_value, left_cost = left
    right_value, right_cost = right
    M = right_value.shape[0]

    value = (left_value.view(-1, 1) + right_value.view(1, -1)).flatten()
    cost = (left_cost.view(-1, 1) + right_cost.view(1, -1)).flatten()

    # del left_value, left_cost, left
    # del right_value, right_cost, right

    merged, idxs = _condense_tradeoff((value, cost), capacity)

    left_keep = torch.div(idxs, M, rounding_mode="trunc")
    right_keep = torch.fmod(idxs, M)

    return merged, left_keep, right_keep


def _group_knapsack_mim_recursive(
    groups: List[KnapsackGroup], capacity: float, keep_idxs: torch.Tensor
) -> Tuple[float, float]:
    """
    Floating-point compatible meet-in-the-middle group knapsack solver.

    In the worst-case, space is O(B^(G/2)) and runtime is O(B^(G/2) log(B)), where B is the size of the largest tradeoff set.
    If the costs are non-negative integer <= C, in the worst-case, space is O(C^2) and runtime is O(G C^2 log(C)).

    Args:
        groups: Holds the value-cost tradeoff for each group. Mutated by the method. Length G.
        capacity: Max capacity of the knapsack.
        keep_idxs: Storage for the keep_idxs making up the optimal knapsack. Shape (G,). Correctly set by the method.

    Returns:
        best_value: Optimal value achieved.
        best_cost: Cost at optimal value
    """
    G = len(groups)

    if G == 1:
        value, cost = groups[0]
        try:
            cidx = value.argmax()
        except ValueError as e:
            raise Exception("Optimal combination was not found! Try to decrease chunk size or compression ratio.") \
                from e
        keep_idxs[0] = cidx

        return value[cidx].item(), cost[cidx].item()

    new_groups: List[KnapsackGroup] = []
    keeps: List[Tuple[torch.Tensor, torch.Tensor]] = []
    while len(groups) > 1:
        left = groups.pop(0)
        right = groups.pop(0)
        merged, left_keep, right_keep = _merge_costs_sets(left, right, capacity)
        new_groups.append(merged)
        keeps.append((left_keep, right_keep))

    new_groups.extend(groups)

    best_value, best_cost = _group_knapsack_mim_recursive(new_groups, capacity, keep_idxs[::2])

    for idx, (left_keep, right_keep) in enumerate(keeps):
        kidx = keep_idxs[2*idx]
        keep_idxs[2*idx + 1] = right_keep[kidx]
        keep_idxs[2*idx] = left_keep[kidx] # Careful: overwrites what kidx refers to

    return best_value, best_cost


def add_zero_tradeoff(group: KnapsackGroup) -> KnapsackGroup:
    # Add an artificial (0, 0) choice to the group
    v, c = group

    return (
        torch.cat([torch.tensor([0], dtype=v.dtype, device=v.device), v]),
        torch.cat([torch.tensor([0], dtype=c.dtype, device=c.device), c])
    )


def _group_knapsack_mim(groups: List[KnapsackGroup], capacity: float, require_item: bool) -> Tuple[float, float, torch.Tensor]:
    """
    Small wrapper around `_group_knapsack_mim_recursive`. Floating-point costs and capacity are allowed.

    See `group_knapsack` for information on signature.
    """
    G = len(groups)

    if require_item:
        agroups = list(groups)
    else:
        agroups = [add_zero_tradeoff(g) for g in groups]

    if G < 1:
        return -float("Inf"), float("Inf"), torch.tensor([])

    keep_idxs = torch.empty(G, dtype=torch.long, device=agroups[0][0].device)
    best_value, best_cost = _group_knapsack_mim_recursive(agroups, capacity, keep_idxs)

    if not require_item:
        keep_idxs -= 1 # Removes artificial 0 choice

    return best_value, best_cost, keep_idxs


def apply_constraint(cgroup, cidxs, orig_group_len):
    """
    Constrain the search space by:
        power of 2
    """
    device = cgroup[0].device
    mask = torch.ones_like(cidxs).type(torch.bool)
    # Also keep original number of channels while searching (orig_group_len) to get closer capacity
    right_idxs = torch.Tensor([is_power_two(check+1) or (check+1)==orig_group_len for check in cidxs]).type(torch.bool)
    right_idxs = right_idxs.to(device)
    mask = mask.logical_and_(right_idxs)
    mask.to(device)
    idxs = cidxs[mask]
    value = cgroup[0][mask]
    cost = cgroup[1][mask]

    return (value, cost), idxs


def group_knapsack(groups: List[KnapsackGroup], capacity: float, require_item: bool = True, constraint: bool = False) -> Tuple[float, float, torch.Tensor]:
    """
    Args:
        groups: List of cumulative value-cost pairs for each group. Shape (G,). Not mutated.
        capacity: Knapsack capacity C.
        require_item: Whether an item from each group must be selected

    Returns:
        best_value: Optimal value achieved
        best_cost: Cost at optimal value
        keep_idxs: Indexes of items from each group used. Shape (G,)
    """

    # Condense the tradeoffs (by rejecting obviously suboptimal choices)
    cgroups = []
    condense_idxs = []
    for _, group in enumerate(groups):
        cgroup, cidxs = _condense_tradeoff(group, capacity)
        if constraint:
            cgroup, cidxs = apply_constraint(cgroup, cidxs, len(group[0]))
        cgroups.append(cgroup)
        condense_idxs.append(cidxs)

    best_value, best_cost, keep_idxs = _group_knapsack_mim(cgroups, capacity, require_item)

    # Adjust the number of items based on the condensed tradeoffs
    for gidx, cidxs in enumerate(condense_idxs):
         if keep_idxs[gidx] >= 0:
            target_keep_idx = keep_idxs[gidx]
            target_keep_value = cidxs[target_keep_idx]
            keep_idxs[gidx] = target_keep_value + 1

    return best_value, best_cost, keep_idxs