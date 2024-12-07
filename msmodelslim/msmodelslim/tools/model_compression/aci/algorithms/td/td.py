import sys
import math
from pathlib import Path
import re
import torch
import torch.nn as nn
import numpy as np

from scipy.sparse.linalg import svds
from tensorly.decomposition import partial_tucker
from skopt.space import Integer
from skopt import Optimizer
from transformers import Conv1D

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # ACI root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from aci.utils import Logger

logger = Logger().logger


class DecomposedLinear(nn.Module):
    """
    Decomposed (or compressed) linear layer.
    """

    def __init__(self, layer, rank, init=True):
        """
        Class initializer.
        """
        super(DecomposedLinear, self).__init__()

        device = 'cpu'
        weight = layer.weight.data.cpu().numpy()
        out_dim, in_dim = weight.shape
        out_rank = rank
        self.in_layer = nn.Linear(in_features=in_dim, out_features=out_rank, bias=False).to(device)
        self.out_layer = nn.Linear(in_features=out_rank, out_features=out_dim, bias=layer.bias is not None).to(device)

        if init:
            u, s, v = svds(weight.astype(np.float32), k=rank)
            first = torch.from_numpy(np.array(v.astype(np.float16))).to(device)
            last = torch.from_numpy(np.array(u.dot(np.diag(s))).astype(np.float16)).to(device)

            if self.out_layer.bias is not None:
                self.out_layer.bias.data = layer.bias.data

            self.in_layer.weight.data = first
            self.out_layer.weight.data = last

    def forward(self, x):
        """
        Forward propagation.
        """
        x = self.in_layer(x)
        x = self.out_layer(x)
        return x


class DecomposedConv1D(nn.Module):
    def __init__(self, layer, rank, init=True):
        """
        Class initializer.
        """
        super(DecomposedConv1D, self).__init__()
        device = layer.weight.device
        weight = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy()
        in_dim, out_dim = weight.shape
        out_rank = rank

        self.in_layer = Conv1D(out_rank, in_dim).to(device)
        self.out_layer = Conv1D(out_dim, out_rank).to(device)

        if init:
            u, s, v = svds(weight.astype(np.float32), k=rank)
            first = torch.from_numpy(np.array(v.astype(np.float16))).to(device)
            last = torch.from_numpy(np.array(u.dot(np.diag(s))).astype(np.float16)).to(device)

            self.out_layer.bias.data = layer.bias.data

            self.in_layer.weight.data = last
            self.out_layer.weight.data = first

    def forward(self, x):
        """
        Forward propagation.
        """
        x = self.in_layer(x)
        x = self.out_layer(x)
        return x


def calculate_parameters_number(model):
    cnt = 0
    for p in model.named_parameters():
        cnt += p[1].numel()
    return cnt


def optimize_rank(rank, model, linear_layer):
    rounded_rank = 2 ** int(rank)

    weight = linear_layer.weight.data.cpu().numpy()
    device = linear_layer.weight.device

    if isinstance(linear_layer, nn.Linear):
        original_output = linear_layer(torch.randn(1, linear_layer.weight.data.cpu().numpy().shape[1]).to(device))
        # Perform SVD decomposition
        u, s, v = svds(weight.astype(np.float32), k=rounded_rank)

        # Truncate the singular values based on the estimated rank
        s[rounded_rank:] = 0

        # Reconstruct the weight matrix
        u = torch.from_numpy(u.copy()).to(device)
        s = torch.from_numpy(s.copy()).to(device)
        v = torch.from_numpy(v.copy()).to(device)
        reconstructed_weight = torch.matmul(torch.matmul(u, torch.diag(s)), v).to(device)

        # Set the new weight matrix for the linear layer
        linear_layer.weight.data = reconstructed_weight
        output = linear_layer(torch.randn(1, linear_layer.weight.data.cpu().numpy().shape[1]).to(device))

    elif isinstance(linear_layer, nn.Conv2d):
        original_output = model(torch.randn(1, 3, 224, 224).to(device))
        # Perform SVD on the convolutional layer weight
        u, s, v = torch.svd(linear_layer.weight.view(linear_layer.weight.size(0), -1).data)

        # Truncate the singular values based on the estimated rank
        u_k = u[:, :rounded_rank].to(device)
        s_k = torch.diag(s[:rounded_rank]).to(device)
        v_k = v[:, :rounded_rank].to(device)
        # Reconstruct the weight matrix
        truncated_weight = torch.mm(u_k, torch.mm(s_k, v_k.t())).view_as(linear_layer.weight).to(device)

        # Update the convolutional layer with the truncated weight
        linear_layer.weight.data = truncated_weight
        output = model(torch.randn(1, 3, 224, 224).to(device))
    elif isinstance(linear_layer, Conv1D):
        original_output = linear_layer(torch.randn(1, linear_layer.weight.data.cpu().numpy().shape[0]).to(device))
        # Perform SVD decomposition
        u, s, v = svds(weight.astype(np.float32), k=rounded_rank)

        # Truncate the singular values based on the estimated rank
        s[rounded_rank:] = 0

        # Reconstruct the weight matrix
        u = torch.from_numpy(u.copy()).to(device)
        s = torch.from_numpy(s.copy()).to(device)
        v = torch.from_numpy(v.copy()).to(device)
        reconstructed_weight = torch.matmul(torch.matmul(u, torch.diag(s)), v).to(device)

        # Set the new weight matrix for the linear layer
        linear_layer.weight.data = reconstructed_weight
        output = linear_layer(torch.randn(1, linear_layer.weight.data.cpu().numpy().shape[0]).to(device))

    # Compute the loss between the original and decomposed layers' outputs
    criterion = nn.MSELoss()
    if isinstance(output, tuple):
        if isinstance(output[0], list):
            loss = criterion(output[0][0], original_output[0][0])
        else:
            loss = criterion(output[0], original_output[0])
    else:
        loss = criterion(output, original_output)

    return loss.item()


def highestPowerof2(n):
    if (math.log2(n).is_integer()):
        return math.log2(n) - 1
    else:
        res = 0
        for i in range(n - 1, 0, -1):
            if ((i & (i - 1)) == 0):
                res = i
                break
        return math.log2(res) - 1


def DecomposedConv(linear_layer, rank):
    core, [last, first] = partial_tucker(
        linear_layer.weight.data.cpu().numpy(), modes=[0, 1], tol=10e-5, rank=[rank, rank], init="svd"
    )

    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, stride=1)

    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[0],
        out_channels=core.shape[0],
        kernel_size=linear_layer.kernel_size,
        stride=linear_layer.stride,
        padding=linear_layer.padding,
        dilation=linear_layer.dilation,
        bias=False,
    )

    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], kernel_size=1, stride=1)

    first = torch.from_numpy(first.copy())
    last = torch.from_numpy(last.copy())
    core = torch.from_numpy(core.copy())

    first_layer.weight.data = (torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1).data)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1).data
    core_layer.weight.data = core.data
    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def get_importance(model):
    weight_magnitude = {}
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            weight_magnitude[name] = [1, torch.sum(torch.abs(module.weight)).item()]
        if isinstance(module, torch.nn.Conv2d):
            if (module.kernel_size[0] <= 3) & (module.kernel_size[1] <= 3):
                weight_magnitude[name] = [module.kernel_size[0] + module.kernel_size[1],
                                          torch.sum(torch.abs(module.weight)).item()]
        if isinstance(module, torch.nn.Linear):
            weight_magnitude[name] = [1, torch.sum(torch.abs(module.weight)).item()]
    weight_magnitude = dict(sorted(weight_magnitude.items(), key=lambda x: (-x[1][0], -x[1][1])))
    return weight_magnitude


def solve_prune_td(prune_td_metrics, task):
    """
    Choose the best puning and td ratios from results dict

    return step
    """

    if len(prune_td_metrics) == 1:
        return 0

    prune_td_metrics_task = {}
    if task == 'acceleration':
        for k, v in prune_td_metrics.items():
            if (abs(0.5 - (v[2]) / (v[3])) != 0.5) & ((round(v[2], 4) - round(v[3], 4)) != 0):
                prune_td_metrics_task[k] = [v[1]]  # , abs(0.5 - (v[2]) / (v[3]))]
    elif task == 'accuracy':
        for k, v in prune_td_metrics.items():
            if (abs(0.5 - (v[2]) / (v[3])) != 0.5) & ((round(v[2], 4) - round(v[3], 4)) != 0):
                prune_td_metrics_task[k] = [v[0]]  # , abs(0.5 - (v[2]) / (v[3]))]
    else:
        for k, v in prune_td_metrics.items():
            if (abs(0.5 - (v[2]) / (v[3])) != 0.5):
                prune_td_metrics_task[k] = [v[0], v[1], abs(0.5 - (v[2]) / (v[3]))]

    ranks_dict = {k: 0 for k in prune_td_metrics_task.keys()}
    for i in range(len(list(prune_td_metrics_task.values())[0])):
        # Sort by KLdiv_loss, flops and prune/td ratio
        sorted_dict = {k: v[i] for k, v in sorted(prune_td_metrics_task.items(), key=lambda item: item[1][i])}
        values = sorted_dict.values()
        min_ = min(values)
        max_ = max(values)

        normalized_sorted_dict = {key: ((v - min_) / (max_ - min_)) for key, v in sorted_dict.items()}
        for j, key in enumerate(sorted_dict):
            ranks_dict[key] += (j + 1) * (1 + abs(normalized_sorted_dict[key]))
            logger.debug(str(key) + ' ' + str((j + 1) * (1 + abs(normalized_sorted_dict[key]))))
    result_ranks = dict(sorted(ranks_dict.items(), key=lambda item: item[1]))
    return list(result_ranks.keys())[0]


def iter_optimizer(iters, optimizer, model, linear_layer):
    for _ in range(iters):
        x = optimizer.ask()
        y = optimize_rank(**dict(zip(['rank', 'model', 'linear_layer'], x)), model=model, linear_layer=linear_layer)
        optimizer.tell(x, y)


def apply_td(model, device, par_sum_baseline, compr_ratio, iters=20):
    min_power_of_2 = 6
    imp_dict = get_importance(model)
    for k in list(imp_dict.keys()):
        logger.debug('started k is: ' + str(k))
        k = 'model.' + k
        k = re.sub('\.([0-9]+)(\.)?', '[\\1]\\2', k)  # make list indexes, e.g. [1]
        logger.debug('k is: ' + str(k))
        linear_layer = eval(k)
        logger.info('linear_layer: ' + str(linear_layer))
        con1 = isinstance(linear_layer, nn.Conv2d) or isinstance(linear_layer, nn.Linear) or isinstance(linear_layer,
                                                                                                     Conv1D)
        con2 = ('classifier' not in k) and ('score' not in k)
        if con1 and con2:
            if hasattr(linear_layer, 'groups') and linear_layer.groups > 1:
                continue
            max_power_of_2 = 2
            check_power_of_2 = highestPowerof2(min(linear_layer.weight.shape[0], linear_layer.weight.shape[1]))
            if check_power_of_2 > max_power_of_2:
                max_power_of_2 = check_power_of_2
            if max_power_of_2 < min_power_of_2:
                continue
            elif max_power_of_2 == min_power_of_2:
                optimal_rank = max_power_of_2
            else:
                optimizer = Optimizer([Integer(min_power_of_2, max_power_of_2, name='rank')], base_estimator='gp')
                iter_optimizer(iters, optimizer, model, linear_layer)
                optimal_rank = optimizer.get_result().x[0]
            logger.info("Optimal rank: " + str(2 ** int(optimal_rank)))
            if isinstance(linear_layer, nn.Conv2d):
                new_co = DecomposedConv(linear_layer, 2 ** int(optimal_rank)).to(device, non_blocking=True)
            elif isinstance(linear_layer, Conv1D):
                new_co = DecomposedConv1D(linear_layer, 2 ** int(optimal_rank)).to(device, non_blocking=True)
            elif isinstance(linear_layer, nn.Linear):
                new_co = DecomposedLinear(linear_layer, 2 ** int(optimal_rank)).to(device, non_blocking=True)
            check_seq = re.sub(r'\[[^\[\]]*\]$', '', k)
            if isinstance(eval(check_seq), nn.Sequential):
                att = re.search(r'\[([^\[\]]*)\]$', k).group(1)
                logger.debug('att: ' + str(att))
                logger.debug('eval(check_seq): ' + str(eval(check_seq)))
                setattr(eval(check_seq), att, new_co)
                logger.debug('eval(check_seq) after TD: ' + str(eval(check_seq)))
            else:
                att = k.rsplit('.', 1)[1].replace('[', '').replace(']', '')
                logger.debug('att: ' + str(att))
                logger.debug('eval(k.rsplit(''.'', 1)[0]): ' + str(eval(k.rsplit('.', 1)[0])))
                setattr(eval(k.rsplit('.', 1)[0]), att, new_co)
                logger.info('TD ended')
                logger.debug('eval(k.rsplit(''.'', 1)[0]) after TD: ' + str(eval(k.rsplit('.', 1)[0])))
            model.float()
            par_sum_td_pruned = calculate_parameters_number(model)
            logger.info('COMPRESSION IS: ' + str(par_sum_baseline / par_sum_td_pruned))
            if (par_sum_baseline / par_sum_td_pruned) > compr_ratio:
                break