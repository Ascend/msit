import os
import math

import scipy
import torch
import numpy as np

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quarot import utils


cur_dir = os.path.dirname(os.path.abspath(__file__))
binary_file_path = os.path.join(cur_dir, "hadamard_matrice.npy")
HADAMARD_MATRICES = np.load(binary_file_path, allow_pickle=True).item()


# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
def get_hadk(n, transpose=False, known_dim=None):
    had_k, k = None, None
    if known_dim:
        n = n // known_dim

    hadamard_mapping = {
        428: "get_had428",
        172: "get_had172",
        160: "get_had160",
        156: "get_had156",
        140: "get_had140",
        112: "get_had112",
        108: "get_had108",
        60: "get_had60",
        52: "get_had52",
        40: "get_had40",
        36: "get_had36",
        28: "get_had28",
        20: "get_had20",
        12: "get_had12",
    }

    for k, had_k_key in hadamard_mapping.items():
        if n % k == 0:
            if not is_pow2(n // k):
                raise ValueError(f"{n}-order hadamard matrix construct failed")
            had_k = HADAMARD_MATRICES.get(had_k_key)
            break
    else:
        if not is_pow2(n):
            raise ValueError(f"{n}-order hadamard matrix construct failed")
        k = 1

    if had_k is not None and transpose:
        had_k = had_k.T
    return had_k, k


def hadamard_transform(x, scale, transpose=False):
    n = x.shape[-1]
    hadk, k = get_hadk(n, transpose)
    shape = x.shape
    x = x.clone().contiguous().view(-1, n, 1)
    output = x.clone()
    while x.shape[1] > k:
        x = x.view(x.shape[0], x.shape[1] // 2, 2, x.shape[2])
        output = output.view(x.shape)
        output[:, :, 0, :] = x[:, :, 0, :] + x[:, :, 1, :]
        output[:, :, 1, :] = x[:, :, 0, :] - x[:, :, 1, :]
        output = output.view(x.shape[0], x.shape[1], -1)
        (x, output) = (output, x)
    del output
    return x.view(shape) * scale


def matmul_had(x, transpose=False, known_dim=None):
    n = x.shape[-1]
    shape = x.shape
    hadk, k = get_hadk(n, transpose, known_dim)
    x = x.clone().view(-1, n, 1)
    output = x.clone()
    while x.shape[1] > k:
        x = x.view(x.shape[0], x.shape[1] // 2, 2, x.shape[2])
        output = output.view(x.shape)
        output[:, :, 0, :] = x[:, :, 0, :] + x[:, :, 1, :]
        output[:, :, 1, :] = x[:, :, 0, :] - x[:, :, 1, :]
        output = output.view(x.shape[0], x.shape[1], -1)
        (x, output) = (output, x)
    del output

    if k > 1:
        x = hadk.view(1, k, k).to(x) @ x

    return x.view(shape) / torch.tensor(n).sqrt()


def matmul_had_t(x):
    return matmul_had(x, transpose=True)


def matmul_had_alt(x, hadk, k):
    n = x.shape[-1]
    if k == 1:
        return hadamard_transform(x.contiguous(), 1.0/torch.tensor(n).sqrt())
    input_ = x.view(-1, k, n // k)
    input_ = hadamard_transform(input_.contiguous(), 1.0/torch.tensor(n).sqrt())
    input_ = hadk.to(device=input_.device, dtype=input_.dtype) @ input_
    return input_.reshape(x.shape)


def kronecker_construct(x, had_k, k, n):
    if had_k is None:
        H = torch.Tensor(scipy.linalg.hadamard(n // k)).float() / math.sqrt(n)
    else:
        H_K = had_k
        H_2n = torch.Tensor(scipy.linalg.hadamard(n // k))
        I_K = torch.eye(k, device=x.device, dtype=x.dtype)
        I_2n = torch.eye(n // k, device=x.device, dtype=x.dtype)
        H = (torch.kron(I_K, H_2n) @ torch.kron(H_K, I_2n)) / math.sqrt(n)
    return (H @ x.t()).t()


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    utils.set_seed(size)
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float32)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    # could be replaced with `Q = torch.eye(size).to(torch.float32)`
    return matmul_had(Q).to(device)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, packed_weight=False, head_dim=None):
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{module} is not a torch.nn.Linear module!")

    if had_dim != -1 and not is_pow2(had_dim):
        raise ValueError("Hadamard dimension must be a power of 2!")

    in_features, out_features = module.in_features, module.out_features
    weight = module.weight.data
    dtype = weight.dtype
    device = weight.device

    weight = weight.to(device=utils.DEV, dtype=torch.float32)
    if packed_weight:
        qkv = torch.split(weight, weight.shape[0] // 3, dim=0)
        weight = qkv[-1]

    if had_dim == -1:
        if output:
            had_k, k = get_hadk(out_features)
            weight = matmul_had_alt(weight.t(), had_k, k).t()
            # could be replaced with weight = kronecker_construct(weight.t(), had_k, k, out_features).t()
        else:
            had_k, k = get_hadk(in_features, known_dim=head_dim)
            weight = matmul_had_alt(weight, had_k, k)
            # could be replaced with weight = kronecker_construct(weight, had_k, k, in_features)
    else:
        # Apply Hadamard to the last had_dim chunks of the weights
        if output:
            weight = weight.t()
            transposed_shape = weight.shape
            weight = hadamard_transform(
                weight.reshape(-1, transposed_shape[-1] // had_dim, had_dim),
                scale=1 / math.sqrt(had_dim)
            ).reshape(transposed_shape).t()
        else:
            raise NotImplementedError("Not implemented (or tested) yet!")
    if packed_weight:
        q, k = qkv[0], qkv[1]
        weight = torch.cat([q, k, weight], dim=0)
    module.weight.data = weight.to(device=device, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
