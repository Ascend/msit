# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import functools
import math

import scipy.linalg
import torch

from ascend_utils.common.security import check_type

EPSILON = 1e-8


class DuQuantConfig:

    def __init__(self, alpha=0.5,
                 total_rotate_step=2,
                 split=True,
                 split_size=128,
                 greedy_search_steps=8,
                 using_hadamard=True,
                 disable_delta1=False,
                 disable_delta2=True,
                 enable=True):
        check_type(alpha, float, param_name='alpha',
                   additional_check_func=lambda x: 0 <= x <= 1,
                   additional_msg='0<=alpha<=1')
        check_type(total_rotate_step, int, param_name='total_rotate_step')
        check_type(split, bool, param_name='split')
        check_type(split_size, int, param_name='split_size',
                   additional_check_func=lambda x: (x & (x - 1)) == 0 and x != 0,
                   additional_msg='split_size=2^n')
        check_type(greedy_search_steps, int, param_name='greedy_search_steps')
        check_type(using_hadamard, bool, param_name='using_hadamard')
        check_type(disable_delta1, bool, param_name='disable_delta1')
        check_type(disable_delta2, bool, param_name='disable_delta2')
        check_type(enable, bool, param_name='enable')

        self.alpha = alpha
        self.total_rotate_step = total_rotate_step
        self.split = split
        self.split_size = split_size
        self.greedy_search_steps = greedy_search_steps
        self.using_hadamard = using_hadamard
        self.disable_delta1 = disable_delta1
        self.disable_delta2 = disable_delta2
        self.enable = enable

    def __str__(self):
        return f"DuQuantConfig(alpha={self.alpha} \total_rotate_step={self.total_rotate_step} \
                split={self.split} split_size={self.split_size} \
                greedy_search_steps={self.greedy_search_steps} using_hadamard={self.using_hadamard})"


def construct_orthogonal_rotation_matrix(n):
    """
    Generate an orthogonal rotation matrix of size n x n using QR decomposition.

    Parameters:
    n (int): The size of the rotation matrix.

    Returns:
    np.ndarray: An orthogonal rotation matrix of size n x n.
    """
    # Generate a random matrix
    random_matrix = torch.randn(n, n)

    # Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix)

    # Ensure the orthogonal matrix has a positive determinant
    d = torch.diag(r)
    q *= torch.sign(d)

    return q


def construct_orthogonal_rotation_matrix_with_uniform_first_row(n):
    """
    Generate an orthogonal rotation matrix of size n x n with the first row uniformly distributed.

    Parameters:
    n (int): The size of the rotation matrix.

    Returns:
    np.ndarray: An orthogonal rotation matrix of size n x n.
    """
    # Step 1: Generate a uniformly distributed vector
    first_row = 1 / torch.tensor(n)

    # Step 2: Normalize this vector to make it a unit vector
    first_row /= torch.linalg.norm(first_row)

    # Step 3: Create a random matrix
    random_matrix = torch.randn(n, n)

    # Replace the first row with the uniformly distributed unit vector
    random_matrix[0, :] = first_row

    # Step 4: Perform QR decomposition to ensure orthogonality
    q, r = torch.linalg.qr(random_matrix)

    # Step 5: Ensure the orthogonal matrix has a positive determinant
    # Adjust signs if necessary to ensure the matrix is a rotation matrix
    if torch.linalg.det(q.to(dtype=torch.float32)) < 0:
        q[:, 0] = -q[:, 0]

    return q


class DuQuant:

    def __init__(self, config: DuQuantConfig):

        self.config = config

    @classmethod
    @torch.no_grad()
    @functools.lru_cache(maxsize=10)
    def get_zigzag_index_from_cache(cls, feature_size, split_size):

        index = torch.zeros([feature_size], dtype=torch.int)

        block_count = math.ceil(feature_size / split_size)
        round_count = min(feature_size, split_size)

        for i in range(0, round_count):
            if i % 2 == 0:
                for j in range(0, block_count):
                    index[j * split_size + i] = i * block_count + j
            else:
                for j in range(0, block_count):
                    index[(block_count - j - 1) * split_size + i] = i * block_count + j

        inverse_index = torch.argsort(index)

        return index, inverse_index

    @torch.no_grad()
    def construct_delta_vector(self, x, w, delta_min, delta_max):
        w_scale = w.abs().max(dim=1).values.pow(1 - self.config.alpha)
        x_scale = x.abs().view(-1, x.shape[-1]).max(dim=0).values.pow(self.config.alpha).reciprocal()
        return (w_scale * x_scale).clamp(min=delta_min, max=delta_max)

    @torch.no_grad()
    def construct_rotations_and_permutations(self, x):

        feature_size = x.shape[-1]
        x_ = x.clone().detach().view(-1, feature_size)
        x_shape = x_.shape

        rotations = None
        permutations = None

        for i in range(0, self.config.total_rotate_step):

            ri = self.construct_r_blocks(x_)

            if rotations is None:
                rotations = ri.unsqueeze(0)
            else:
                rotations = torch.cat((rotations, ri.unsqueeze(0)), dim=0)

            x_ = x_.reshape(-1, feature_size // self.config.split_size, self.config.split_size).transpose(0, 1)
            x_ = (x_ @ ri).transpose(0, 1).reshape(-1, feature_size)

            if i == self.config.total_rotate_step - 1:
                pass
            else:
                pi = self.construct_zigzag_and_sorted_index(x_)

                if permutations is None:
                    permutations = pi.unsqueeze(0)
                else:
                    permutations = torch.cat((permutations, pi.unsqueeze(0)), dim=0)

                zigzag_index, sorted_index = pi[0], pi[1]
                x_ = x_[:, sorted_index[zigzag_index]]

        return rotations, permutations

    @torch.no_grad()
    def construct_rn(self, x):

        feature_size = x.shape[-1]

        max_each_feature = torch.max(torch.abs(x.view(-1, feature_size)), dim=0).values
        d = torch.argmax(max_each_feature)
        e_d = torch.eye(feature_size).to(device=x.device, dtype=x.dtype)
        e_d[:, [0, d]] = e_d[:, [d, 0]]

        if self.config.using_hadamard:
            r_wavy_head = torch.Tensor(scipy.linalg.hadamard(feature_size)).to(device=x.device,
                                                                               dtype=x.dtype) / math.sqrt(feature_size)
        else:
            r_wavy_head = construct_orthogonal_rotation_matrix_with_uniform_first_row(feature_size)

        q = torch.block_diag(torch.eye(1), construct_orthogonal_rotation_matrix(feature_size - 1)).to(device=x.device,
                                                                                                      dtype=x.dtype)

        return e_d @ r_wavy_head.t() @ q @ e_d

    @torch.no_grad()
    def construct_r_blocks(self, x):

        # apply a block-wise(by last dim) compute if needed

        if self.config.split and self.config.split_size > 0:

            block_count = x.shape[-1] // self.config.split_size

            r_block = None

            for x_block_i in torch.chunk(x, chunks=block_count, dim=-1):

                r_block_i = self.construct_r_block(x_block_i).to(dtype=x.dtype)

                if r_block is None:
                    r_block = r_block_i.unsqueeze(0)
                else:
                    r_block = torch.cat((r_block, r_block_i.unsqueeze(0)), dim=0)

            return r_block
        else:
            return self.construct_r_block(x).unsqueeze(0)

    @torch.no_grad()
    def construct_r_block(self, x):

        # in each block, we do a greedy search for N step to find the n that make max|x·r1···rn| smallest

        feature_size = x.shape[-1]

        x_r1_to_ri = x.clone().detach().view(-1, feature_size).to(device=x.device, dtype=x.dtype)
        x_rn_abs_max = x_r1_to_ri.abs().max().item()
        r1_to_ri = torch.eye(feature_size).to(device=x.device, dtype=x.dtype)
        target = r1_to_ri

        for _ in range(0, self.config.greedy_search_steps):

            ri = self.construct_rn(x_r1_to_ri)

            x_r1_to_ri = x_r1_to_ri @ ri
            r1_to_ri = r1_to_ri @ ri

            x_ri_abs_max = x_r1_to_ri.abs().max().item()

            if x_ri_abs_max < x_rn_abs_max:
                x_rn_abs_max = x_ri_abs_max
                target = r1_to_ri

        return target.to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def construct_zigzag_and_sorted_index(self, x):
        feature_size = x.shape[-1]

        sorted_index = x.abs().view(-1, x.shape[-1]).max(dim=0).values.sort(descending=True).indices
        sorted_inverse_index = torch.argsort(sorted_index)
        zigzag_index, zigzag_inverse_index = self.get_zigzag_index_from_cache(feature_size, self.config.split_size)

        return torch.cat((zigzag_index.to(device=x.device).unsqueeze(0),
                          sorted_index.to(device=x.device).unsqueeze(0),
                          zigzag_inverse_index.to(device=x.device).unsqueeze(0),
                          sorted_inverse_index.to(device=x.device).unsqueeze(0)), dim=0)
