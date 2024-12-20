import math
import typing

import tqdm
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quarot import model_utils
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quarot import utils
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quarot.hadamard_utils import (
    random_hadamard_matrix, apply_exact_had_to_linear, is_pow2, get_hadk, matmul_had, hadamard_transform
)
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import NormBias, PatternProcess
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter


class OnlineRotationWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Linear):
        super(OnlineRotationWrapper, self).__init__()
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{module} is not a torch.nn.Linear module!")
        self.module = module
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in fp32
                x = matmul_had(x.float()).to(x_dtype)
            else:  # Full Hadamard in fp16/bf16
                x = matmul_had(x)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = hadamard_transform(
                    x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                self.had_K = self.had_K.to(x.device)
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(
                    init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        x = self.module(x).to(x_dtype)

        return x


def quarot_rotate_weights(model: torch.nn.Module, dag_adapter: TorchDAGAdapter, fp32_had=True) -> None:
    norm_linear_subgraph = dag_adapter.get_norm_linear_subgraph()
    norm_linear_mapping = [
        (
            PatternProcess.get_module_by_name(model, norm),
            [PatternProcess.get_module_by_name(model, linear) for linear in norm_linear_subgraph[norm]]
        ) for norm in norm_linear_subgraph
    ]
    pre_head_norm, lm_head = dag_adapter.get_pre_head_norm_and_head_pair()
    norm_linear_mapping.append(
        (
            PatternProcess.get_module_by_name(model, pre_head_norm),
            [PatternProcess.get_module_by_name(model, lm_head)]
        )
    )

    embeddings = model_utils.get_embeddings(model)
    norm_class = list(set([m.__class__ for m in model.modules() if "norm" in m.__class__.__name__.lower()]))[0]

    fuse_layer_norms(model, embeddings, norm_linear_mapping, norm_class)
    rotate_model(model, dag_adapter)

    # Add Online Rotation Wrapper to the model
    attention_output_projs = dag_adapter.get_attn_or_mlp_linear_layers(
            require_attn_over_mlp=True,
            require_input_over_output=False
    )
    mlp_output_projs = dag_adapter.get_attn_or_mlp_linear_layers(
        require_attn_over_mlp=False,
        require_input_over_output=False
    )
    attention_output_projs = [
        (linear, PatternProcess.get_module_by_name(model, linear))
        for layer in attention_output_projs for linear in layer
    ]
    mlp_output_projs = [
        (linear, PatternProcess.get_module_by_name(model, linear))
        for layer in mlp_output_projs for linear in layer
    ]

    wrap_online_rotation_and_configure(attention_output_projs, mlp_output_projs, fp32_had, model)


def wrap_online_rotation_and_configure(attention_output_projs, mlp_output_projs, fp32_had, model):
    for name, module in attention_output_projs + mlp_output_projs:
        wrapped = OnlineRotationWrapper(module)
        is_output_layer = name in [n for n, _ in attention_output_projs]
        if is_output_layer:
            had_K, K = get_hadk(model.config.num_attention_heads)
            wrapped.online_partial_had = True
        else:
            had_K, K = get_hadk(model.config.intermediate_size)
            wrapped.online_full_had = True
        wrapped.had_K = had_K
        wrapped.K = K
        wrapped.had_dim = model.config.hidden_size // model.config.num_attention_heads
        wrapped.fp32_had = fp32_had

        levels = name.split(".")
        preceding_levels, lowest_level = levels[:-1], levels[-1]

        cur_mod = model
        for s in preceding_levels:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, lowest_level, wrapped)


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        linear_device = linear.weight.device

        # Calculating new weight and bias
        W_ = linear.weight.data.to(device=utils.DEV, dtype=torch.float32)
        if isinstance(layernorm, NormBias):
            linear.weight.data = (W_ * layernorm.module.weight.to(device=utils.DEV, dtype=torch.float32))
        else:
            linear.weight.data = (W_ * layernorm.weight.to(device=utils.DEV, dtype=torch.float32))


        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
            linear.bias.data = (linear.bias.data.to(device=utils.DEV, dtype=torch.float32)
                                + torch.matmul(W_, layernorm.bias.to(device=utils.DEV, dtype=torch.float32)))
            linear.bias.data = linear.bias.data.to(device=linear_device, dtype=linear_dtype)
        
        linear.weight.data = linear.weight.data.to(device=linear_device, dtype=linear_dtype)
        del W_
        utils.cleanup_memory(verbose=False)


def fuse_layer_norms(
        model: torch.nn.Module,
        embeddings: typing.List[torch.nn.Module],
        norm_linear_mapping: typing.Iterable,
        norm_type: typing.Type[torch.nn.Module]
):
    # Embedding fusion
    for W in embeddings:
        dtype = W.weight.dtype
        device = W.weight.device
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(device=device, dtype=dtype)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for norm_module, linear_modules in norm_linear_mapping:
        fuse_ln_linear(norm_module, linear_modules)

    # After fusing the weights into linear layers, replace the norm class with customized norm class
    model_utils.replace_modules(
        model,
        norm_type,
        lambda module: model_utils.RMSN(model.config.hidden_size, module.weight.device, dtype=model.dtype),
        replace_layers=False,
    )
    

def random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    utils.cleanup_memory(verbose=False)
    random_matrix = torch.randn(size, size, dtype=torch.float32).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size: int, mode: str, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model: torch.nn.Module, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        dtype = W.weight.data.dtype
        device = W.weight.data.device
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
        W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)

        del W_


def rotate_attention_or_mlp_inputs(Ws: typing.List[torch.nn.Linear], Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer, and MLP layer input weight matrices
    for W in Ws:
        dtype = W.weight.data.dtype
        device = W.weight.data.device
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).to(device)

        del W_


def rotate_attention_output(W: torch.nn.Linear, Q: torch.Tensor) -> None:
    # Rotate output matrices of the self-attention layers.
    dtype = W.weight.data.dtype
    device = W.weight.data.device
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device=device)
    if W.bias is not None:
        b = W.bias.to(device=utils.DEV, dtype=torch.float32)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype, device=device)

    del W_


def rotate_mlp_output(W: torch.nn.Linear, Q: torch.Tensor):
    # Rotate the MLP output weights and bias.
    dtype = W.weight.data.dtype
    device = W.weight.data.device
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype).to(device)
    apply_exact_had_to_linear(W, had_dim=-1, output=False)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float32)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype).to(device)

    del W_


def rotate_head(lm_head: torch.nn.Linear, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = lm_head
    dtype = W.weight.data.dtype
    device = W.weight.data.device
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float32)
    W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)

    del W_


def rotate_v_proj(W: torch.nn.Linear, head_dim: int, packed_weight = False):
    apply_exact_had_to_linear(W, had_dim=head_dim, output=True, packed_weight=packed_weight)


def rotate_o_proj(W: torch.nn.Linear, head_dim: int):
    apply_exact_had_to_linear(W, had_dim=-1, output=False, head_dim=head_dim)


# @torch.inference_mode()
def rotate_model(model, dag_adapter: TorchDAGAdapter):
    Q = get_orthogonal_matrix(model.config.hidden_size, "hadamard", utils.DEV)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q)

    _, lm_head = dag_adapter.get_pre_head_norm_and_head_pair()
    lm_head = PatternProcess.get_module_by_name(model, lm_head)
    rotate_head(lm_head, Q)

    attention_input_projs = [
        [PatternProcess.get_module_by_name(model, linear) for linear in layer]
        for layer in dag_adapter.get_attn_or_mlp_linear_layers(True, True)
    ]
    attention_output_projs = [
        PatternProcess.get_module_by_name(model, linear)
        for layer in dag_adapter.get_attn_or_mlp_linear_layers(True, False) for linear in layer
    ]
    mlp_input_projs = [
        [PatternProcess.get_module_by_name(model, linear) for linear in layer]
        for layer in dag_adapter.get_attn_or_mlp_linear_layers(False, True)
    ]
    mlp_output_projs = [
        PatternProcess.get_module_by_name(model, linear)
        for layer in dag_adapter.get_attn_or_mlp_linear_layers(False, False) for linear in layer
    ]

    for idx, _ in enumerate(tqdm.tqdm(model_utils.get_transformer_layers(model), unit="layer", desc="Rotating")):
        rotate_attention_or_mlp_inputs(attention_input_projs[idx], Q)
        rotate_attention_output(attention_output_projs[idx], Q)
        rotate_attention_or_mlp_inputs(mlp_input_projs[idx], Q)
        rotate_mlp_output(mlp_output_projs[idx], Q)
        rotate_o_proj(attention_output_projs[idx], head_dim)

        if len(attention_input_projs[idx]) == 1:
            # W_pack scenario
            packed_weight = True
        else:
            packed_weight = False
        v_proj = attention_input_projs[idx][-1]
        rotate_v_proj(v_proj, head_dim, packed_weight)

        utils.cleanup_memory(verbose=True)
