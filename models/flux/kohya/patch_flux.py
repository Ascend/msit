# copy from FLUX repo: https://github.com/black-forest-labs/flux
# license: Apache-2.0 License


import torch
import torch_npu
import einops
from torch import Tensor
import library

library.flux_models.rope = torch.no_grad(library.flux_models.rope)
rope = library.flux_models.rope


def apply_npu_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    cosine, sine = freqs_cis
    xq_out = torch_npu.npu_rotary_mul(xq.float(), cosine.contiguous(), sine.contiguous(), 'interleave')
    xk_out = torch_npu.npu_rotary_mul(xk.float(), cosine.contiguous(), sine.contiguous(), 'interleave')
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xq.shape).type_as(xk)


def embed_nd_forward(self, ids: Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = torch.cat(
        [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
        dim=-3,
    )
    emb = emb.unsqueeze(1)
    cosine, sine = emb[0, ..., 0, 0], emb[0, ..., 1, 0]
    # Repeat data along the last dim for interleaving rotary mul
    cosine = einops.repeat(cosine, '... d -> ...(d 2)').unsqueeze(0)
    sine = einops.repeat(sine, '... d -> ...(d 2)').unsqueeze(0)
    return torch.stack((cosine, sine))


def rms_norm_forward(self, x: Tensor):
    if self.scale is not None:
        if self.scale.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.scale.dtype)
            x = torch_npu.npu_rms_norm(x, self.scale)[0]
    return x


library.flux_models.apply_rope = apply_npu_rope
library.flux_models.EmbedND.forward = embed_nd_forward
library.flux_models.RMSNorm.forward = rms_norm_forward