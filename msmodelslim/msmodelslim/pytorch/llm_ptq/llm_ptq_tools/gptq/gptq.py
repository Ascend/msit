# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
import torch

from .quant import quantize, asym_quant, asym_dequant


def _cholesky_try(hessian, upper=False, inverse=False):
    original_error = None
    try:
        if inverse:
            res = torch.cholesky_inverse(hessian, upper=upper)
        else:
            res = torch.linalg.cholesky(hessian, upper=upper)
    except RuntimeError as e:
        indef = True
        original_error = str(e)
    else:
        if torch.sum(torch.isnan(res)):
            indef = True
            original_error = "Cholesky decomposition resulted in NaN values"
        else:
            indef = False
    return res, indef, original_error


def cholesky_q(diag, percdamp, hessian, upper=False, try_count=10):
    res, indef, original_error = _cholesky_try(hessian, upper)
    
    while indef and try_count > 0:
        damp = percdamp * torch.mean(abs(torch.diag(hessian)))
        hessian[diag, diag] += damp
        try_count -= 1
        res, indef, original_error = _cholesky_try(hessian, upper)
    
    if indef:
        raise RuntimeError(f"Cholesky decomposition failed after multiple attempts. "
                          f"Original error: {original_error}. Try increasing the percdamp parameter.")
    
    return res


def cholesky_inverse_q(diag, percdamp, hessian, try_count=10):
    res, indef, original_error = _cholesky_try(hessian, upper=False, inverse=True)
            
    while indef and try_count > 0:
        damp = percdamp * torch.mean(abs(torch.diag(hessian)))
        hessian[diag, diag] += damp
        try_count -= 1
        res, indef, original_error = _cholesky_try(hessian, upper=False, inverse=True)
    
    if indef:
        raise RuntimeError(f"Cholesky inverse failed after multiple attempts. "
                         f"Original error: {original_error}. Try increasing the percdamp parameter.")
    
    return res


def post_opt_scale(w, w_quant, zero, hessian=None):
    temp = w_quant - zero
    if hessian is not None:
        hessian_qp = torch.matmul(temp, hessian)
        b_qp = torch.matmul(w, hessian)
    else:
        hessian_qp = temp
        b_qp = w
    hessian_qp = torch.sum(hessian_qp * temp, 1, keepdim=True)
    b_qp = torch.sum(b_qp * temp, 1, keepdim=True)

    scale_opt = b_qp / hessian_qp
    return scale_opt


class GPTQConfig:
    def __init__(self, 
                 blocksize=128, 
                 percdamp=.01, 
                 groupsize=-1, 
                 actorder=False,
                 post_opt=True, 
                 scale_refine=True):
        """
        Configuration class for GPTQ quantization.
        
        Args:
            blocksize (int): Size of blocks for quantization
            percdamp (float): Percentage of damping to apply
            groupsize (int): Size of groups for quantization, -1 means no grouping
            actorder (bool): Whether to use activation order
            post_opt (bool): Whether to use post optimization
            scale_refine (bool): Whether to refine scales
        """
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.post_opt = post_opt
        self.scale_refine = scale_refine


class GPTQ:
    def __init__(self, config=None, **kwargs):
        """
        GPTQ quantization implementation.
        """
        if config is None:
            self.config = GPTQConfig(**kwargs)
        else:
            self.config = config
            
        self.nsamples = 0
        self.has_calib = False
        self.quantizer = None
        self.columns = None
        self.hessian = None

    @torch.no_grad()
    def add_batch(self, inp):
        if self.columns is None:
            self.columns = inp.shape[-1]
            self.hessian = torch.zeros((self.columns, self.columns), device=inp.device)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.float()
        tmp = inp.shape[0]
        self.hessian *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hessian += inp.t().matmul(inp)
        self.has_calib = True

    @torch.no_grad()
    def fasterquant(self, layer, hessian_mse=True):
        blocksize = self.config.blocksize
        percdamp = self.config.percdamp
        groupsize = self.config.groupsize
        actorder = self.config.actorder
        post_opt = self.config.post_opt
        scale_refine = self.config.scale_refine
        
        dev = layer.weight.data.device
        if groupsize != -1:
            actorder = False
        dtype = layer.weight.data.dtype
        weight = layer.weight.data.clone()
        
        if len(weight.shape) == 3:
            weight = weight.reshape((-1, weight.shape[-1]))
        weight = weight.float()
        hessian = self.hessian
        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(self.columns, device=dev)
        hessian[diag, diag] += damp

        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        if hessian_mse:
            self.quantizer.find_params(weight, hessian=hessian)
        else:
            self.quantizer.find_params(weight)

        if scale_refine:
            weight_quant, _, _ = asym_quant(weight, self.quantizer.scale, 
                                        self.quantizer.zero, self.quantizer.minq, self.quantizer.maxq)
            scale_temp = post_opt_scale(weight, weight_quant, self.quantizer.zero, hessian=hessian)
            self.quantizer.scale = scale_temp
        
        if actorder:
            perm = torch.argsort(torch.diag(hessian), descending=True)
            weight = weight[:, perm]
            hessian = hessian[perm][:, perm]
        
        if 'npu' in dev.type:
            hessian_cpu = hessian.to('cpu')
            hessian_inv = cholesky_q(diag, percdamp, hessian_cpu)
            hessian_inv = cholesky_inverse_q(diag, percdamp, hessian_inv)
            hessian_inv = cholesky_q(diag, percdamp, hessian_inv, upper=True)
            hessian_inv = hessian_inv.to(dev)
        else:
            hessian_inv = cholesky_q(diag, percdamp, hessian)
            hessian_inv = cholesky_inverse_q(diag, percdamp, hessian_inv)
            hessian_inv = cholesky_q(diag, percdamp, hessian_inv, upper=True)

        quant_weight = torch.zeros_like(weight)

        now_idx = 1
        scale = []
        zero = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            w_tmp = weight[:, i1:i2].clone()
            quant_weight1 = torch.zeros_like(w_tmp)
            error = torch.zeros_like(w_tmp)
            hessian_inv1 = hessian_inv[i1:i2, i1:i2].to(dev)
            hessian_inv2 = hessian_inv[i1:i2, i2:].to(dev)

            for i in range(count):
                w = w_tmp[:, i]
                d = hessian_inv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(weight[:, (i1 + i):(i1 + i + groupsize)], hessian=hessian)
                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                q = quantize(w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero,
                                self.quantizer.minq, self.quantizer.maxq).flatten()
                
                quant_weight1[:, i] = q
                err1 = (w - q) / d
                w_tmp[:, i:] -= err1.unsqueeze(1).matmul(hessian_inv1[i, i:].unsqueeze(0))
                error[:, i] = err1
            quant_weight[:, i1:i2] = quant_weight1
            weight[:, i2:] -= error.matmul(hessian_inv2)

        if post_opt:
            w_quant, _, _ = asym_quant(quant_weight, self.quantizer.scale, self.quantizer.zero, \
                                           self.quantizer.minq, self.quantizer.maxq)

            weight = layer.weight.data.clone()
            if len(weight.shape) == 3:
                weight = weight.reshape((-1, weight.shape[-1]))
            weight = weight.float()
            if actorder:
                weight = weight[:, perm]
            self.quantizer.scale = post_opt_scale(weight, w_quant, self.quantizer.zero, hessian=hessian)

            quant_weight = asym_dequant(w_quant, self.quantizer.scale, self.quantizer.zero)
        
        if actorder:
            invperm = torch.argsort(perm)

            quant_weight = quant_weight[:, invperm]
        layer.weight.data = quant_weight.reshape(layer.weight.shape).to(dtype)
        del weight, quant_weight, w_tmp, quant_weight1, error, hessian_inv1, hessian_inv2, w, d, q, err1
        if len(scale) == 0:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero

