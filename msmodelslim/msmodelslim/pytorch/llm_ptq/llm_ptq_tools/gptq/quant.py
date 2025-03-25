# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from functools import partial
import torch
import torch.nn as nn


@torch.no_grad()
def asym_quant(x, scale, zero, minq, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return q, scale, zero


@torch.no_grad()
def asym_dequant(q, scale, zero):
    return scale * (q - zero)


@torch.no_grad()
def asym_quant_dequant(x, scale, zero, minq, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, minq, maxq))


@torch.no_grad()
def get_scale_zero(xmin, xmax, bits, sym=False, q_signed=True):
    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    maxq = 2**bits - 1
    scale = (xmax - xmin) / maxq
    if sym:
        if q_signed:
            zero = torch.full_like(scale, 0)
        else:
            zero = torch.full_like(scale, maxq)
    else:
        zero = torch.round(-xmin / scale)
        if q_signed:
            qmin = - (2 ** (bits - 1))
            zero += qmin
    return scale, zero


def quantize(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)


def mult_loss(x, quant_x, hessian):
    error = x - quant_x
    error = torch.sum(torch.mul(error, torch.matmul(error, hessian)), 1)
    return error


def norm_loss(x, quant_x, norm):
    q = quant_x - x
    q.abs_()
    q.pow_(norm)
    error = torch.sum(q, 1)
    return error


def get_minmax(x):
    dev = x.device
    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)
    return xmin, xmax


class BaseParamsFinder:
    def __init__(self, bits, sym=True, mse=True, q_signed=True):
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.q_signed = q_signed
        
        if q_signed:
            self.minq = -2**(bits - 1)
            self.maxq = 2**(bits - 1) - 1
        else:
            self.minq = 0
            self.maxq = 2**bits - 1
        self.grid = 100
        self.maxshrink = 0.8

    def _optimize_params(self, x, xmin, xmax, opt_func):
        dev = x.device
        shape = x.shape
        grid = self.grid
        maxshrink = self.maxshrink
        scale, zero = get_scale_zero(xmin, xmax, self.bits, self.sym, self.q_signed)
        
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1, zero1 = get_scale_zero(xmin1, xmax1, self.bits, self.sym, self.q_signed)
                q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.minq, self.maxq)
                error = opt_func(x, q)
                tmp = error < best
                if torch.any(tmp):
                    best[tmp] = error[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]

        scale = scale.reshape(shape[0], -1)
        zero = zero.reshape(shape[0], -1)
        
        return scale, zero


class HessianParamsFinder(BaseParamsFinder):
    def __init__(self, bits, sym=True, mse=False, q_signed=True):
        super().__init__(bits, sym, mse, q_signed)
    
    @torch.no_grad()
    def find_params(self, x, hessian):
        xmin, xmax = get_minmax(x)
        opt_func = partial(mult_loss, hessian=hessian)
        return self._optimize_params(x, xmin, xmax, opt_func)


class NormParamsFinder(BaseParamsFinder):
    def __init__(self, bits, sym=True, mse=False, q_signed=True):
        super().__init__(bits, sym, mse, q_signed)
    
    @torch.no_grad()
    def find_params(self, x, norm=2.4):
        xmin, xmax = get_minmax(x)
        opt_func = partial(norm_loss, norm=norm)
        return self._optimize_params(x, xmin, xmax, opt_func)


class GPTQQuantizer(nn.Module):
    def __init__(
                self,
                bits, 
                perchannel=False, 
                sym=True,
                mse=True, 
                norm=2.4, 
                q_signed=True
                ):
        super(GPTQQuantizer, self).__init__()

        self.bits = bits 
        self.scale = None
        self.zero = None
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.q_signed = q_signed
        if q_signed:
            self.minq = -2**(bits - 1)
            self.maxq = 2**(bits - 1) - 1
        else:
            self.minq = 0
            self.maxq = 2**bits - 1

    def find_params(self, x, hessian=None):
        if hessian is not None:
            finder = HessianParamsFinder(self.bits, self.sym, self.mse, self.q_signed)
            self.scale, self.zero = finder.find_params(x, hessian)
        else:
            finder = NormParamsFinder(self.bits, self.sym, self.mse, self.q_signed)
            self.scale, self.zero = finder.find_params(x, self.norm)

