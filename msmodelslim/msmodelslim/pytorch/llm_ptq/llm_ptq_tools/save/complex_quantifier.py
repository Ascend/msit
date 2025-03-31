# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import numpy as np

# KIA part
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers
from msmodelslim.pytorch.lowbit.calibration import LlamaRMSNormBias
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import fake_quantize_save

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_config_manager import LayerConfigManager
from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import NormBias
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer, LinearNf4Quantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    export_fa_quant_params,
    is_attn_module_and_then_check_quantizer
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol


def deqscale_process(input_scale, scale):
    deq_scale = input_scale * scale
    if deq_scale.ndim > 1:
        deq_scale = deq_scale.squeeze(1)
    deq_scale = deq_scale.cpu()
    return deq_scale


def change_bias(fp_weight, module):
    if module.bias is None:
        bias_shape = fp_weight.shape[0]
        fp_bias = torch.zeros(bias_shape)
    else:
        fp_bias = module.bias.cpu()
    return fp_bias


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def deqscale2int64_by_dtype(scale, is_bf16):
    if is_bf16:
        return scale
    else:
        return deqscale2int64(scale)


def _get_module_quant_input(module):
    fp_weight = module.weight.cpu()
    weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
    device = weight_scale.device if weight_scale is not None else None
    scale = weight_scale.cpu() if weight_scale is not None else None
    offset = weight_offset.cpu() if weight_offset is not None else None
    round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt

    if hasattr(module.quant_weight, 'weight_scale_second') and \
       hasattr(module.quant_weight, 'weight_offset_second'):
        weight_scale_second = module.quant_weight.weight_scale_second
        weight_offset_second = module.quant_weight.weight_offset_second
        scale_second = weight_scale_second.cpu() if weight_scale_second is not None else None
        offset_second = weight_offset_second.cpu() if weight_offset_second is not None else None
        if scale_second is not None and offset_second is not None:
            scale = [scale, scale_second]
            offset = [offset, offset_second]

    ret = fp_weight, device, scale, offset, round_opt
    return ret


# for clean code
def generate_weight_of_rms_norm_module(name, module):
    yield name + '.weight', QuantType.FLOAT, module.weight.clone().cpu()


# for clean code
def generate_weight_of_fa_module(name, module):
    quant_param_scale, quant_param_offset, _ = export_fa_quant_params(module, name)
    for name, param in quant_param_scale.items():
        yield name, QuantType.FAQuant, param
    for name, param in quant_param_offset.items():
        yield name, QuantType.FAQuant, param


# for clean code
def generate_weight_of_nf_module(name, module):
    yield name + '.weight', QuantType.NF4, module.weight
    if module.bias is not None:
        yield name + '.bias', QuantType.NF4, module.bias


class ComplexQuantifier:
    def __init__(self, cfg, rollback_names, torch_dtype, layer_cfg_manager):
        self.cfg = cfg
        self.rollback_names = rollback_names
        self.torch_dtype = torch_dtype
        self.layer_cfg_manager: LayerConfigManager = layer_cfg_manager

        self.skip_module_names = set()

    def generate_weight_of_module(self, name: str, module: torch.nn.Module):
        if name in self.skip_module_names:
            return

        model_quant_type = self.layer_cfg_manager.get_layer_config(name).model_quant_type
        if self.cfg.use_fa_quant and is_attn_module_and_then_check_quantizer(module, name):
            yield from generate_weight_of_fa_module(name, module)
        elif isinstance(module, LinearNf4Quantizer):
            yield from generate_weight_of_nf_module(name, module)
        elif isinstance(module, ParallelLinearCol):
            yield from self.generate_weight_of_tp_module(name, module, model_quant_type)
        # 处理 Norm 对应的 weight、bias
        elif isinstance(module, NormBias):
            yield from self.generate_weight_of_norm_module(name, module, model_quant_type)
        elif isinstance(module, LlamaRMSNormBias):
            yield from generate_weight_of_rms_norm_module(name, module)
        # 处理Linear、以及附属scale、offset等params
        elif isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
            yield from self.generate_weight_of_linear_module(name, module, model_quant_type)
        else:
            for key, param in module.named_parameters(name, recurse=False):
                yield key, QuantType.FLOAT, param

    def generate_weight_of_tp_module(self, name, module, model_quant_type):
        quant_param, _ = module.get_quant_param()
        for mod_name, mod in module.named_modules(prefix=name):
            if mod_name == name:
                continue

            for param_name, _, param in self.generate_weight_of_module(mod_name, mod):
                quant_param[param_name] = param
            self.skip_module_names.add(mod_name)

        if hasattr(self.cfg, 'tp_size'):
            self.concat_simulate_linear(name, module, quant_param)

        for name, param in quant_param.items():
            yield name, model_quant_type, param

    def generate_weight_of_norm_module(self, name, module, model_quant_type):
        # 不暴露原norm
        self.skip_module_names.add(name + '.module')

        anti_norm_weight: torch.Tensor = module.module.weight
        anti_norm_bias: torch.Tensor = module.bias
        yield name + '.weight', model_quant_type, anti_norm_weight.clone().cpu()
        yield name + '.bias', model_quant_type, anti_norm_bias.clone().cpu()

    def generate_weight_of_linear_module(self, name, module, model_quant_type):
        if not module.quant_weight.is_enable:
            return

        quant_weight, fp_weight, weight_scale, weight_offset = self.get_param_from_quantizer(module)
        if quant_weight is None:
            return

        # 各种量化均需要提供 weight
        quant_weight: torch.Tensor = quant_weight.to(device=fp_weight.device)
        save_quant_weight = quant_weight.cpu().to(torch.int8)
        yield name + '.weight', model_quant_type, save_quant_weight
        if hasattr(module, 'bias') and module.bias is not None:
            yield name + '.bias', QuantType.FLOAT, module.bias

        # W4A8_DYNAMIC 有两种量化模式，一种分阶段量化（将浮点先量化成int8，然后再量化成 int4），一种直接量化（将浮点直接量化成 int4）
        if model_quant_type in [QuantType.W4A8_DYNAMIC]:
            is_scale_list = isinstance(weight_scale, list) and len(weight_scale) == 2
            is_offset_list = isinstance(weight_offset, list) and len(weight_offset) == 2
            if is_scale_list and is_offset_list:
                yield name + '.weight_scale', model_quant_type, weight_scale[0].cpu()
                yield name + '.weight_offset', model_quant_type, weight_offset[0].cpu()
                ori_shape = save_quant_weight.shape
                weight_scale[1] = weight_scale[1].reshape(ori_shape[0], -1)
                weight_offset[1] = weight_offset[1].reshape(ori_shape[0], -1)
                yield name + '.weight_scale_second', model_quant_type, weight_scale[1].cpu()
                yield name + '.weight_offset_second', model_quant_type, weight_offset[1].cpu()
            else:
                yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
                yield name + '.weight_offset', model_quant_type, weight_offset.cpu()

        # W4A16/W8A16 需要提供 weight_scale、weight_offset
        if model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC, QuantType.W8A8]:
            yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
            yield name + '.weight_offset', model_quant_type, weight_offset.cpu()

        # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
        if model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            input_scale = module.quant_input.input_scale
            input_offset = module.quant_input.input_offset
            yield name + '.input_scale', model_quant_type, input_scale.cpu()
            yield name + '.input_offset', model_quant_type, input_offset.cpu()

            input_scale = input_scale.to(device=quant_weight.device)
            input_offset = input_offset.to(device=quant_weight.device)
            deq_scale = deqscale_process(input_scale, weight_scale).to(torch.float32)
            quant_weight = quant_weight.to(torch.float32)
            input_offset = input_offset.to(torch.float32)
            correction = (quant_weight.sum(dim=1) * input_offset).cpu()
            fp_bias = change_bias(fp_weight, module)
            quant_bias = torch.round(fp_bias / deq_scale - correction)
            deq_scale = deqscale2int64_by_dtype(deq_scale,
                                                self.torch_dtype == torch.bfloat16)

            yield name + '.quant_bias', model_quant_type, quant_bias.cpu().to(torch.int32)
            yield name + '.deq_scale', model_quant_type, deq_scale.cpu()

    def concat_simulate_linear(self, name, module, quant_param):
        if name in self.rollback_names:
            return
        if module.cfg.model_quant_type == QuantType.FLOAT:
            return
        concat_weight_list = []
        for tp_index in range(module.cfg.tp_size):
            concat_name = '.'.join([name, f'tp_list', str(tp_index), 'weight'])
            concat_weight_list.append(quant_param.get(concat_name))
        concat_weight = torch.cat(concat_weight_list, dim=-1)
        quant_param[name + '.weight'] = concat_weight

    @torch.no_grad()
    def get_param_from_quantizer(self, module):
        quant_weight = None
        fp_weight, device, weight_scale, weight_offset, round_opt = _get_module_quant_input(module)
        if isinstance(module, LinearQuantizer):
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=8,
                                                 round_opt=round_opt, device=device)
        if isinstance(module, LinearSparseQuantizer):
            _, _, quant_weight, _ = quant_one_weight_by_outliers(
                fp_weight, powerquant=self.cfg.nonuniform, fraction=self.cfg.fraction, num_bits=self.cfg.w_bit,
                per_channel=not self.cfg.mm_tensor)
        if isinstance(module, LowBitLinearQuantizer):
            if not module.cfg.is_stage_quant:
                fp_weight = module.fp_weight
            if module.disable_input:
                res = None, fp_weight, weight_scale, weight_offset
                return res
            if module.cfg.model_quant_type == QuantType.W8A8S:
                bit = 8
            else:
                bit = module.cfg.w_bit
            
            is_scale_list = isinstance(weight_scale, list) and len(weight_scale) == 2
            is_offset_list = isinstance(weight_offset, list) and len(weight_offset) == 2
            if is_scale_list and is_offset_list:
                # w4a8 分阶段量化，因此需要两次fake_quantize_save得到量化后的权重
                first_quant_weight, _ = fake_quantize_save(fp_weight, weight_scale[0], weight_offset[0], bit=8,
                                                          round_opt=round_opt, device=module.weight.device)
                quant_weight, _ = fake_quantize_save(first_quant_weight, weight_scale[1], weight_offset[1], bit=4,
                                                            round_opt=round_opt, device=module.weight.device,
                                                            group_size=module.cfg.group_size)
            else:
                quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=bit,
                                                 round_opt=round_opt, device=module.weight.device,
                                                 group_size=module.cfg.group_size)
        res = quant_weight, fp_weight, weight_scale, weight_offset
        return res
