# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
from enum import Enum
from typing import Tuple
from collections import defaultdict

import torch
from transformers.configuration_utils import PretrainedConfig

from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import linear_quantization_params, fake_quantize


class TensorType(Enum):
    Q = "q"
    K = "k"
    V = "v"

    @classmethod
    def get_values(cls):
        return [t.value for t in cls]


class QKVQuantizer:
    def __init__(self):
        self.bit = None
        self.sym = None
        self.num_head = None
        self._min_values = None
        self._max_values = None
        self.scale = None
        self.offset = None
        self.is_record = False
        self.single_record = None

    def configure(self, bit: int, sym: bool, num_head: int):
        self.bit = bit
        self.sym = sym
        self.num_head = num_head

    def update(self, samples: torch.Tensor, tp_size: int):
        if self.is_record:
            self.single_record = samples
            self.is_record = False

        batch_size, num_head, seq_len, head_dim = samples.shape
        if num_head != self.num_head:
            raise ValueError("The number of heads in the input states tensor does not match the preset value!")

        num_head_per_device = self.num_head // tp_size
        samples = samples.contiguous().view(tp_size * num_head_per_device, -1)
        samples_max = samples.max(dim=-1, keepdim=True)[0]
        samples_min = samples.min(dim=-1, keepdim=True)[0]
        ## min value
        if self._min_values is None:
            self._min_values = samples_min
        else:
            index = samples_min < self._min_values
            self._min_values[index] = samples_min[index]
        ## max value
        if self._max_values is None:
            self._max_values = samples_max
        else:
            index = samples_max > self._max_values
            self._max_values[index] = samples_max[index]

    def get_scale_offset(
            self,
            states_tensor: torch.Tensor = None,
            tp_size: int = None,
            update: bool = False
    ):
        if update:
            self.update(states_tensor, tp_size)
            self.scale, self.offset = self._calculate_scale_offset()
        return self.scale, self.offset

    def is_calibrated(self):
        return self.scale is not None and self.offset is not None

    def reset_quant_params(self):
        self.scale, self.offset = None, None
    
    def enable_record(self):
        self.is_record = True
    
    def clear_record(self):
        self.single_record = None

    def _calculate_scale_offset(self, integral_zero_point: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._max_values.dtype
        scale, offset = linear_quantization_params(
            self.bit,
            self._min_values,
            self._max_values,
            sym=self.sym,
            integral_zero_point=integral_zero_point
        )
        if dtype == torch.bfloat16:
            scale = scale.to(dtype)
            offset = offset.to(dtype)
        return scale, offset


class FAQuantizer:
    def __init__(self, config, logger):
        check_type(config, PretrainedConfig, param_name="config")
        check_type(logger, logging.Logger, param_name="logger")

        self.tp_size = None
        self.num_head = None
        self.num_kv_head = None # GQA模型的KV所用到的注意力头数量，如果是MHA模型则与num_head相同
        self.head_dim = None
        self.is_calib = False
        self.dequant_infer = False
        self.logger = logger
        self.debug_mode = False

        self.q_observer = QKVQuantizer()
        self.k_observer = QKVQuantizer()
        self.v_observer = QKVQuantizer()

        self._set_head_params(config)

    def quant(self, states_tensor: torch.Tensor, qkv: TensorType):
        check_type(states_tensor, torch.Tensor, param_name="states_tensor")

        if not self.is_calib and not self.dequant_infer:
            return states_tensor

        if TensorType(qkv) == TensorType.Q:
            scale, offset = self.q_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.q_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.K:
            scale, offset = self.k_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.k_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.V:
            scale, offset = self.v_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.v_observer.num_head // self.tp_size
        else:
            values_string = ", ".join(TensorType.get_values())
            raise ValueError(f"Unsupported current TensorType. "
                             f"Please confirm if the parameter is in `{values_string}`")


        _, dequant_tensor = fake_quantize(
            states_tensor.contiguous().view(self.tp_size * num_head_per_device, -1),
            scale,
            offset,
            bit=8
        )
        return dequant_tensor.view(states_tensor.shape)

    def configure(self, bit: int, sym: bool, tp_size: int = 1):
        self.tp_size = tp_size

        self.q_observer.configure(bit, sym, self.num_head)
        self.k_observer.configure(bit, sym, self.num_kv_head)
        self.v_observer.configure(bit, sym, self.num_kv_head)
        
    def set_head_params(self, num_head: int, head_dim: int, num_kv_head: int):
        self.num_head = num_head
        self.head_dim = head_dim
        self.num_kv_head = num_kv_head
    
    def is_calibrated(self):
        return self.q_observer.is_calibrated() and self.k_observer.is_calibrated() and self.v_observer.is_calibrated()

    def disable_calibration(self):
        self.is_calib = False
        if self.is_calibrated():
            self.dequant_infer = True

    def enable_calibration(self):
        self.is_calib = True
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()
    
    def record_once(self):
        self.q_observer.enable_record()
        self.k_observer.enable_record()
        self.v_observer.enable_record()
    
    def reset(self):
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()

    def export_quant_params(self) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        q_scale, q_offset = self.q_observer.get_scale_offset()
        k_scale, k_offset = self.k_observer.get_scale_offset()
        v_scale, v_offset = self.v_observer.get_scale_offset()
        return (
            q_scale.to(device="cpu"),
            k_scale.to(device="cpu"),
            v_scale.to(device="cpu")
        ), (
            q_offset.to(device="cpu", dtype=torch.int8),
            k_offset.to(device="cpu", dtype=torch.int8),
            v_offset.to(device="cpu", dtype=torch.int8)
        )

    def _set_head_params(self, config):
        try:
            num_head = config.num_attention_heads
            hidden_size = config.hidden_size
            head_dim = hidden_size // num_head
        except AttributeError as e:
            self.logger.warning("Failed to obtain attention head parameters. Please manually set them")
        try:
            num_key_value_heads = config.num_key_value_heads
        except AttributeError as e:
            self.logger.info("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
            num_key_value_heads = num_head
        self.set_head_params(num_head, head_dim, num_key_value_heads)


def configure_fa(model: torch.nn.Module, bit: int = 8, sym: bool = True, tp_size: int = 1):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.configure(bit, sym, tp_size)


def enable_fa_calibration(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.enable_calibration()


def disable_fa_calibration(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.disable_calibration()


def enable_fa_quantizer_record(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.record_once()


def collect_fa_quantizer_record(model: torch.nn.Module):
    records = {}
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            records[f"{name}.query_states"] = module.fa_quantizer.q_observer.single_record
            records[f"{name}.key_states"] = module.fa_quantizer.k_observer.single_record
            records[f"{name}.value_states"] = module.fa_quantizer.v_observer.single_record
            module.fa_quantizer.q_observer.clear_record()
            module.fa_quantizer.k_observer.clear_record()
            module.fa_quantizer.v_observer.clear_record()
    return records
    

def export_fa_quant_params(module: torch.nn.Module, name: str) -> Tuple[dict, ...]:
    quant_params_scale = defaultdict(torch.Tensor)
    quant_params_offset = defaultdict(torch.Tensor)
    attach_map = defaultdict(list)

    param_types = TensorType.get_values()

    if module.fa_quantizer.is_calibrated():
        scales, offsets = module.fa_quantizer.export_quant_params()

    for i, param_type in enumerate(param_types):
        scale_name = f"{name}.fa_{param_type}.scale"
        offset_name = f"{name}.fa_{param_type}.offset"
        
        if module.fa_quantizer.is_calibrated():
            quant_params_scale[scale_name] = scales[i]
            quant_params_offset[offset_name] = offsets[i]

        attach_map[name].extend([scale_name, offset_name])

    return quant_params_scale, quant_params_offset, attach_map


def is_attn_module_and_then_check_quantizer(module: torch.nn.Module, module_name: str) -> bool:
    if "Attention" not in module.__class__.__name__:
        return False
    if not (hasattr(module, "fa_quantizer") or isinstance(module.fa_quantizer, FAQuantizer)):
        raise AttributeError(f"`FAQuantizer` is not detected in {module_name}. "
                             f"Please check the modeling file and insert FAQuantizer in the correct place.")
    return True
