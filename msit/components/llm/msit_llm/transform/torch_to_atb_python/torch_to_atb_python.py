# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import stat
import json
import inspect
import string
from copy import deepcopy
from collections import namedtuple
from functools import reduce

import torch
import torch_npu

from msit_llm.common.log import logger, set_log_level
from msit_llm.transform.utils import write_file

atb_speed_path = os.getenv("ATB_SPEED_HOME_PATH", None)
if not atb_speed_path:
    logger.warning("ATB_SPEED_HOME_PATH environment variable not valid, will skip build. Try install mindie")
else:
    sys.path.append(os.path.join(atb_speed_path, "lib"))

try:
    from _libatb_torch import _GraphOperation as GraphOperation  # May error
    from _libatb_torch import _BaseOperation as BaseOperation  # May error
except Exception:  # Could be import error or mindie errors, just catch all Exceptions
    logger.warning("import _libatb_torch failed, will skip build. Try install compatible mindie")
    GraphOperation, BaseOperation = None, None

# Force setting ASD print error log to stdout
if os.environ.get("ASDOPS_LOG_LEVEL") == "FATAL":
    os.environ["ASDOPS_LOG_LEVEL"] = "ERROR"
os.environ["ASDOPS_LOG_TO_STDOUT"] = "1"

CONFIG_ATTR_CANDIDATES = {
    "num_hidden_layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "num_attention_heads": ["num_attention_heads"],
    "num_key_value_heads": ["num_key_value_heads"],
    "hidden_size": ["hidden_size"],
    "rms_norm_eps": ["rms_norm_eps"],
    "rope_theta": ["rope_theta"],
    "vocab_size": ["vocab_size"],
    "text_config": ["text_config", "llm_config"],
}

NN_MODULE_STACK = "nn_module_stack"
SKIP_NODES = ["size", "getitem", "to", "float", "finfo", "dropout"]
SKIP_MODULES = ["Dropout"]
TORCH_MODULE_TO_ATB_MAP = {
    "Embedding": dict(op_type="Gather", op_param={}, is_weights_first=True),
    "Gather": dict(op_type="Gather", op_param={}),
    ".*RMSNorm$": dict(op_type="RmsNorm", op_param={"layerType": "RMS_NORM_NORM", "epsilon": 1e-5}),
    ".{0,100}LayerNorm$": dict(
        op_type="LayerNorm",
        op_param={"layerType": "LAYER_NORM_NORM", "normParam": {"beginParamsAxis": 1, "beginNormAxis": 1}},
    ),
    "Linear": dict(op_type="Linear", op_param={"hasBias": False, "enAccum": False}),
    ".*Rotary.*": dict(op_type="Rope", op_param={"rotaryCoeff": 2}),
    ".*Attention$": dict(
        op_type="SelfAttention",
        op_param={"headNum": 1, "kvHeadNum": 1, "calcType": "PA_ENCODER", "qkScale": 1, "maskType": "MASK_TYPE_NORM"},
    ),
    "SiLU.{0,100}": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_SWISH"}),
    "Gelu.{0,100}": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_GELU"}),
    ".{0,100}Gelu": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_GELU"}),
    "add": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_ADD"}),
    "mul": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_MUL"}),
}

_FX_OP_TYPES = ["call_method", "call_module", "call_function", "placeholder", "output", "get_attr"]
FX_OP_TYPES = namedtuple("FX_OP_TYPES", _FX_OP_TYPES)(*_FX_OP_TYPES)

_FIXED_INPUTS = {
    "input_ids",
    "position_ids",
    "inputs_embeds",
    "cos_table",
    "sin_table",
    "slots_mapping",
    "attention_mask",
    "seq_len",
}
FIXED_INPUTS = namedtuple("FIXED_INPUTS", _FIXED_INPUTS)(*_FIXED_INPUTS)
KV_CACHE_SURFFIX = namedtuple("FIXED_INPUTS", ["k_cache", "v_cache"])("k_cache", "v_cache")
BASIC_INPUT_NAMES = (FIXED_INPUTS.input_ids,)

_RESHPAE_KIND = ["reshape_qkv", "reshape_0_12"]
RESHPAE_KIND = namedtuple("RESHPAE_KIND", _RESHPAE_KIND)(*_RESHPAE_KIND)

VALID_NAME_CHARS = string.ascii_letters + string.digits + "_."
FLOAT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def get_config_attr(config, attr, default=None):
    if attr not in CONFIG_ATTR_CANDIDATES:
        return getattr(config, attr, default)

    for sub_attr in CONFIG_ATTR_CANDIDATES[attr]:
        if hasattr(config, sub_attr):
            return getattr(config, sub_attr)
    return default


def build_transformers_model(source_path):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path)
        llm_model_config = get_config_attr(config, "text_config", default=config)
        is_vl_model = llm_model_config is not config
        model = AutoModelForCausalLM.from_config(llm_model_config)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error
    return model, llm_model_config, is_vl_model


def to_transformers_traced_module(model, input_names=BASIC_INPUT_NAMES, disable_check=True):
    from transformers.utils.fx import symbolic_trace

    return symbolic_trace(model, input_names=input_names, disable_check=disable_check)


def get_lambda_source_code(function):
    source_code = inspect.getsource(function).split("function=")[-1].split(", inputs=")[0].strip()
    return source_code[:-1] if source_code.endswith(",") else source_code


def get_valid_name(name):
    return "".join([ii for ii in name if ii in VALID_NAME_CHARS])


class Operation:
    def __init__(
        self, op_type, op_param=None, inputs=None, outputs=None, op_name="", function=None, is_weights_first=False
    ):
        self.op_type, self.op_name, self.function, self.is_weights_first = op_type, op_name, function, is_weights_first
        self.op_param, self.inputs, self.outputs = op_param or {}, inputs or [], outputs or []

    def __repr__(self):
        dd = self.to_json()
        basic_info_keys = ["op_name", "op_type"]
        info = f"op_name={self.op_name}, op_type={self.op_type}\n  "
        info += "\n  ".join([f"{kk}={vv}" for kk, vv in dd.items() if kk not in basic_info_keys])
        return info

    def to_dict(self):
        return dict(
            op_type=self.op_type,
            op_param=self.op_param,
            inputs=self.inputs,
            outputs=self.outputs,
            op_name=self.op_name,
            function=self.function,
            is_weights_first=self.is_weights_first,
        )

    def to_json(self):
        # Keep only valid attributes
        dd = dict(op_type=self.op_type, inputs=self.inputs, outputs=self.outputs, op_name=self.op_name)
        if self.function is not None:
            dd["function"] = get_lambda_source_code(self.function)
        else:
            dd["op_param"] = json.dumps(self.op_param)
        return dd

    def copy(self):
        return Operation(**deepcopy(self.to_dict()))


class ATBModelConfig:
    def __init__(
        self,
        vocab_size=1,
        num_attention_heads=1,
        num_key_value_heads=-1,
        head_dim=1,
        max_batch_size=1,
        max_seq_len=1024,
        rope_theta=1e4,
        **kwargs,
    ):
        self.vocab_size, self.num_attention_heads, self.head_dim = vocab_size, num_attention_heads, head_dim
        self.max_batch_size, self.max_seq_len, self.kwargs = max_batch_size, max_seq_len, kwargs
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_attention_heads
        self.rope_theta = rope_theta
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    def __repr__(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return dict(
            vocab_size=self.vocab_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            **self.kwargs,
        )


class ATBModel:
    def __init__(self, atb_model, atb_model_config=None, output_shape=None, dtype="float16"):
        atb_model_config = atb_model_config or {}
        if isinstance(atb_model_config, dict):
            self.atb_model_config = ATBModelConfig(**atb_model_config)
        elif not isinstance(atb_model_config, ATBModelConfig):
            raise ValueError("atb_model_config is neither a dict or an instance of ATBModelConfig")
        else:
            self.atb_model_config = atb_model_config

        self.atb_model = atb_model
        self.inputs = self.input_names = set(atb_model.input_names)
        self.outputs = self.output_names = atb_model.output_names
        self.model_outputs, self.weights, self.inv_freq_weight, self.attention_mask = {}, {}, None, None

        if output_shape is not None and not isinstance(output_shape, (dict, list, tuple)):
            raise ValueError("output_shape should be None or type in one of (dict, list, tuple)")
        elif isinstance(output_shape, dict):
            required, provided = set(self.output_names), set(output_shape.keys())
            if required != provided:
                raise ValueError(f"output_shape is a dict with keys {provided}, while required are {required}")
        elif isinstance(output_shape, (list, tuple)):
            required, provided = len(self.output_names), len(output_shape)
            if required != provided:
                raise ValueError(f"output_shape len {provided} not equal to required len {required}")
        self.output_shape = output_shape

        if dtype not in FLOAT_DTYPES:
            raise ValueError(f"dtype={dtype} not supported, valid ones are {list(FLOAT_DTYPES.keys())}")
        self.dtype = FLOAT_DTYPES.get(dtype)

        self.head_dim = getattr(atb_model, "head_dim", self.atb_model_config.head_dim)
        self.num_attention_heads = getattr(atb_model, "num_attention_heads", self.atb_model_config.num_attention_heads)
        self.num_key_value_heads = getattr(atb_model, "num_key_value_heads", self.atb_model_config.num_key_value_heads)
        self.vocab_size = getattr(atb_model, "vocab_size", self.atb_model_config.vocab_size)
        self.rope_theta = getattr(atb_model, "rope_theta", self.atb_model_config.rope_theta)
        self.cache_shape = [
            self.atb_model_config.max_batch_size,
            self.atb_model_config.max_seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ]

        self.kv_cache_names = [ii for ii in self.input_names if ii.split(".")[-1] in KV_CACHE_SURFFIX]
        self.past_key_values = {}
        self.atb_model_has_set_weights = hasattr(self.atb_model, "set_weights")  # Supported after RC3

    def __call__(self, input_ids=None, position_ids=None, slots_mapping=None, **kwargs):
        return self.forward(input_ids, position_ids, slots_mapping, **kwargs)

    def init_kv_cache(self):
        self.past_key_values = {ii: torch.zeros(self.cache_shape).to(self.dtype).npu() for ii in self.kv_cache_names}
        self.weights.update(self.past_key_values)
        if self.atb_model_has_set_weights:
            self.atb_model.set_weights(self.past_key_values)

    def _calc_inv_freq_by_rope_theta(self):
        inv_freq_weight = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        return inv_freq_weight.float().npu()

    def _calc_cos_sin_table_from_inv_freq(self, position_ids):
        logger.debug(f"self.inv_freq_weight.shape = {self.inv_freq_weight.shape}")
        logger.debug(f"position_ids.shape = {position_ids.shape}")
        left = self.inv_freq_weight[None, :, None]
        right = position_ids[:, None, :] if position_ids.dim() == 2 else position_ids[None, None, :]
        freq = (left.to(right.device).float() @ right.float()).transpose(1, 2)
        freq = torch.cat([freq, freq], dim=-1)[0].npu()  # has to be float npu values, and dim == 2
        logger.debug(f"freq.shape = {freq.shape}")
        return freq.cos().to(self.dtype), freq.sin().to(self.dtype)

    def set_weights(self, weights):
        source_weights = set(weights.keys())
        valid_weights = source_weights & self.inputs
        missing_weights = self.inputs - source_weights - set(FIXED_INPUTS)
        unused_weights = source_weights - self.inputs

        for kk in valid_weights:
            cur_weight = weights[kk]
            dtype_str = str(cur_weight.dtype).split(".")[-1]
            if dtype_str in FLOAT_DTYPES and cur_weight.dtype != self.dtype:
                cur_weight = cur_weight.to(self.dtype)
            self.weights[kk] = cur_weight.npu()

        for weight_name in unused_weights:
            if "invfreq" in weight_name.split(".")[-1].replace("_", "").lower():
                self.inv_freq_weight = weights[weight_name].squeeze().float().npu()
                unused_weights.remove(weight_name)
                break

        need_cos_sin_table = FIXED_INPUTS.cos_table in self.inputs or FIXED_INPUTS.sin_table in self.inputs
        if need_cos_sin_table and self.inv_freq_weight is None:
            logger.info(f"Set inv_freq_weight by rope_theta={self.rope_theta}")
            self.inv_freq_weight = self._calc_inv_freq_by_rope_theta()

        if FIXED_INPUTS.attention_mask in self.inputs:
            mask_tensor = torch.ones([self.atb_model_config.max_seq_len, self.atb_model_config.max_seq_len])
            attention_mask = torch.where((1 - torch.tril(mask_tensor)).to(torch.bool), -torch.inf, 0)
            self.attention_mask = attention_mask.to(self.dtype).npu()

        if self.atb_model_has_set_weights:
            self.atb_model.set_weights(self.weights)  # ATB provided function, no need to pass weights again

        if len(self.kv_cache_names) > 0:
            self.init_kv_cache()
            missing_weights -= set(self.kv_cache_names)

        if len(unused_weights) > 0:
            logger.warning(f"unused weights: {unused_weights}")
        if len(missing_weights) > 0:
            logger.warning(f"missing weights: {missing_weights}")

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, slots_mapping=None, **kwargs):
        # Basic inputs
        model_inputs, batch_size, cur_pos, input_len = {}, 1, 1, 1
        if not self.atb_model_has_set_weights:
            model_inputs.update(self.weights)
        if input_ids is not None:
            batch_size = input_ids.shape[0] if input_ids.dim() == 2 else 1
            input_len = cur_pos = input_ids.shape[-1]
            model_inputs[FIXED_INPUTS.input_ids] = input_ids.npu()
            logger.debug(f"batch_size = {batch_size}, input_len = {input_len}")
        if inputs_embeds is not None:  # For VL model
            batch_size = inputs_embeds.shape[0] if inputs_embeds.dim() == 3 else 1
            input_len = cur_pos = inputs_embeds.shape[-2]
            model_inputs[FIXED_INPUTS.inputs_embeds] = inputs_embeds.npu()
            logger.debug(f"batch_size = {batch_size}, input_len = {input_len}")
        if position_ids is not None:
            cur_pos = (position_ids[0, -1] if position_ids.dim() == 2 else position_ids[-1]).cpu() + 1
            model_inputs[FIXED_INPUTS.position_ids] = position_ids.npu()
            logger.debug(f"cur_pos = {cur_pos}")

        # inputs interpreted from others, or with default values
        if FIXED_INPUTS.slots_mapping in self.inputs:
            if slots_mapping is None:
                slots_mapping = torch.zeros([batch_size * input_len], dtype=torch.int)
            model_inputs[FIXED_INPUTS.slots_mapping] = slots_mapping.npu()

        # Check kwargs
        model_inputs.update({kk: vv.npu() for kk, vv in kwargs.items()})
        if FIXED_INPUTS.cos_table in self.inputs or FIXED_INPUTS.sin_table in self.inputs:
            meets_cos_sin_table = FIXED_INPUTS.cos_table in model_inputs and FIXED_INPUTS.sin_table in model_inputs
            if not meets_cos_sin_table and self.inv_freq_weight is not None and position_ids is not None:
                # if either cos_table or sin_table is missing, the values of both will be calculated
                cos_table, sin_table = self._calc_cos_sin_table_from_inv_freq(position_ids)
                model_inputs[FIXED_INPUTS.cos_table], model_inputs[FIXED_INPUTS.sin_table] = cos_table, sin_table
        if FIXED_INPUTS.attention_mask in self.inputs and FIXED_INPUTS.attention_mask not in model_inputs:
            model_inputs[FIXED_INPUTS.attention_mask] = self.attention_mask
        if FIXED_INPUTS.seq_len in self.inputs and FIXED_INPUTS.seq_len not in model_inputs:
            seq_len = torch.ones([batch_size], dtype=torch.int) * cur_pos
            model_inputs[FIXED_INPUTS.seq_len] = seq_len.npu()

        # Show missing inputs, in some cases like testing scenario, this may not an error
        missing_inputs = self.inputs - set(model_inputs.keys()) - set(self.weights.keys())
        if len(missing_inputs) != 0:
            logger.warning(
                f"Missing inputs: {missing_inputs}\nProvided: {model_inputs.keys()}\nCall `set_weights` if not already"
            )  # Not raising error, model may still can be executed

        # Creats output. Here output_shape maybe None or a dict or list
        if self.output_shape is None:
            self.model_outputs = {
                ii: torch.ones([batch_size * input_len, self.vocab_size]).to(self.dtype).npu() for ii in self.outputs
            }
        elif isinstance(self.output_shape, dict):
            self.model_outputs = {kk: torch.ones(vv).to(self.dtype).npu() for kk, vv in self.output_shape.items()}
        else:
            self.model_outputs = {
                kk: torch.ones(vv).to(self.dtype).npu() for kk, vv in zip(self.outputs, self.output_shape)
            }

        # Run forward
        bind_map = {}
        if FIXED_INPUTS.seq_len in self.inputs:
            bind_map[FIXED_INPUTS.seq_len] = model_inputs[FIXED_INPUTS.seq_len].cpu()
        return self.atb_model.forward(model_inputs, self.model_outputs, bind_map)


class ATBModelFromTorch(ATBModel):
    """Create ATB model from raw transformers torch model
    >>> import torch, torch_npu
    >>> from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    >>> from msit_llm.transform.torch_to_atb_python import ATBModel, ATBModelConfig, ATBModelFromTorch
    >>>
    >>> cc = LlamaConfig()
    >>> cc.num_hidden_layers, cc.hidden_size, cc.intermediate_size = 2, 1024, 4096  # smaller
    >>> mm = LlamaForCausalLM(cc).eval()
    >>> torch.save(mm.state_dict(), 'stat_dict.pt')  # Save for later usage
    >>> input_ids, position_ids = torch.arange(32), torch.arange(32)
    >>> with torch.no_grad():
    >>>     torch_out = mm(input_ids=input_ids[None], position_ids=position_ids[None]).logits
    >>>
    >>> atb_model = ATBModelFromTorch(mm)
    >>> atb_model.set_weights(mm.state_dict())
    >>> # Also set buffers in, which includes inv_freq. Ignore WARNINGs of missing weights
    >>> atb_model.set_weights(dict(mm.named_buffers()))
    >>> out = atb_model(input_ids=input_ids, position_ids=position_ids)
    >>> print({kk: vv.shape for kk, vv in out.items()})
    # 输出Size为{'output': torch.Size([32, 32000])}
    >>> print(torch.allclose(torch_out, out['output'].cpu().float(), atol=5e-2))
    # True
    >>> atb_model.to_file()  # Save atb model to a py file
    # 'llamaforcausallm_atb_float.py'

    Create ATB model from saved file and load saved weights
    >>> import torch, torch_npu, llamaforcausallm_atb_float
    >>> from msit_llm.transform.torch_to_atb_python import ATBModel
    >>> aa = ATBModel(llamaforcausallm_atb_float.Model(), {'vocab_size': 32000})
    >>> ss = torch.load('stat_dict.pt')  # Load from previously saved weights
    >>> aa.set_weights(ss)
    >>> # Set inv_freq for cos_table for sin_table. Ignore WARNINGs of missing weights
    >>> inv_freq = 1.0 / (1e4 ** (torch.arange(0, aa.atb_model.head_dim, 2) / aa.atb_model.head_dim))
    >>> aa.set_weights({'inv_freq': inv_freq})
    >>> out = atb_model(input_ids=input_ids, position_ids=position_ids)
    >>> print({kk: vv.shape for kk, vv in out.items()})
    # 输出Size为{'output': torch.Size([32, 32000])}
    """

    def __init__(
        self,
        torch_model,
        config=None,
        input_names=BASIC_INPUT_NAMES,
        max_batch_size=1,
        max_seq_len=1024,
        to_quant=False,
        quant_disable_names=None,
        is_vl_model=False,
        dtype="float16",
    ):
        self.torch_model, self.config, self.is_vl_model = torch_model, config, is_vl_model
        self.to_quant, self.quant_disable_names = to_quant, quant_disable_names or ()
        self.config = getattr(torch_model, "config", None) if config is None else config
        if self.config is None:
            raise ValueError("Either config or torch_model.config shold be not empty one")

        self.num_attention_heads = get_config_attr(self.config, "num_attention_heads")
        if self.num_attention_heads is None or self.num_attention_heads <= 0:
            raise ValueError("Invalid config, num_attention_heads not exists or < 0")

        self.hidden_size = get_config_attr(self.config, "hidden_size")
        if self.hidden_size is None or self.hidden_size <= 0:
            raise ValueError("Invalid config, hidden_size not exists or < 0")

        self.vocab_size = get_config_attr(self.config, "vocab_size", -1)
        if self.vocab_size is None or self.vocab_size <= 0:
            logger.warning("vocab_size from config not exists or < 0. Ignore if this is intended")

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.traced_module = to_transformers_traced_module(torch_model, input_names=input_names)
        self.model_name = get_valid_name(torch_model.__class__.__name__)

        self.weight_names, self.weight_stack_map = list(self.traced_module.state_dict().keys()), {}
        for ii in self.weight_names:
            self.weight_stack_map.setdefault(".".join(ii.split(".")[:-1]), []).append(ii)

        self.num_key_value_heads = get_config_attr(self.config, "num_key_value_heads", default=self.num_attention_heads)
        self.rope_theta = get_config_attr(self.config, "rope_theta", default=1e4)
        self.rms_norm_eps = get_config_attr(self.config, "rms_norm_eps", default=1e-5)

        self.torch_module_to_atb_map = {}
        for kk, vv in TORCH_MODULE_TO_ATB_MAP.items():
            re_key = re.compile(kk)
            if vv.get("op_type", None) == "SelfAttention" and "op_param" in vv:
                vv["op_param"].update({"headNum": self.num_attention_heads, "kvHeadNum": self.num_key_value_heads})
                vv["op_param"].update({"qkScale": 1 / (float(self.head_dim) ** 0.5)})
            elif vv.get("op_type", None) == "RmsNorm" and "op_param" in vv:
                vv["op_param"].update({"epsilon": self.rms_norm_eps})
            self.torch_module_to_atb_map[re_key] = Operation(**vv)

        self.pre_qkv_name, self.pre_query_name, self.pre_key_name, self.pre_value_name = "", "", "", ""
        self.is_apply_rope = False
        self.model_inputs, self.model_outputs, self.operations = [], [], []
        # base graph is set execute_as_single=False, has to keep all operaions as property
        self.base_graph_operations, self.k_cache_names, self.v_cache_names = [], [], []

        self.convert_fx_traced_module()
        if to_quant:
            logger.info(f"calling convert_to_quant, quant_disable_names = {self.quant_disable_names}")
            self.convert_to_quant()
        self.to_file()

        self.atb_model_config = ATBModelConfig(
            vocab_size=self.vocab_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        if GraphOperation is None or BaseOperation is None:
            logger.warning("Will skip build")
        else:
            self.atb_model = self.build_atb_model()
            super().__init__(atb_model=self.atb_model, atb_model_config=self.atb_model_config, dtype=dtype)

    @staticmethod
    def get_cur_repeat_block_idx(module_name):
        for tok in module_name.split("."):
            if str.isdigit(tok):
                return int(tok)
        return -1

    @staticmethod
    def _get_module_type_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].values())
        return None if len(node_module_stack) == 0 else node_module_stack[-1].__name__

    @staticmethod
    def _get_module_name_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].keys())
        return None if len(node_module_stack) == 0 else node_module_stack[-1]

    @staticmethod
    def _should_skip_node(node):
        if node.op == FX_OP_TYPES.call_method and node.target in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.meta.get("is_wrapped", False):
            return True
        return False

    def to_file(self, output_file=None):
        indent = " " * 4
        stacked_operations, stacked_inputs, stacked_outputs = self._stack_operations()
        base_model_name = self.model_name

        contents = [
            "import os",
            "import sys",
            "import json",
            "import torch",
            "import torch_npu",
            "from functools import reduce",
            "",
            "atb_speed_path = os.getenv('ATB_SPEED_HOME_PATH')",
            "if not atb_speed_path:",
            f"{indent}raise OSError('ATB_SPEED_HOME_PATH environment variable not valid. Try install mindie')",
            "sys.path.append(os.path.join(atb_speed_path, 'lib'))",
            "from _libatb_torch import _GraphOperation as GraphOperation",
            "from _libatb_torch import _BaseOperation as BaseOperation",
            "",
            "if os.environ.get('ASDOPS_LOG_LEVEL') == 'FATAL':",
            f"{indent}os.environ['ASDOPS_LOG_LEVEL'] = 'ERROR'",
            "os.environ['ASDOPS_LOG_TO_STDOUT'] = '1'  # Force setting ASD printing error log to stdout",
            "",
            "class Model(GraphOperation):",
            f"{indent}def __init__(self, outputs=None):",
            f"{indent * 2}self.model_name = '{base_model_name}'",
            f"{indent * 2}super().__init__(self.model_name)",
            "",
            f"{indent * 2}self.num_attention_heads, self.head_dim = {self.num_attention_heads}, {self.head_dim}",
            f"{indent * 2}self.num_key_value_heads, self.vocab_size = {self.num_key_value_heads}, {self.vocab_size}",
            f"{indent * 2}self.rope_theta = {self.rope_theta}",
        ]

        def _get_input_output_name(graph_name):
            return f"{graph_name}_inputs", f"{graph_name}_outputs"

        def _to_file(graph_name, operations, depth=0):
            contents.append("")
            if depth > 0:
                contents.append(f"{indent * 2}{graph_name} = GraphOperation('{graph_name}')")
                this_name = graph_name
            else:
                this_name = "self"

            graph_inputs, graph_outputs = stacked_inputs.pop(0), stacked_outputs.pop(0)
            contents.append(f"{indent * 2}{graph_name}_inputs = [")
            for ii in sorted(graph_inputs, key=lambda xx: "0" if xx in FIXED_INPUTS else xx):
                contents.append(f"{indent * 3}'{ii}',")
            contents.append(f"{indent * 2}]")

            if depth > 0:
                contents.append(f"{indent * 2}{graph_name}_outputs = {graph_outputs}")
            else:
                contents.append(f"{indent * 2}{graph_name}_outputs = outputs or {graph_outputs}")
            cur_inputs, cur_outputs = _get_input_output_name(graph_name)
            contents.append(f"{indent * 2}{this_name}.add_input_output(input={cur_inputs}, output={cur_outputs})")
            contents.append("")

            sub_graph_id = 0
            for op in operations:
                if isinstance(op, list):
                    sub_graph_name = f"sub_graph_{sub_graph_id}"
                    sub_graph_inputs, sub_graph_outputs = stacked_inputs[0], stacked_outputs[0]
                    _to_file(sub_graph_name, op, depth=depth + 1)
                    cur_inputs, cur_outputs = _get_input_output_name(sub_graph_name)
                    contents.append(
                        f"{indent * 2}{this_name}.add_operation({sub_graph_name}, {cur_inputs}, {cur_outputs})"
                    )
                    contents.append(f"{indent * 2}{this_name}.{sub_graph_name} = {sub_graph_name}")
                    contents.append("")
                    sub_graph_id += 1
                elif op.op_type == "add_reshape":
                    function = get_lambda_source_code(op.function)
                    contents.append(
                        f"{indent * 2}{this_name}.add_reshape('{op.inputs[0]}', '{op.outputs[0]}', {function})"
                    )
                else:
                    op_kwargs = f"op_type='{op.op_type}', op_param='{json.dumps(op.op_param)}', op_name='{op.op_name}'"

                    if depth == 0:
                        cur_op = f"{this_name}." + op.op_name.replace(".", "_")
                        contents.append(f"{indent * 2}{cur_op} = BaseOperation({op_kwargs})")
                    else:
                        cur_op = f"BaseOperation({op_kwargs})"
                    contents.append(f"{indent * 2}{this_name}.add_operation(")
                    contents.append(f"{indent * 3}operation={cur_op},")
                    contents.append(f"{indent * 3}input={op.inputs},")
                    contents.append(f"{indent * 3}output={op.outputs},")
                    contents.append(f"{indent * 2})")

            contents.append("")
            contents.append(f"{indent * 2}{this_name}.execute_as_single = {False if depth == 0 else True}")
            contents.append(f"{indent * 2}{this_name}.build()")

        _to_file(base_model_name, stacked_operations)
        contents.append("")
        contents_str = "\n".join(contents)

        if output_file is None:
            output_file = "_".join([base_model_name.lower(), "atb", "quant.py" if self.to_quant else "float.py"])
        elif not output_file.endswith(".py"):
            output_file += ".py"
        write_file(output_file, contents_str)
        return output_file

    def _get_node_type_and_inputs_and_name(self, node, output_node_map=None):
        output_node_map = output_node_map or {}
        cur_module_name = self._get_module_name_by_nn_module_stack(node)
        cur_model_type = self._get_module_type_by_nn_module_stack(node)
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ == "mul" and "gelu" in cur_model_type.lower():
            cur_inputs = [None]
            module_name = cur_module_name
        elif node.op == FX_OP_TYPES.call_function and self._find_in_torch_module_to_atb_map(node.target.__name__):
            cur_model_type = node.target.__name__
            # No other inputs if function
            cur_inputs = []
            for ii in node.all_input_nodes:
                if ii.name not in output_node_map:
                    continue
                cur_inputs += output_node_map[ii.name]
            module_name = cur_module_name + "." + node.name
        else:
            # None marks for placeholder of other inputs
            cur_inputs = self.weight_stack_map.get(cur_module_name, []) + [None]
            module_name = cur_module_name
        return cur_model_type, cur_inputs, module_name

    def _check_and_set_pre_qkv_name(self, atb_operation):
        if atb_operation.op_type == "Linear":
            sub_name = atb_operation.op_name.split(".")[-1]
            logger.debug(f"Checking if qkv Linear: {atb_operation}")
            if all(sub in sub_name for sub in ("q", "k", "v")):
                self.pre_qkv_name = atb_operation.outputs[0]
            elif "q" in sub_name:
                self.pre_query_name = atb_operation.outputs[0]
            elif "k" in sub_name:
                self.pre_key_name = atb_operation.outputs[0]
            elif "v" in sub_name:
                self.pre_value_name = atb_operation.outputs[0]

    def _find_in_torch_module_to_atb_map(self, node_type):
        for kk, vv in self.torch_module_to_atb_map.items():
            if kk.fullmatch(node_type):
                return vv.copy()
        return None

    def _convert_module(self, node_module_type, node_module_name, input_names):
        atb_operation = self._find_in_torch_module_to_atb_map(node_module_type)
        outputs = [node_module_name + ".out"]
        atb_operation.op_name = node_module_name
        atb_operation.inputs = getattr(atb_operation, "inputs", []) + input_names
        atb_operation.outputs = getattr(atb_operation, "outputs", []) + outputs
        return atb_operation

    def _op_process_linear(self, atb_operation=None, module_name=""):
        bias_name = f"{atb_operation.op_name}.bias"
        if bias_name in self.weight_names:
            atb_operation.op_param.update({"hasBias": True})
        self.operations.append(atb_operation)

    def _op_process_rope(self, atb_operation=None, module_name=""):
        self.is_apply_rope = True
        self.model_inputs += [FIXED_INPUTS.cos_table, FIXED_INPUTS.sin_table]
        self.operations.append(
            self._convert_module("Gather", "gather_cos", [FIXED_INPUTS.cos_table, FIXED_INPUTS.position_ids])
        )
        self.operations.append(
            self._convert_module("Gather", "gather_sin", [FIXED_INPUTS.sin_table, FIXED_INPUTS.position_ids])
        )

        if FIXED_INPUTS.position_ids not in self.model_inputs:
            self.model_inputs += [FIXED_INPUTS.position_ids]

    def _op_process_attention(self, atb_operation=None, module_name=""):
        atb_operation.inputs = [
            module_name + ".q_embed_",
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            FIXED_INPUTS.attention_mask,
            FIXED_INPUTS.seq_len,
        ]

        query_name, key_name, value_name = self.pre_query_name, self.pre_key_name, self.pre_value_name
        if self.pre_qkv_name:
            logger.debug(f"Got stacked QKV Linear: {self.pre_qkv_name}")
            inputs = self.pre_qkv_name
            query_name, key_name, value_name = module_name + ".q", module_name + ".k", module_name + ".v"
            self.operations.append(
                Operation(
                    op_type="Split",
                    op_param={"splitDim": 1, "splitNum": 3},
                    inputs=[inputs],
                    outputs=[query_name, key_name, value_name],
                    op_name=module_name + ".split",
                )
            )
        # Set back to default value
        self.pre_qkv_name, self.pre_query_name, self.pre_key_name, self.pre_value_name = "", "", "", ""

        if self.is_apply_rope:
            inputs = [query_name, key_name, "gather_cos.out", "gather_sin.out", FIXED_INPUTS.seq_len]
            query_name, key_name = module_name + ".q_embed", module_name + ".k_embed"
            self.operations.append(
                Operation(
                    op_type="Rope",
                    op_param={"rotaryCoeff": 2},
                    inputs=inputs,
                    outputs=[query_name, key_name],
                    op_name=module_name + ".rope",
                )
            )

        k_cache_name = module_name + "." + KV_CACHE_SURFFIX.k_cache
        v_cache_name = module_name + "." + KV_CACHE_SURFFIX.v_cache
        reshape_and_cache_inputs = [
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            k_cache_name,
            v_cache_name,
            FIXED_INPUTS.slots_mapping,
        ]
        self.k_cache_names.append(k_cache_name)
        self.v_cache_names.append(v_cache_name)
        self.operations += [
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_attention_heads, self.head_dim],
                inputs=[query_name],
                outputs=[module_name + ".q_embed_"],
                op_name=module_name + ".q." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_key_value_heads, self.head_dim],
                inputs=[key_name],
                outputs=[module_name + ".k_embed_"],
                op_name=module_name + ".k." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_key_value_heads, self.head_dim],
                inputs=[value_name],
                outputs=[module_name + ".v_embed_"],
                op_name=module_name + ".v." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="ReshapeAndCache",
                op_param={},
                inputs=reshape_and_cache_inputs,
                outputs=[k_cache_name, v_cache_name],
                op_name=module_name + ".reshape_and_cache",
            ),
            atb_operation,
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], reduce(lambda xx, yy: xx * yy, org_shape[1:])],
                inputs=atb_operation.outputs,
                outputs=[ii + "_" for ii in atb_operation.outputs],
                op_name=module_name + "." + RESHPAE_KIND.reshape_0_12,
            ),
        ]

        if FIXED_INPUTS.slots_mapping not in self.model_inputs:
            self.model_inputs += [
                FIXED_INPUTS.slots_mapping,
                FIXED_INPUTS.attention_mask,
                FIXED_INPUTS.seq_len,
            ]
        self.model_inputs += [k_cache_name, v_cache_name]

    def _refine_inputs_outputs(self, input_node_map, output_node_map, operation_outputs):
        gathered_module_inputs = {}
        for kk, vv in input_node_map.items():
            gathered_inputs = []
            for ii in vv:
                if ii in output_node_map:
                    gathered_inputs += output_node_map[ii]
            for ii in set(gathered_inputs):
                if ii == kk or ii == kk + ".out":
                    continue
                gathered_module_inputs.setdefault(kk, []).extend(
                    operation_outputs.get(ii, ii if isinstance(ii, list) else [ii])
                )

        for op in self.operations:
            if not isinstance(op, list) and (len(op.inputs) == 0 or op.inputs[-1] is not None):
                continue

            op.inputs = op.inputs[:-1]  # Exclude last None
            if op.is_weights_first:
                op.inputs = op.inputs + gathered_module_inputs.get(op.op_name, [])
            else:
                op.inputs = gathered_module_inputs.get(op.op_name, []) + op.inputs

    def _replace_input_ids_by_inputs_embeds_for_vl_model(self):
        if FIXED_INPUTS.input_ids in self.model_inputs:
            self.model_inputs = [
                FIXED_INPUTS.inputs_embeds if ii == FIXED_INPUTS.input_ids else ii for ii in self.model_inputs
            ]

        embed_op_id, embed_outputs = -1, None
        for op_id, op in enumerate(self.operations):
            if len(op.inputs) == 2 and len(op.outputs) == 1 and FIXED_INPUTS.input_ids in op.inputs:  # Embedding
                op.inputs = [FIXED_INPUTS.inputs_embeds]
                embed_outputs = op.outputs[0]
                embed_op_id = op_id
                logger.info(f"Got Embedding op, embed_op_id: {embed_op_id}")
                continue
            if embed_outputs is None:
                continue
            if embed_outputs in op.inputs:
                op.inputs = [FIXED_INPUTS.inputs_embeds if ii == embed_outputs else ii for ii in op.inputs]

        if embed_op_id >= 0:
            logger.info(f"Remove op: {self.operations[embed_op_id]}")
            logger.info(f"Replace op inner names {embed_outputs} -> {FIXED_INPUTS.inputs_embeds}")
            self.operations.pop(embed_op_id)

    def _stack_operations(self):
        stacked_operations, cur_stack, pre_stack_id = [], [], -1
        for ii in self.operations:
            cur_stack_id = self.get_cur_repeat_block_idx(ii.op_name)
            if cur_stack_id == -1:  # Not a repeated block, could be in or out block
                if len(cur_stack) > 0:
                    stacked_operations.append(cur_stack)
                stacked_operations.append(ii)
                cur_stack = []
                continue
            if pre_stack_id != cur_stack_id:  # Changed to another block
                if len(cur_stack) > 0:
                    stacked_operations.append(cur_stack)
                cur_stack = [ii]
                pre_stack_id = cur_stack_id
            else:  # Repeat block
                cur_stack.append(ii)

        all_inputs = set([ii for op in self.operations for ii in op.inputs])
        base_graph_in_tensors = self.model_inputs + self.weight_names
        valid_base_graph_inputs = [ii for ii in base_graph_in_tensors if ii in all_inputs]

        all_outputs = set([ii for op in self.operations for ii in op.outputs])
        base_graph_out_tensors = [ii for ii in all_outputs if ii not in all_inputs]

        # record inputs and outputs for stacked GraphOperations
        stacked_inputs, stacked_outputs = [valid_base_graph_inputs], [base_graph_out_tensors]
        for ops in stacked_operations:
            if not isinstance(ops, list):
                continue
            cur_inputs, cur_outputs, inplace_outputs = [], [], []
            for op in ops:
                for ii in op.inputs:
                    cur_inputs.append(ii)
                for ii in op.outputs:
                    cur_outputs.append(ii)
                    if ii in op.inputs:
                        inplace_outputs.append(ii)
            cur_inputs_set = set(cur_inputs)
            cur_outputs_set = set(cur_outputs)
            inplace_outputs_set = set(inplace_outputs)
            stacked_inputs.append(list(cur_inputs_set - (cur_outputs_set - inplace_outputs_set)))
            stacked_outputs.append(list(cur_outputs_set & (all_inputs - cur_inputs_set)))
        return stacked_operations, stacked_inputs, stacked_outputs

    def convert_fx_traced_module(self):
        previous_module_name, cur_module_name, previous_operation_out, base_module_name = None, None, None, None
        input_node_map, output_node_map, operation_outputs = {}, {}, {}

        for node in self.traced_module.graph.nodes:
            logger.debug("=" * 30 + "\n")
            logger.debug(f"node.name = {node.name}, node.op = {node.op}, node.meta = {node.meta}")
            logger.debug(f"node.all_input_nodes = {node.all_input_nodes}, node.target = {node.target}")
            if node.op == FX_OP_TYPES.placeholder:  # Input node
                self.model_inputs.append(node.name)
                input_node_map[node.name] = [node.name]
                output_node_map[node.name] = [node.name]
                continue
            if node.op == FX_OP_TYPES.output:  # Output node
                self.model_outputs.append(node.name)
                continue
            if not hasattr(node, "meta") or not node.meta.get(NN_MODULE_STACK, []):
                if node.op == FX_OP_TYPES.call_function:
                    output_node_map[node.name] = previous_operation_out  # op like getitem, set to previous output
                continue

            if base_module_name is None:
                base_module_name = list(node.meta[NN_MODULE_STACK].keys())[0]

            cur_module_name = self._get_module_name_by_nn_module_stack(node)
            input_node_map.setdefault(cur_module_name, []).extend([ii.name for ii in node.all_input_nodes])

            logger.debug(f"cur_module_name = {cur_module_name}, previous_module_name = {previous_module_name}")
            if cur_module_name != base_module_name:
                output_node_map[node.name] = previous_operation_out  # will be overwriten later
            if cur_module_name == previous_module_name:
                continue
            if self._should_skip_node(node):
                logger.debug(f"Current node skipped: {node.name}")
                continue
            if node.op == FX_OP_TYPES.call_module and self._get_module_type_by_nn_module_stack(node) in SKIP_MODULES:
                logger.debug(f"Current module skipped: {node.name}")
                cur_module_name = previous_module_name  # Module like Dropout skipped, set back to previous name
                continue
            previous_module_name = cur_module_name

            logger.debug(f"cur_module_name = {cur_module_name}, node.name = {node.name}")
            node_module_type, cur_inputs, module_name = self._get_node_type_and_inputs_and_name(node, output_node_map)
            logger.debug(f"node_module_type={node_module_type}, cur_inputs={cur_inputs}, module_name={module_name}")
            if self._find_in_torch_module_to_atb_map(node_module_type) is None:
                logger.warning(f"node not supported: node.name = {node.name}, node.type = {node_module_type}")
                continue
            if len(cur_inputs) == 0:
                logger.warning(f"found none valid input: node.name = {node.name}, node.type = {node_module_type}")
                continue

            atb_operation = self._convert_module(node_module_type, module_name, cur_inputs)
            self._check_and_set_pre_qkv_name(atb_operation)
            logger.debug(f"atb_operation = {atb_operation.to_json()}")
            if not self.is_apply_rope and atb_operation.op_type == "Rope":
                self._op_process_rope(atb_operation=atb_operation, module_name=module_name)
            elif atb_operation.op_type == "SelfAttention":
                self._op_process_attention(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            elif atb_operation.op_type in ["Linear", "LinearParallel"]:
                self._op_process_linear(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            else:
                self.operations.append(atb_operation)
                cur_outputs = atb_operation.outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs

        self._refine_inputs_outputs(input_node_map, output_node_map, operation_outputs)
        if self.is_vl_model:
            self._replace_input_ids_by_inputs_embeds_for_vl_model()
        self.operations[-1].outputs = self.model_outputs
        return self.model_inputs, self.model_outputs, self.weight_names, self.operations

    def convert_to_quant(self):
        # Has to split out from convert_fx_traced_module, needs actual Linear input names
        quant_disable_names = set([ii for ii in self.quant_disable_names if ii is not None and len(ii) > 0])
        operations_with_quant = []
        for _, op in enumerate(self.operations):
            logger.debug(f"op.op_name = {op.op_name}")
            if op.op_type not in ["Linear", "LinearParallel"]:
                operations_with_quant.append(op)
                continue
            if op.op_name in quant_disable_names:
                operations_with_quant.append(op)
                quant_disable_names.remove(op.op_name)
                continue

            logger.debug(f"Convert {op.op_name} to quant node")
            linear_quant_weights = []
            bias, deq_scale = f"{op.op_name}.bias", f"{op.op_name}.deq_scale"
            input_scale, input_offset = f"{op.op_name}.input_scale", f"{op.op_name}.input_offset"
            self.weight_names += [input_scale, input_offset, deq_scale]

            if bias not in op.inputs:
                linear_quant_weights.append(bias)
                self.weight_names.append(bias)
            linear_quant_weights.append(deq_scale)

            elewise_quant_node = Operation(
                op_type="Elewise",
                op_param={"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"},
                op_name=f"{op.op_name}.elewise_quant",
                inputs=[f"{op.inputs[0]}", input_scale, input_offset],
                outputs=[f"{op.op_name}.elewise_quant.out"],
            )

            op.op_param.update({"transposeA": False, "transposeB": True, "hasBias": True, "outDataType": "ACL_FLOAT16"})
            op.inputs = elewise_quant_node.outputs + op.inputs[1:] + linear_quant_weights

            operations_with_quant += [elewise_quant_node, op]
        self.operations = operations_with_quant

        if len(quant_disable_names) > 0:
            logger.warning(f"Some layers in quant_disable_names not found in model: {quant_disable_names}")

    def build_atb_model(self):
        stacked_operations, stacked_inputs, stacked_outputs = self._stack_operations()

        def _build_atb_model(graph_name, operations, depth=0):
            atb_model = GraphOperation(graph_name)
            graph_inputs, graph_outputs = stacked_inputs.pop(0), stacked_outputs.pop(0)
            logger.debug(f"graph_name = {graph_name}")
            logger.debug(f"len(graph_inputs) = {len(graph_inputs)}, graph_inputs = {graph_inputs}")
            logger.debug(f"len(graph_outputs) = {len(graph_outputs)}, graph_outputs = {graph_outputs}")
            logger.debug(f"operations: {[ii.op_name for ii in operations if not isinstance(ii, list)]}")

            atb_model.add_input_output(input=graph_inputs, output=graph_outputs)
            sub_graph_id = 0
            for op in operations:
                if isinstance(op, list):
                    sub_graph_name = f"sub_graph_{sub_graph_id}"
                    cur_operation = _build_atb_model(sub_graph_name, op, depth=depth + 1)
                    atb_model.add_operation(cur_operation, cur_operation.input_names, cur_operation.output_names)
                    sub_graph_id += 1
                elif op.op_type == "add_reshape":
                    atb_model.add_reshape(op.inputs[0], op.outputs[0], op.function)
                    continue
                else:
                    cur_operation = BaseOperation(
                        op_type=op.op_type, op_param=json.dumps(op.op_param), op_name=op.op_name
                    )
                    atb_model.add_operation(operation=cur_operation, input=op.inputs, output=op.outputs)
                if depth == 0:
                    self.base_graph_operations.append(cur_operation)
            atb_model.execute_as_single = False if depth == 0 else True
            atb_model.build()
            return atb_model

        atb_model = _build_atb_model(self.model_name, stacked_operations)
        return atb_model


def generate_infer_file(output_file, source_path, is_vl_model=False):
    from pathlib import Path

    file_name = "run_vl.py" if is_vl_model else "run.py"
    infer_file = Path(output_file).with_name(file_name)
    contents_str = Path(__file__).with_name(file_name).read_text()
    contents_str = contents_str.replace("atb_model_placeholder", Path(output_file).stem)
    contents_str = contents_str.replace("model_path_placeholder", os.path.abspath(source_path))
    write_file(infer_file, contents_str)
    return infer_file


def transform(source_path, input_names=BASIC_INPUT_NAMES, output_file=None, to_quant=False, quant_disable_names=None):
    logger.info("Building model using transformers...")
    model, config, is_vl_model = build_transformers_model(source_path)

    if is_vl_model:
        logger.info("Got VL model")

    logger.info("Transforming to atb")
    atb_model = ATBModelFromTorch(
        torch_model=model,
        config=config,
        input_names=input_names,
        to_quant=to_quant,
        quant_disable_names=quant_disable_names,
        is_vl_model=is_vl_model,
    )
    output_file = atb_model.to_file(output_file=output_file)

    logger.info("=" * 30)
    logger.info(f"Saved to: {output_file}\n")

    logger.info("=" * 30)
    logger.info(f"atb_model config:\n{atb_model.atb_model_config.to_dict()}\n")

    model_name = os.path.splitext(os.path.basename(output_file))[0]
    input_info = "input_ids=torch.arange(input_len)"
    if FIXED_INPUTS.position_ids in atb_model.inputs:
        input_info += ", position_ids=torch.arange(input_len)"
    logger.info("=" * 30)
    logger.info(
        f"""Run simple inference like:

    python3 -c "
    import torch, torch_npu
    import {model_name}
    from msit_llm.transform.torch_to_atb_python import ATBModel

    atb_model = ATBModel({model_name}.Model())
    weights = torch.load(\'$WEIGHT_PATH\')  # Use actual WEIGHT_PATH
    atb_model.set_weights(weights)

    input_len = 32
    out = atb_model.forward({input_info})
    print(out)
    "
    """.replace(
            " " * 4, ""
        )
    )

    infer_file = generate_infer_file(output_file, source_path, is_vl_model=is_vl_model)
    logger.info("=" * 30)
    logger.info(f"End-to-end inference example saved to: {infer_file}")
    logger.info(f"Execute by: python {infer_file}\n")
