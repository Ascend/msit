# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import absolute_import, division, print_function

import re
import os
import gc
import functools
from collections import defaultdict

from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from safetensors.torch import save_file

from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from ascend_utils.common.security import (get_valid_write_path, SafeWriteUmask, check_element_type,
                                          check_type, check_dict_element, get_write_directory, check_number, check_int)
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    NormBias, extract_dag, input_to_cpu, norm_class_detect, class_detect
)
# KIA part
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_outlier import deepcopy_model
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer, LinearQuantizer, layer_wise_calib
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize_save, get_features, linear_quantization_params, fully_analyze_activation
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.kv_cache_utils import (
    set_kvcache_vari_func, new_forward
)
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers
from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
    quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit
from msmodelslim.pytorch.lowbit.calibration import (
    replace_RMSNorm, QuantXDecoderLayer, LlamaRMSNormBias
)
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.lowbit.quant_modules import Quantizer as LowBitQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, WeightQuantMethod
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import (
    SAVE_TYPE_LIST,
    SAVE_TYPE_NUMPY,
    SAVE_TYPE_SAFE_TENSOR,
)

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    configure_fa,
    enable_fa_calibration,
    disable_fa_calibration,
    enable_fa_quantizer_record,
    collect_fa_quantizer_record,
    export_fa_quant_params,
    is_attn_module_and_then_check_quantizer
)

from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save_utils import save_file_partial

HF_HOOK = "_hf_hook"


class Calibrator(object):
    """ Calibrator for post-training quantization."""

    def __init__(self, model,
                 cfg: QuantConfig,
                 calib_data=None,
                 disable_level='L0',
                 all_tensors=None):
        check_type(model, nn.Module, param_name="model")
        check_type(cfg, QuantConfig, param_name="cfg")
        check_type(disable_level, str, param_name='disable_level')
        if all_tensors:
            check_element_type(all_tensors, torch.Tensor, dict, param_name="all_tensors")

        self.cfg = cfg
        self.logger = msmodelslim_logger
        self.calib_data = self.get_calib_data([]) if calib_data is None else self.get_calib_data(calib_data)
        self.use_kvcache_quant = cfg.use_kvcache_quant
        self.norm_class_name = cfg.norm_class_name

        if model.dtype != model.config.torch_dtype:
            self.logger.warning(f'The model dtype {model.dtype} is not consistent with the model.config.torch_dtype '
                                f'{model.config.torch_dtype}. The model will be regarded as {model.config.torch_dtype}'
                                f' type in subsequent process.')

        self.quant_param_dict = {}
        # 记录被量化module名称，相关的scale、offset等参数名称 key:weight的名称， value:scale、offset等参数的名称
        self.quantized_module_param_dict = defaultdict(list)
        self.fa_module_param_dict = defaultdict(list)

        self.init_model_accelerate(model)
        # 初始化dag类
        self.dag = self.extract_dag(model)
        # 初始化kvcache类
        self.attention_class = None
        if self.use_kvcache_quant:
            self.get_kvcache(model)
            self.attention_class = class_detect(model, 'attention')
            self.transformer_class = class_detect(model, 'GLMTransformer')

        # 记录浮点模型权重
        self.ori_fp_weight = self.get_ori_model_weight(model)

        if self.cfg.do_smooth:
            replace_RMSNorm(model)
        
        if not hasattr(self.cfg, 'use_fa_quant'):
            self.cfg.use_fa_quant = False
            self.cfg.fa_tp_size = 1
            self.cfg.fa_amp = 0

        # 初始化模型权重json描述
        self.quant_model_json_description = QuantModelJsonDescription(self.cfg.model_quant_type, 
                                                                      self.cfg.use_kvcache_quant,
                                                                      self.cfg.use_fa_quant)
        if not re.match(r'^L((?!0)\d+|0)$', disable_level):
            raise ValueError('Please check the `disable_level` configuration.')
        self.disable_level = disable_level

        model = self.init_model_device(model)
        self.last_layer_name = None
        self.rollback_names = None
        self.act_states = None
        self.rollback_names_process(model)
        if self.cfg.use_fa_quant:
            configure_fa(model, tp_size=self.cfg.fa_tp_size)

        if self.cfg.calib_mode == 1:
            if (self.cfg.a_bit <= 8) or self.cfg.w_hessian:
                self.all_tensors = all_tensors
            else:
                self.all_tensors = None

        try:
            self.model = self.quantize(model)
            if self.calib_data:
                self.enable_quant()
        except Exception as e:
            raise Exception("Please check the model and configuration.", e) from e
        self.logger.info("Quantizer initialized successful!")

    def init_model_accelerate(self, model):
        try:
            self.model_with_accelerate = judge_model_with_accelerate(model)
        except Exception as e:
            raise Exception("Please check the model and configuration.", e) from e
        if not self.model_with_accelerate:
            self.device_org = next(model.parameters()).device
        else:
            self.device_org = None

    def init_model_device(self, model):
        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device
        if not same_device:
            self.logger.warning("Model is not on the deivce indicated in `QuantConfig`, "
                                "Model is on the device `{}` while `QuantConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transfering model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `QuantConfig` indicates.")
        return model

    def init_dag(self):
        if self.cfg.do_smooth:
            return True
        if hasattr(self.cfg, 'tp_size'):
            return True
        if self.cfg.use_kvcache_quant:
            return True
        return False

    def get_ori_model_weight(self, model):
        if hasattr(model, 'ori_state_dict'):
            ori_fp_weight = getattr(model, 'ori_state_dict')
        else:
            ori_fp_weight = {}
            for key, value in model.state_dict().items():
                if not isinstance(value, torch.Tensor):
                    self.logger.warning("The original float weight[{key}]is not torch.Tensor, "
                                        "it won't be saved, may raise error.")
                    continue
                ori_fp_weight[key] = value.cpu()
        check_dict_element(ori_fp_weight, value_type=torch.Tensor, param_name='ori_fp_weight',
                           additional_msg="Failed to get original float weight, please check the model.")
        return ori_fp_weight

    def extract_dag(self, model):
        if not self.init_dag():
            return None
        if not self.calib_data:
            dummy_input = torch.randint(0, 100, (1, 128)).type(torch.int64)
        else:
            dummy_input = self.calib_data[0]
        norm_class = self.get_norm_class(model, norm_class_name=self.norm_class_name)
        dag = TorchDAGAdapter(model, dummy_input, hook_nodes=norm_class)
        return dag

    def get_kvcache(self, model):
        self.kv_sym = True if not hasattr(self.cfg, 'kv_sym') else self.cfg.kv_sym
        kv_linears, num_kv = self.dag.get_kv_linears()
        self.kv_cache = self.get_kvcache_features(model, kv_linears, num_kv)
        self.get_kvcache_quant_param(num_kv)

    def get_kvcache_features(self, model, kv_linears, num_kv):
        model.eval()
        kv_cache = {}

        def update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, chunk_size):
            comming_max, comming_min = max_min[0], max_min[1]
            # 分块操作
            if chunk_size == 2:
                key_max, value_max = torch.chunk(comming_max, chunk_size, dim=0)
                key_min, value_min = torch.chunk(comming_min, chunk_size, dim=0)
            elif chunk_size == 3:
                res_dim = comming_max.shape[-1] - in_hidden_size

                _, kv_max = torch.split(comming_max, [in_hidden_size, res_dim])
                _, kv_min = torch.split(comming_min, [in_hidden_size, res_dim])
                key_max, value_max = torch.chunk(kv_max, 2, dim=0)
                key_min, value_min = torch.chunk(kv_min, 2, dim=0)

            # 获取k_name和v_name
            k_name = 'k_proj'
            v_name = 'v_proj'

            # 更新key极值
            update_extremum(kv_cache[name], k_name, 'max', torch.max, key_max)
            update_extremum(kv_cache[name], k_name, 'min', torch.min, key_min)
            # 更新value极值
            update_extremum(kv_cache[name], v_name, 'max', torch.max, value_max)
            update_extremum(kv_cache[name], v_name, 'min', torch.min, value_min)

        def update_extremum(kv_cache, name, key, torch_function, value):
            if name not in kv_cache:
                kv_cache[name] = {}
                
            if key in kv_cache[name]:
                kv_cache[name][key] = torch_function(kv_cache[name][key], value)
            else:
                kv_cache[name][key] = value

        def kv_tensor(name, in_hidden_size, tensor):
            if name not in kv_cache:
                kv_cache[name] = {}

            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).detach()
            comming_max = torch.max(tensor, dim=0)[0]
            comming_min = torch.min(tensor, dim=0)[0]
            max_min = [comming_max, comming_min]

            if num_kv == 2:
                # 更新最大值
                update_extremum(kv_cache, name, 'max', torch.max, comming_max)
                # 更新最小值
                update_extremum(kv_cache, name, 'min', torch.min, comming_min)
            elif num_kv == 1:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 2)
            elif num_kv == 0:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 3)

        def kv_cache_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            kv_tensor(name, x.shape[-1], y)
        
        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if name in kv_linears:
                    hooks.append(
                        m.register_forward_hook(
                            functools.partial(kv_cache_hook, name=name))
                    )

        for data in tqdm(self.calib_data):
            if isinstance(data, tuple) or isinstance(data, list):
                with torch.no_grad():
                    model(*data)
            elif isinstance(data, dict):
                with torch.no_grad():
                    model(**data)
        for h in hooks:
            h.remove()

        return kv_cache

    def get_kvcache_quant_param(self, num_kv):
        if num_kv == 2:
            for key in self.kv_cache.keys():
                self.get_quant_module_param(key, key, self.kv_cache[key])
        else:
            for key in self.kv_cache.keys():
                for kv_key in self.kv_cache[key].keys():
                    new_key = key + '.' + kv_key
                    self.get_quant_module_param(key, new_key, self.kv_cache[key][kv_key])

    def get_quant_module_param(self, key, new_key, kv_cache):
        key_weight = key + '.weight'
        new_key_scale, new_key_offset = new_key + '.kv_cache_scale', new_key + '.kv_cache_offset'

        scale, zero_point = linear_quantization_params(8, kv_cache['min'], kv_cache['max'],
                                                       integral_zero_point=True, q_signed=True, sym=self.kv_sym)

        self.quant_param_dict[new_key_scale] = scale.to('cpu')
        self.quant_param_dict[new_key_offset] = zero_point.to('cpu')
        self.quantized_module_param_dict[key_weight].append(new_key_scale)
        self.quantized_module_param_dict[key_weight].append(new_key_offset)

    def rollback_names_process(self, model):
        # 自动回退lm_head层
        quant_name_list = []
        conv_name_list = []
        for name, module in list(model.named_modules()):
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                quant_name_list.append(name)
            elif isinstance(module, nn.Conv2d):
                conv_name_list.append(name)
        if quant_name_list:
            last_layer_name = quant_name_list[-1]
        else:
            raise ValueError("No nn.Linear found in the model, please check the model.")
        if conv_name_list:
            self.logger.info("conv2d is in the model and will not be quantified")
        # 校验用户指定的回退层是否存在
        for name in self.cfg.disable_names:
            if name not in quant_name_list and name not in conv_name_list:
                raise ValueError(f"`disable_names` has invalid key `{name}`, please check your model configurations.")
        if self.cfg.disable_last_linear:
            self.logger.info(f"Automatically disabling the last linear layer: {last_layer_name} "
                             f"based on the `disable_last_linear` parameter setting.")
            self.rollback_names = list(set(self.cfg.disable_names + [last_layer_name] + conv_name_list))
        else:
            self.rollback_names = list(set(self.cfg.disable_names + conv_name_list))

        # 如果使用low bit稀疏量化，且disable_level不为L0，则使用amp自动回退
        if self.cfg.is_lowbit and self.disable_level != 'L0':
            self.cfg.use_amp = True
            self.logger.info("Automatically disabling layer set while calibrating "
                             "based on the `use_amp` parameter setting.")
            self.cfg.amp_num = int(self.disable_level[1:])
        # 根据disable level获取自动回退层，Data-Free场景下自动回退设为L0
        elif self.calib_data:
            enable_tensor_dump = False  # 模型规模较大的时候，dump_tensor非常消耗计算、内存和存储空间，默认关闭
            try:
                self.act_states = get_features(model, self.calib_data[:5], "features.npy", enable_tensor_dump)
            except Exception as e:
                raise Exception("Please check the model and calibration data, "
                                "ensure that your model can run with `model(*(calib_data[0]))`.", e) from e

            label_threshold_dict = self.get_label_threshold_dict()
            self.logger.info(f"The model contains a total of {len(label_threshold_dict)} nn.Linear layers.")
            auto_disable_names = self.get_auto_disable_names(label_threshold_dict)
            self.rollback_names = list(set(self.rollback_names + auto_disable_names))

            if self.disable_level != 'L0':
                self.logger.info('Automatically disabled layer names are:\n\t' +
                                 '\n\t'.join([str(name) for name in sorted(auto_disable_names)]))
        # for Data-Free
        else:
            self.logger.info("Running in Data-Free mode, `disable_level` set to `L0`")
        self.logger.info('The subsequent layers will maintain the use of floating-point weights'
                         ' for forward computations.:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(self.rollback_names)]))

    def get_label_threshold_dict(self):
        label_threshold_dict = {}
        for key in self.act_states.keys():
            abs_max_tensor = max(abs(self.act_states[key]['t_max']), abs(self.act_states[key]['t_min']))
            range_param = abs_max_tensor / self.act_states[key]['std']
            label_threshold_dict[key] = range_param.item()
        return label_threshold_dict

    def get_auto_disable_names(self, label_threshold_dict):
        disable_level_number = int(self.disable_level[1:])
        if self.disable_level == 'L0':
            auto_disable_names = []
        else:
            sorted_label_list = list(dict(sorted(label_threshold_dict.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)).keys())
            auto_disable_names = [label for label in sorted_label_list
                                  if label not in self.rollback_names][:disable_level_number]
        return auto_disable_names

    def quantize(self, model):
        if self.model_with_accelerate:
            return self.quantize_model(model)
        else:
            return self.quantize_model(model).to(self.cfg.device)

    def enable_quant(self):
        enable_quantization(self.model, self.act_states, self.logger, self.cfg.use_fa_quant)

    def get_calib_data(self, calib_data):
        check_type(calib_data, list, param_name='calib_data')
        if not calib_data:
            QuantType.check_datafree_quant_type(self.cfg.model_quant_type)
            WeightQuantMethod.check_datafree_wmethod(self.cfg.w_method)
        else:
            self.check_calib_data(calib_data)
        return calib_data

    def check_calib_data(self, calib_data):
        element_not_tensor = False
        for i, calib_data_item in enumerate(calib_data):
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            for _, item in enumerate(calib_data_item):
                if not isinstance(item, torch.Tensor):
                    element_not_tensor = True
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")

    def run(self, int_infer=False):
        check_type(int_infer, bool, additional_msg="`int_infer` should be boolean type!")

        try:
            self._run(int_infer=int_infer)
        except Exception as ex:
            raise Exception("Please check the model and configuration.", ex) from ex

    def save(self, output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None):
        check_type(output_path, str, param_name="output_path")
        if part_file_size is not None:
            check_int(part_file_size, min_value=1)
        if not save_type:
            save_type = [SAVE_TYPE_NUMPY]
        check_element_type(save_type, element_type=str, value_type=list, param_name="save_type")
        if [False for save_type_item in save_type if save_type_item not in SAVE_TYPE_LIST]:
            self.logger.warning(
                f"`save_type` should be one of the choices in {SAVE_TYPE_LIST}, but received {save_type}. "
                f"Defaulting to `{SAVE_TYPE_NUMPY}` type."
            )
            save_type = [SAVE_TYPE_NUMPY]
        output_path = get_write_directory(output_path, write_mode=0o750)
        self.get_quant_params()
        if SAVE_TYPE_NUMPY in save_type:
            self.save_npy(output_path)
        if SAVE_TYPE_SAFE_TENSOR in save_type:
            if not isinstance(safetensors_name, str):
                default_safetensors_name = f"quant_model_weight_{self.cfg.model_quant_type.lower()}.safetensors"
                self.logger.warning(f"invalid `safetensors_name`, defaulting to `{default_safetensors_name}`")
                safetensors_name = default_safetensors_name
            if not isinstance(json_name, str):
                default_json_name = f"quant_model_description_{self.cfg.model_quant_type.lower()}.json"
                self.logger.warning(f"invalid `json_name`, defaulting to `{default_json_name}`")
                json_name = default_json_name
            self.save_safetensor(output_path, safetensors_name, json_name, part_file_size)

    def save_safetensor(self, output_path, safetensors_name, json_name, part_file_size):
        """
        基于浮点、量化两份独立权重，存储完整的量化、浮点混合权重，用户仅需加载一个混合权重即可
        """
        quant_model_weight_path = os.path.join(output_path, safetensors_name)
        quant_model_description_path = os.path.join(output_path, json_name)
        quant_model_weight_path = get_valid_write_path(quant_model_weight_path, extensions=[".safetensors"])
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])

        # 修改 safetensor 权重
        safetensor_weight = {}
        quant_model_state_dict_list = list(self.quant_param_dict.keys())
        for ori_model_state_dict_name, ori_model_state_dict in self.ori_fp_weight.items():
            # 如果浮点权重名称不在量化权重名称中，说明是浮点独有的权重，需要把浮点权重加入safetensor_weight
            if ori_model_state_dict_name not in quant_model_state_dict_list:
                # Norm 有额外的anti weight、bias，单独补充
                self.set_fp_safetensor(ori_model_state_dict_name, safetensor_weight, ori_model_state_dict)

            # 如果浮点权重名称在量化权重名称中，说明是浮点转换为量化的权重，需要把量化权重加入safetensor_weight
            else:
                self.set_quant_safetensor(ori_model_state_dict_name, safetensor_weight)
        
        if self.cfg.use_fa_quant:
            for attention_module_name in self.fa_module_param_dict:
                self.set_fa_quant_safetensor(attention_module_name, safetensor_weight)
            
        for key, item in safetensor_weight.items():
            safetensor_weight[key] = item.cpu().contiguous()

        self.logger.info("The directory path for the saved safetensors is %s", quant_model_weight_path)
        with SafeWriteUmask(umask=0o377):
            if part_file_size is not None:
                save_file_partial(safetensor_weight, quant_model_weight_path, part_file_size)
            else:
                save_file(safetensor_weight, quant_model_weight_path)
        self.logger.info("Safetensors weight saved successfully!")
        self.quant_model_json_description.save(quant_model_description_path)

    def set_fp_safetensor(self, ori_model_state_dict_name, safetensor_weight, ori_model_state_dict):
        safetensor_weight[ori_model_state_dict_name] = ori_model_state_dict.clone()
        self.quant_model_json_description.change_weight_type(ori_model_state_dict_name, QuantType.FLOAT)
        if ori_model_state_dict_name in self.quantized_module_param_dict:
            for quant_param_name in self.quantized_module_param_dict[ori_model_state_dict_name]:
                safetensor_weight[quant_param_name] = self.quant_param_dict.get(quant_param_name)
                self.quant_model_json_description.change_weight_type(
                    quant_param_name, self.quant_model_json_description.model_quant_type)

    def set_quant_safetensor(self, ori_model_state_dict_name, safetensor_weight):
        safetensor_weight[ori_model_state_dict_name] = self.quant_param_dict.get(ori_model_state_dict_name)
        self.quant_model_json_description.change_weight_type(
            ori_model_state_dict_name,
            self.quant_model_json_description.model_quant_type)
        # 将该量化linear的附属参数，scale、offset 等加入safetensor_weight
        for quant_param_name in self.quantized_module_param_dict.get(ori_model_state_dict_name):
            safetensor_weight[quant_param_name] = self.quant_param_dict.get(quant_param_name)
            self.quant_model_json_description.change_weight_type(
                quant_param_name, self.quant_model_json_description.model_quant_type)

    def set_fa_quant_safetensor(self, attention_module_name, safetensor_weight):
        for quant_param_name in self.fa_module_param_dict.get(attention_module_name):
            quant_param = self.quant_param_dict.get(quant_param_name)
            if quant_param is not None:
                safetensor_weight[quant_param_name] = quant_param
                self.quant_model_json_description.change_weight_type(quant_param_name, QuantType.FAQuant)
            else:
                self.quant_model_json_description.change_weight_type(quant_param_name, QuantType.FLOAT)


    def save_npy(self, output_path):
        quant_weight_dict = {}
        scale_dict = {}
        offset_dict = {}
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            deq_scale_dict = {}
            quant_bias_dict = {}
        if self.use_kvcache_quant:
            kv_cache_scale = {}
            kv_cache_offset = {}
        if self.cfg.use_fa_quant:
            fa_quant_scale = {}
            fa_quant_offset = {}

        for name, module in self.model.named_modules():
            weight_name = name + '.weight'
            if isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
                quant_weight_dict[name] = self.quant_param_dict.get(weight_name)
                if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
                    quant_bias_dict[name] = self.quant_param_dict.get(name + '.quant_bias')
                    deq_scale_dict[name] = self.quant_param_dict.get(name + '.deq_scale')
                    scale_dict[name] = self.quant_param_dict.get(name + '.input_scale')
                    offset_dict[name] = self.quant_param_dict.get(name + '.input_offset')
                if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
                    scale_dict[name] = self.quant_param_dict.get(name + '.weight_scale')
                    offset_dict[name] = self.quant_param_dict.get(name + '.weight_offset')
            if self.use_kvcache_quant:
                self.get_kvcache_quant_params(weight_name, kv_cache_scale, kv_cache_offset)
            if self.cfg.use_fa_quant and is_attn_module_and_then_check_quantizer(module, name):
                quant_params_scale, quant_params_offset, _ = export_fa_quant_params(module, name)
                fa_quant_scale.update(quant_params_scale)
                fa_quant_offset.update(quant_params_offset)

        self.save_param(output_path, "quant_weight.npy", quant_weight_dict)
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            self.save_param(output_path, "input_scale.npy", scale_dict)
            self.save_param(output_path, "input_offset.npy", offset_dict)
            self.save_param(output_path, "quant_bias.npy", quant_bias_dict)
            self.save_param(output_path, "deq_scale.npy", deq_scale_dict)
        if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
            self.save_param(output_path, "weight_scale.npy", scale_dict)
            self.save_param(output_path, "weight_offset.npy", offset_dict)
        if self.use_kvcache_quant:
            self.save_param(output_path, "kv_cache_scale.npy", kv_cache_scale)
            self.save_param(output_path, "kv_cache_offset.npy", kv_cache_offset)
        if self.cfg.use_fa_quant:
            self.save_param(output_path, "fa_quant_scale.npy", fa_quant_scale)
            self.save_param(output_path, "fa_quant_offset.npy", fa_quant_offset)

        anti_norm_wb = self.get_anti_fp_weight()
        if anti_norm_wb:
            self.save_param(output_path, "anti_fp_norm.npy", anti_norm_wb)

        self.logger.info("Numpy weight saved successfully!")

    def save_param(self, output_path, output_name, output_file):
        output_path = os.path.join(output_path, output_name)
        self.logger.debug("The directory path for the quant param is %s ", output_path)
        output_path = get_valid_write_path(output_path)
        with SafeWriteUmask(umask=0o377):
            np.save(output_path, output_file)

    def get_anti_fp_weight(self):
        anti_norm_wb = {}
        for name, module in self.model.named_modules():
            anti_weight_name = name + '.weight'
            anti_bias_name = name + '.bias'
            if isinstance(module, NormBias):
                anti_norm_wb[anti_weight_name] = module.module.weight.cpu()
                anti_norm_wb[anti_bias_name] = module.bias.cpu()
            if isinstance(module, LlamaRMSNormBias):
                anti_norm_wb[anti_weight_name] = module.weight.cpu()
                anti_norm_wb[anti_bias_name] = module.bias.cpu()
        return anti_norm_wb

    def get_kvcache_quant_params(self, weight_name, kv_cache_scale, kv_cache_offset):
        if self.quantized_module_param_dict[weight_name]:
            for param_name in self.quantized_module_param_dict.get(weight_name):
                if 'kv_cache_scale' in param_name:
                    kv_cache_scale[param_name] = self.quant_param_dict.get(param_name)
                elif 'kv_cache_offset' in param_name:
                    kv_cache_offset[param_name] = self.quant_param_dict.get(param_name)

    def get_quant_params(self):
        self.model.eval()
        for name, module in self.model.named_modules():
            if self.cfg.use_fa_quant and is_attn_module_and_then_check_quantizer(module, name):
                quant_param_scale, quant_param_offset, attach_map = export_fa_quant_params(module, name)
                self.quant_param_dict.update(quant_param_scale)
                self.quant_param_dict.update(quant_param_offset)
                self.fa_module_param_dict.update(attach_map)

            if isinstance(module, ParallelLinearCol):
                quant_param, attach_map = module.get_quant_param()
                self.quant_param_dict.update(quant_param)
                self.quantized_module_param_dict.update(attach_map)
            # 处理 Norm 对应的 weight、bias
            if isinstance(module, (NormBias, LlamaRMSNormBias)):
                anti_norm_weight = module.module.weight.cpu() if isinstance(module, NormBias) else module.weight.cpu()
                anti_norm_bias = module.bias.cpu()
                anti_norm_name_weight = name + '.module.weight'
                anti_norm_name_bias = name + '.module.bias'
                self.quant_param_dict[anti_norm_name_weight] = anti_norm_weight.clone().detach()
                self.quant_param_dict[anti_norm_name_bias] = anti_norm_bias.clone().detach()
                self.quantized_module_param_dict[name + '.weight'] = [anti_norm_name_weight, anti_norm_name_bias]

            # 处理Linear、以及附属scale、offset等params
            if isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
                if not module.quant_weight.is_enable:
                    continue

                quant_weight, fp_weight, weight_scale, weight_offset = self.get_param_from_quantizer(module)
                if quant_weight is None:
                    continue
                # 各种量化均需要提供 weight
                quant_weight = quant_weight.cpu()
                self.quant_param_dict[name + '.weight'] = quant_weight.to(torch.int8)

                # W4A16/W8A16 需要提供 weight_scale、weight_offset
                if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
                    self.quant_param_dict[name + '.weight_scale'] = weight_scale
                    self.quant_param_dict[name + '.weight_offset'] = weight_offset
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.weight_scale')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.weight_offset')
                # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
                if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
                    input_scale = module.quant_input.input_scale.cpu()
                    input_offset = module.quant_input.input_offset.cpu()
                    self.quant_param_dict[name + '.input_scale'] = input_scale
                    self.quant_param_dict[name + '.input_offset'] = input_offset
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.input_scale')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.input_offset')
                    deq_scale = self.deqscale_process(input_scale, weight_scale).to(torch.float32)
                    correction = (quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)).cpu()
                    fp_bias = self.change_bias(fp_weight, module)
                    quant_bias = torch.round(fp_bias / deq_scale - correction)
                    deq_scale = deqscale2int64_by_dtype(deq_scale, self.model.config.torch_dtype == torch.bfloat16)

                    self.quant_param_dict[name + '.quant_bias'] = quant_bias.cpu().to(torch.int32)
                    self.quant_param_dict[name + '.deq_scale'] = deq_scale.cpu()
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.quant_bias')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.deq_scale')

        if hasattr(self.cfg, 'tp_size'):
            self.concat_simulate_linear()

    def concat_simulate_linear(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ParallelLinearCol):
                if name in self.rollback_names:
                    continue
                if module.cfg.model_quant_type == QuantType.FLOAT:
                    continue
                concat_weight_list = []
                for tp_index in range(module.cfg.tp_size):
                    concat_name = '.'.join([name, f'tp_list', str(tp_index), 'weight'])
                    concat_weight_list.append(self.quant_param_dict.get(concat_name))
                concat_weight = torch.cat(concat_weight_list, dim=-1)
                self.quant_param_dict[name + '.weight'] = concat_weight

    def get_param_from_quantizer(self, module):
        quant_weight = None
        fp_weight, device, weight_scale, weight_offset, round_opt = self._get_module_quant_input(module)
        if isinstance(module, LinearQuantizer):
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=8,
                                                 round_opt=round_opt, device=device)
        if isinstance(module, LinearSparseQuantizer):
            _, _, quant_weight, _ = quant_one_weight_by_outliers(
                fp_weight, powerquant=self.cfg.nonuniform, fraction=self.cfg.fraction, num_bits=self.cfg.w_bit,
                per_channel=not self.cfg.mm_tensor)
        if isinstance(module, LowBitLinearQuantizer):
            fp_weight = module.fp_weight
            if module.disable_input:
                res = None, fp_weight, weight_scale, weight_offset
                return res
            if self.cfg.model_quant_type == QuantType.W8A8S:
                bit = 8
            else:
                bit = self.cfg.w_bit
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=bit,
                                                 round_opt=round_opt, device=module.weight.device,
                                                 group_size=self.cfg.group_size)
        res = quant_weight, fp_weight, weight_scale, weight_offset
        return res

    def deqscale_process(self, input_scale, scale):
        deq_scale = input_scale * scale
        if deq_scale.ndim > 1:
            deq_scale = deq_scale.squeeze(1)
        deq_scale = deq_scale.cpu()
        return deq_scale

    def change_bias(self, fp_weight, module):
        if module.bias is None:
            bias_shape = fp_weight.shape[0]
            fp_bias = torch.zeros(bias_shape)
        else:
            fp_bias = module.bias.cpu()
        return fp_bias

    def run_calib_mode(self):
        amp_done = False
        for data in tqdm(self.calib_data):
            if not amp_done and self.cfg.fa_amp:
                enable_fa_quantizer_record(self.model)

            if isinstance(data, tuple) or isinstance(data, list):
                self.model(*data)
            elif isinstance(data, dict):
                self.model(**data)

            if not amp_done and self.cfg.use_amp:
                self.run_amp()
                amp_done = True
            if not amp_done and self.cfg.fa_amp:
                self.run_fa_amp()
                amp_done = True
        self.run_datafree_after_calib()

    def run_datafree_after_calib(self):
        # 将稀疏模型校准集没有运行的部分切换成data-free模式
        for name, module in self.model.named_modules():
            if isinstance(module, LinearQuantizer) and module.quant_weight.weight_scale is None:
                if module.quant_weight.w_hessian:
                    module.quant_weight.w_hessian = False
                    self.logger.info(f"run MIN-MAX quantization on linear layer: {name}")
                module.quant_weight(module.weight)


    def run_amp(self):
        max_input_dict = {}

        for name, module in self.model.named_modules():
            if isinstance(module, LowBitLinearQuantizer):
                max_input_dict[name] = module.max_input_num
        layers_to_disable = list(
            dict(sorted(max_input_dict.items(),
                        key=lambda x: x[1], reverse=True)).keys()
        )[:self.cfg.amp_num]
        for name, module in self.model.named_modules():
            if name in layers_to_disable:
                module.disable_input = True
                module.disable_quant_weight()
        self.rollback_names.extend(layers_to_disable)
        self.logger.info('The following linear layers will continue to use floating-point weights for forward computation:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(self.rollback_names)]))

    def run_fa_amp(self):
        qkv_states_record = collect_fa_quantizer_record(self.model)
        states_num_per_layer = 3
        model_attention_layer_num = len(qkv_states_record.keys()) // states_num_per_layer
        if model_attention_layer_num < self.cfg.fa_amp:
            self.logger.warning("`fa_amp` exceeds the total attention layer number. Therefore, only up to the total attention layers will skip quantization")
        disabled_module_names = fully_analyze_activation(qkv_states_record, self.cfg.fa_amp)
        self.logger.info('The following attention layers will continue to use floating-point weights for forward computation:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(disabled_module_names)]))
        for name, module in self.model.named_modules():
            if is_attn_module_and_then_check_quantizer(module, name) and name in disabled_module_names:
                module.fa_quantizer.reset()
                module.fa_quantizer.disable_calibration()

    def run_datafree_mode(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LinearQuantizer):
                self.logger.info(f"Running in Data-Free mode, quantizing the layer: {name}")
                module.quant_weight(module.weight)
            elif isinstance(module, LowBitLinearQuantizer):
                with torch.no_grad():
                    module.fp_weight = module.weight.cpu().clone()
                    self.logger.info(f"Running in Data-Free mode, quantizing the layer: {name}")
                    recovered_weight, lowbit_weight_scale, _, lowbit_weight_offset = \
                        quant_one_weight_by_outliers_low_bit(
                            module.weight,
                            powerquant=self.cfg.nonuniform,
                            fraction=self.cfg.fraction,
                            num_bits=self.cfg.w_bit,
                            isolate_outlier_amax=False,
                            per_channel=not self.cfg.mm_tensor,
                            use_cuda=True if self.cfg.dev_type == 'gpu' else False,
                            use_sigma=self.cfg.use_sigma,
                            sigma_factor=self.cfg.sigma_factor,
                            open_outlier=self.cfg.open_outlier,
                            group_size=self.cfg.group_size,
                            w_sym=self.cfg.w_sym,
                            use_hqq=self.cfg.hqq
                        )
                    # 为降低显存占用，把部分权重放到cpu上
                    module.quant_weight.weight_scale = lowbit_weight_scale.cpu()
                    # 通过深拷贝的方式降低显存占用
                    module.weight[:] = recovered_weight[:]
                    module.quant_weight.weight_offset = lowbit_weight_offset.cpu()

                module.weight_quant_flag = True

    def quantize_model(self, model):
        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        if self.cfg.do_smooth:
            list_infos = self.dag.get_llm_network_pattern_auto()
            attn_list, mhsa_ln_list, ffn_list, ffn_ln_list = list_infos[0], list_infos[2], list_infos[3], list_infos[4]

            if not (len(attn_list) == len(mhsa_ln_list) == len(ffn_list) == len(ffn_ln_list)):
                raise ValueError("Failed to get network pattern by DAG, please check the model!")
            for _, (attn, mhsa_ln, ffn, ffn_ln) in enumerate(zip(attn_list, mhsa_ln_list, ffn_list, ffn_ln_list)):
                modules = [attn, mhsa_ln, ffn, ffn_ln]
                cur_decoder, cur_decoder_type, modules_info, cur_decoder_name = \
                    self.split_module_name_get_info(modules, model, self.cfg.down_proj_type)
                decoder_info = {
                    'attn': modules_info[0],
                    'mlp': modules_info[2],
                    'ln1': modules_info[1],
                    'ln2': modules_info[3],
                    'qkv': modules_info[4],
                    'ffn': modules_info[5]
                }
                quant_mod = QuantXDecoderLayer(cur_decoder, self.cfg, cur_decoder_name, decoder_info)
                if hasattr(cur_decoder, HF_HOOK):
                    add_hook_to_module(quant_mod, cur_decoder._hf_hook)
                    remove_hook_from_module(cur_decoder)
                _set_module(model, cur_decoder_name, quant_mod)

        for name, mod in model.named_modules():
            if name in self.rollback_names:
                continue
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear):
                if self.cfg.is_lowbit:
                    quant_mod = LowBitLinearQuantizer(cfg=self.cfg, logger=self.logger, name=name)
                elif self.cfg.model_quant_type is not QuantType.W8A8S:
                    quant_mod = LinearQuantizer(cfg=self.cfg, logger=self.logger)
                else:
                    quant_mod = LinearSparseQuantizer(cfg=self.cfg, logger=self.logger)
                quant_mod.set_param(mod)

                # 拷贝accelerate定义的hook
                if hasattr(mod, HF_HOOK):
                    add_hook_to_module(quant_mod, mod._hf_hook)
                    remove_hook_from_module(mod)

                _set_module(model, name, quant_mod)
                del mod

        if hasattr(self.cfg, 'tp_size'):
            simulate_linear = self.dag.get_allreduce_linear()
            self.logger.info(
                'The following layers will be quant by simulating tp:\n\t'
                + '\n\t'.join([str(name) for name in sorted(simulate_linear)]))
        else:
            simulate_linear = []

        for name, mod in model.named_modules():
            if name in simulate_linear:
                tp_mod = ParallelLinearCol()
                tp_mod.set_param(mod, name, cfg=self.cfg)
                if name in self.rollback_names:
                    tp_mod.quant_type = QuantType.FLOAT
                else:
                    tp_mod.quant_type = self.cfg.model_quant_type
                if hasattr(mod, HF_HOOK):
                    add_hook_to_module(tp_mod, mod._hf_hook)
                    remove_hook_from_module(mod)

                _set_module(model, name, tp_mod)
                del mod

        gc.collect()

        return model

    # 获取decoder相关信息
    def split_module_name_get_info(self, module_names, model, down_proj_type):
        item0 = module_names[0][0] if isinstance(module_names[0], list) else module_names[0]
        item2 = module_names[2][0] if isinstance(module_names[2], list) else module_names[2]
        module_names_ = [item0, module_names[1], item2, module_names[3]]

        qkv_item = module_names[0] if isinstance(module_names[0], list) else [module_names[0]]
        ffn_item = module_names[2] if isinstance(module_names[2], list) else [module_names[2]]

        qkv_tokens = [item.split('.') for item in qkv_item]
        qkv_list = [item[-1] for item in qkv_tokens]

        ffn_tokens = [item.split('.') for item in ffn_item]
        ffn_list = [item[-1] for item in ffn_tokens if item[-1] not in down_proj_type]

        tokens = [module_names_[i].split('.') for i in range(len(module_names_))]
        cur_mod = model
        layername = ''
        for i, item in enumerate(tokens[0]):
            layername += item + '.'
            cur_mod = getattr(cur_mod, item)
            if item.isnumeric():
                module_infos = [token[i + 1] for token in tokens]
                module_infos.append(qkv_list)
                module_infos.append(ffn_list)
                break
        mod_type = cur_mod.__class__.__name__

        ret = cur_mod, mod_type, module_infos, layername[:-1]
        return ret

    def get_norm_class(self, model, norm_class_name=None):
        if norm_class_name is not None:
            norm_class = list(set([m.__class__ for m in model.modules() \
                                   if norm_class_name.lower() in m.__class__.__name__.lower()]))
        else:
            norm_class = list(set([m.__class__ for m in model.modules() if "norm" in m.__class__.__name__.lower()]))
            if len(norm_class) != 1:
                raise ValueError("No customized normalization class detected! Please check the model and configuration")
        return norm_class

    def enable_kvcache_fake_quantization(self):
        num_layers = getattr(self.model.config, 'num_layers', None) \
            if hasattr(self.model, 'config') else None
        for name, mod in self.model.named_modules():
            # set the variable and function of kvcache in attention class
            if self.attention_class and isinstance(mod, self.attention_class):
                cache_sub_dict = {key: value for key, value in self.quant_param_dict.items() if name in key}
                set_kvcache_vari_func(mod, cache_sub_dict, self.cfg, num_layers=num_layers)

                setattr(mod, 'original_forward', mod.forward)
                setattr(mod, 'forward', new_forward.__get__(mod, mod.__class__))

    def _get_module_quant_input(self, module):
        fp_weight = module.weight.cpu()
        weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
        device = weight_scale.device if weight_scale is not None else None
        scale = weight_scale.cpu() if weight_scale is not None else None
        offset = weight_offset.cpu() if weight_offset is not None else None
        round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt
        ret = fp_weight, device, scale, offset, round_opt
        return ret

    def _run(self, calib_amp=5, int_infer=False):
        if not isinstance(calib_amp, int) or calib_amp < 1:
            raise TypeError("`calib_amp` should be an integer greater than 0 and not exceeding the "
                            "length of the calibration data. Please check the value.")

        self.logger.info("Calibration start!")
        self.model.eval()
        if self.calib_data:
            if self.cfg.calib_mode == 0:
                with torch.no_grad():
                    self.run_calib_mode()
            elif self.cfg.calib_mode == 1:
                try:
                    layer_wise_calib(self.model, self.all_tensors, self.cfg.device)
                except Exception as e:
                    raise Exception("Please check the model and configuration.", e) from e
                del self.all_tensors
            else:
                raise ValueError("Calibration mode not supported!")
        else:
            self.run_datafree_mode()

        self.logger.info("Calibration end!")

        disable_calibration(self.model, self.logger, use_fa_quant=self.cfg.use_fa_quant)
        if self.use_kvcache_quant:
            self.enable_kvcache_fake_quantization()
        if self.cfg.a_bit != 8 and int_infer:
            self.logger.warning("`int_infer` works only in W8A8 case. Defaulting to false")
            int_infer = False
        if int_infer:
            enable_int_infer(self.model, self.logger)


def load_backup_model(backup_model: nn.Module, device=None, model_with_accelerate: bool = False):
    # 如果原model带有accelerate的hook，则这些hook也会被拷贝并且这些hook指向的还是model而非backup_model
    # 1. 修改hook的指向
    # 2. 将backup_model的权重放在正确的设备上

    if model_with_accelerate:
        for mod in backup_model.modules():
            # 拷贝accelerate定义的hook
            if hasattr(mod, HF_HOOK):
                add_hook_to_module(mod, mod._hf_hook)
    else:
        backup_model.to(device)

    return backup_model


def enable_quantization_by_module(name, module, act_states):
    if '.tp_list' in name:
        states_name = ".".join(name.split(".")[:-3])
    else:
        states_name = ".".join(name.split(".")[:-1])
    if states_name in act_states:
        abs_max_tensor = max(abs(act_states[states_name]["t_max"]), abs(act_states[states_name]["t_min"]))
        range_param = abs_max_tensor / act_states[states_name]["std"]
        module.enable_quantization(name, range_param)
        if module.is_input:
            module.init_act_and_observer(module.cfg)


def enable_quantization(model, act_states, logger=None, use_fa_quant=False):
    _ = logger  # Bypassing not using
    if use_fa_quant:
        enable_fa_calibration(model)
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            enable_quantization_by_module(name, module, act_states)
        if isinstance(module, LowBitQuantizer):
            module.enable_quantization()
            if module.is_input:
                module.init_act_and_observer(module.cfg)
        if isinstance(module, QuantXDecoderLayer):
            module.calibration = True


def disable_calibration(model, logger=None, custom_class=None, use_fa_quant=False):
    _ = logger  # Bypassing not using
    if use_fa_quant:
        disable_fa_calibration(model)
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_calib()
        if custom_class and isinstance(module, custom_class):
            module.disable_calib()
        if isinstance(module, ParallelLinearCol):
            module.disable_calib()


def enable_calibration(model, logger=None, custom_class=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_calib()
    if not custom_class and isinstance(module, custom_class):
        module.enable_calib()


def disable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_int_infer()


def enable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_int_infer()


def set_ratio(model, ratio=0.9, logger=None):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            if logger:
                logger.debug('Set the ratio: %s', name)
            module.set_ratio(ratio)


def judge_model_with_accelerate(model: nn.Module):
    for _, mod in model.named_modules():
        if hasattr(mod, HF_HOOK):
            return True
    return False


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
