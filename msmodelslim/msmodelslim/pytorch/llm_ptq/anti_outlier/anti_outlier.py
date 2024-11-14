#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import absolute_import, division, print_function

import os
import gc
import stat
import copy
import functools
from typing import OrderedDict
from tqdm import tqdm
from tqdm.contrib import tzip

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ascend_utils.common.security import get_valid_write_path, check_type
from msmodelslim import logger as msmodelslim_logger

try:
    import torch_npu
except ImportError:
    msmodelslim_logger.warning("Unable to import torch_npu.")

from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    extract_dag,
    GraphOpt,
    PatternProcess,
    NormBias,
    input_to_cpu,
)
from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import (
    iter_smooth,
    smooth_ln_fcs,
    os_ln_fcs,
    weight_aware,
)

STAT_KEY_MAX = "max"
STAT_KEY_MIN = "min"
STAT_KEY_SHIFT = "shift"
STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
STAT_KEY_THRESHOLD_TENSOR = "thres_t"
STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
STAT_KEY_SMOOTH_SCALE = "smooth_scale"
STAT_KEY_VARIANCE = "std"


def judge_model_with_accelerate(model: nn.Module):
    for _, mod in model.named_modules():
        if hasattr(mod, '_hf_hook'):
            return True
    return False


def model_to_cpu(model):
    if not judge_model_with_accelerate(model):
        model.to("cpu")
        return

    for _, mod in model.named_modules():
        try:
            # npu, cuda -> cpu
            mod.cpu()
        except Exception as e:
            # meta -> cpu
            logger.info("Transfering meta model to cpu device...", e)
            if hasattr(mod, "_hf_hook"):
                mod._hf_hook.detach_hook(mod)


def model_to_org_device_with_buffer(model, device_org='cpu'):
    if not judge_model_with_accelerate(model):
        model.to(device_org)

    # 将原模型的权重恢复到GPU（或meta）上
    for _, mod in model.named_modules():
        if hasattr(mod, '_hf_hook'):
            mod._hf_hook.init_hook(mod)
    device_map = model.hf_device_map
    for name, mod in model.named_modules():
        # 需要将之前在cpu上可能产生的buffer同步转移到module所在的设备上
        if not hasattr(mod, '_buffers'):
            continue
        if name in device_map:
            device = device_map[name]
        elif hasattr(mod, 'device'):
            device = mod.device
        else:
            continue
        for buffer_name, buffer in mod._buffers.items():
            if buffer is not None:
                mod.register_buffer(buffer_name, buffer.to(device))


def deepcopymodel(model: nn.Module):
    device_org = next(model.parameters()).device
    model_to_cpu(model)

    new_model = copy.deepcopy(model)

    remove_hook_from_module(new_model, True)

    model_to_org_device_with_buffer(model, device_org)

    return new_model


def model_to_org_device(model, device_org='cpu'):
    # 将原模型的权重恢复到NPU、CUDA（或meta）上
    if judge_model_with_accelerate(model):
        for mod in model.modules():
            if hasattr(mod, "_hf_hook"):
                mod._hf_hook.init_hook(mod)
    else:
        model.to(device_org)


def deepcopy_model(model,
                   logger,
                   device_org=None,
                   model_with_accelerate=True):
    # 原模型转移到CPU上
    for mod in model.modules():
        try:
            # npu, cuda -> cpu
            mod.cpu()
        except Exception as e:
            # meta -> cpu
            logger.info("Transfering meta model to cpu device...", e)
            if hasattr(mod, "_hf_hook"):
                mod._hf_hook.detach_hook(mod)

    # 深拷贝model
    new_model = copy.deepcopy(model)

    # 删除accelerate封装的forward函数，将备份的forward函数恢复
    new_model = remove_hook_from_module(new_model, True)

    model_to_org_device(model, device_org)

    return new_model


def replace_rms_norm(model: nn.Module, norm_class_name: str):
    if judge_model_with_accelerate(model):
        model_to_cpu(model)

    for name, module in model.named_modules():
        if module.__class__.__name__.lower() == 'layernorm':
            pass
        elif norm_class_name != 'layernorm' and module.__class__.__name__.lower() == norm_class_name.lower():
            new_module = NormBias(module)

            if hasattr(module, '_hf_hook'):
                module._hf_hook.weights_map = None
                new_module.old_hook = module._hf_hook
            else:
                new_module.to(module.weight.data.device)

            GraphOpt.set_module(model, name, new_module)

    # 将原模型的权重恢复到GPU（或meta）上
    if judge_model_with_accelerate(model):
        model_to_org_device(model)
        for _, module in model.named_modules():
            if isinstance(module, NormBias) and hasattr(module, 'old_hook'):
                add_hook_to_module(module, module.old_hook)
                del module.old_hook


class AntiOutlier(object):
    """Anti-outlier for LLM activation quantization."""

    def __init__(
            self,
            model: nn.Module,
            calib_data=None,
            cfg: AntiOutlierConfig = None,
            norm_class_name=None,
    ):
        self.logger = msmodelslim_logger

        if norm_class_name is not None and not isinstance(norm_class_name, str):
            raise TypeError("norm_class_name must be str, please check it.")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module, please check it.")
        if not isinstance(calib_data, list):
            raise TypeError("calib_data must be list, please check it.")
        check_type(cfg, AntiOutlierConfig, param_name="config")
        self.with_accelerate = judge_model_with_accelerate(model)

        self.cfg = cfg
        self.device = self.cfg.device

        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device

        if not same_device:
            self.logger.warning("Model is not on the deivce indicated in `AntiOutlierConfig`, "
                                "Model is on the device `{}` while `AntiOutlierConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transfering model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `AntiOutlierConfig` indicates.")

        self.norm_class_name = norm_class_name
        self.org_model = model

        # 保存anti_outlier处理前的原始权重，为避免显存的额外占用，原始权重放在内存上
        states_dic = {}
        for key, value in self.org_model.state_dict().items():
            states_dic[key] = copy.deepcopy(value).to('cpu')

        try:
            self.model_with_accelerate = judge_model_with_accelerate(model)
        except Exception as e:
            raise Exception("Please check the model and configuration.", e) from e
        if not self.model_with_accelerate:
            self.device_org = next(model.parameters()).device
        else:
            self.device_org = None

        self.model = deepcopy_model(model,
                                    self.logger,
                                    device_org=self.device_org,
                                    model_with_accelerate=self.model_with_accelerate).float()
        self.norm_linear_subgraph = None

        if calib_data is None:
            calib_data = []
        else:
            self.calib_data = calib_data

        try:
            self.init_dag()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

        # 保存anti_outlier处理前的原始权重，作为属性存入model中
        setattr(self.model, 'ori_state_dict', states_dic)

    def init_dag(self):
        dummy_input = input_to_cpu(self.calib_data[0][0])
        dummy_input = dummy_input[:1]

        if self.norm_class_name is not None:
            norm_class = list(OrderedDict.fromkeys([m.__class__ for m in self.model.modules() if
                                                    self.norm_class_name.lower() == m.__class__.__name__.lower()]))
        else:
            norm_class = list(
                OrderedDict.fromkeys(
                    [m.__class__ for m in self.model.modules() if "norm" in m.__class__.__name__.lower()]))
            norm_class = [norm_class[0]]
            self.norm_class_name = norm_class[0].__name__.lower()

        self.dag = extract_dag(self.model, dummy_input,
                               hook_nodes=norm_class, anti_method=self.cfg.anti_method)

        self.norm_linear_subgraph = self.dag.get_norm_linear_subgraph()
        if self.cfg.anti_method == 'm4':
            self.linear_linear_subgraph = self.dag.get_linear_linear_subgraph()
            self.norm_linear_subgraph.update(self.linear_linear_subgraph)

        del self.model
        self.model = self.org_model
        replace_rms_norm(self.model, self.norm_class_name)
        gc.collect()
        return

    def trans_to_dict(self, data):
        data_dict = {}
        data_dict['input_ids'] = data[0]
        if len(data) == 2:
            data_dict['attention_mask'] = data[1]
        return data_dict

    def stat_tensor(self, act_stats, name, tensor):

        # buffer the latest tensor
        if name not in act_stats:
            act_stats[name] = {}
            act_stats[name]['tensor'] = tensor

        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()  # [N,C]
        coming_max = torch.max(tensor, dim=0)[0]  # [C]
        coming_min = torch.min(tensor, dim=0)[0]  # [C]

        stat_dict = act_stats[name]

        # collect the min-max value
        if STAT_KEY_MAX in stat_dict:
            stat_dict[STAT_KEY_MAX] = torch.max(stat_dict[STAT_KEY_MAX], coming_max)  # [C]
        else:
            stat_dict[STAT_KEY_MAX] = coming_max

        if STAT_KEY_MIN in stat_dict:
            stat_dict[STAT_KEY_MIN] = torch.min(stat_dict[STAT_KEY_MIN], coming_min)  # [C]
        else:
            stat_dict[STAT_KEY_MIN] = coming_min

        # channel shifting
        if STAT_KEY_SHIFT in stat_dict:
            if self.cfg.ch_align:
                stat_dict[STAT_KEY_SHIFT] = (stat_dict[STAT_KEY_MAX] + stat_dict[STAT_KEY_MIN]) / 2  # [C]
            else:
                stat_dict[STAT_KEY_SHIFT] = torch.zeros(coming_max.size(0)).to(coming_max.device)
        else:
            if self.cfg.ch_align:
                stat_dict[STAT_KEY_SHIFT] = (coming_max + coming_min) / 2
            else:
                stat_dict[STAT_KEY_SHIFT] = torch.zeros(coming_max.size(0)).to(coming_max.device)

        # the tensor-wise max threshold
        tensor_max = torch.max(tensor - stat_dict[STAT_KEY_SHIFT])  # [N, C]
        if STAT_KEY_THRESHOLD_TENSOR in stat_dict:
            stat_dict[STAT_KEY_THRESHOLD_TENSOR] = torch.max(stat_dict[STAT_KEY_THRESHOLD_TENSOR],
                                                             tensor_max)  # [N, C]
        else:
            stat_dict[STAT_KEY_THRESHOLD_TENSOR] = tensor_max

        # the channel-wise max threshold
        channel_max = torch.max(tensor - stat_dict[STAT_KEY_SHIFT], dim=0)[0]  # [C]
        if STAT_KEY_THRESHOLD_CHANNEL in stat_dict:
            stat_dict[STAT_KEY_THRESHOLD_CHANNEL] = torch.max(stat_dict[STAT_KEY_THRESHOLD_CHANNEL],
                                                              channel_max)  # [C]
        else:
            stat_dict[STAT_KEY_THRESHOLD_CHANNEL] = channel_max

        # the tensor-wise std
        tensor_std = torch.std(tensor - stat_dict[STAT_KEY_SHIFT])  # [N, C]
        if STAT_KEY_VARIANCE in stat_dict:
            stat_dict[STAT_KEY_VARIANCE] = torch.max(stat_dict[STAT_KEY_VARIANCE], tensor_std)
        else:
            stat_dict[STAT_KEY_VARIANCE] = tensor_std

        tensor_std_unshifted = torch.std(tensor)
        tensor_mean_unshifted = torch.mean(tensor)
        scale_mask = (((tensor - tensor_mean_unshifted).abs() > 7 * tensor_std_unshifted).sum(dim=0).bool())

        if STAT_KEY_SMOOTH_SCALE_MASK in stat_dict:
            stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] = stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] | scale_mask
        else:
            stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] = scale_mask

        # if anti_method is m4, set tensor_shift to False
        tensor_shift = self.cfg.ch_align and not self.cfg.anti_method == 'm4'
        if tensor_shift:
            channel_max = torch.max((tensor - stat_dict[STAT_KEY_SHIFT]).abs().detach(), dim=0)[0]
        else:
            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

        if STAT_KEY_SMOOTH_SCALE in stat_dict:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = torch.max(stat_dict[STAT_KEY_SMOOTH_SCALE], channel_max)
        else:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = channel_max

    def os_stats(self):
        """Collect the activations into Dict for outlier suppression."""
        self.model.eval()
        act_stats = {}

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            self.stat_tensor(act_stats, name, x)

        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name))
                )

        for i in tqdm(range(len(self.calib_data))):
            inputs = self.calib_data[i]
            input_dict = self.trans_to_dict(inputs)
            self.model(**input_dict)

        for name in act_stats:
            stat_dict = act_stats[name]
            for feature in stat_dict:
                stat_dict[feature] = stat_dict[feature].to("cpu")

        for h in hooks:
            h.remove()

        return act_stats

    @torch.no_grad()
    def process(self):
        """The processing of anti-outlier."""
        try:
            self._process()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

    def save_model(self, output_path):

        output_path = get_valid_write_path(output_path)

        write_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

        if os.path.islink(output_path):
            os.unlink(output_path)

        if os.path.exists(output_path):
            os.remove(output_path)

        with os.fdopen(os.open(output_path, write_flags, write_mode), "w") as f:
            torch.save(self.model, f)

    def get_num_attention_heads(self):
        check_type(self.model.config, PretrainedConfig, param_name="model.config")

        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise ValueError(
                f"the config of model must have num_attention_heads or n_head, \
                                please check or moddify the config file"
            )
        return num_attention_heads

    def _process(self):
        act_stats = self.os_stats()
        if self.cfg.anti_method == 'm4':
            num_attention_heads = self.get_num_attention_heads()

        for norm_name in tqdm(self.norm_linear_subgraph.keys()):
            norm_module = PatternProcess.get_module_by_name(self.model, norm_name)

            linear_names = self.norm_linear_subgraph[norm_name]
            linear_modules = []
            linear_name = linear_names[0]

            if linear_name not in act_stats.keys():
                raise RuntimeError(f"key {linear_name} not in act_stats")

            stats = act_stats[linear_name]

            is_expert = any("expert" in name.lower() for name in linear_names)
            if (is_expert):
                continue

            for name in linear_names:
                mod = PatternProcess.get_module_by_name(self.model, name)
                linear_modules.append(mod)

            if self.cfg.anti_method == 'm1' or self.cfg.anti_method == 'm5':
                smooth_ln_fcs(self.cfg, norm_module, linear_modules, stats, alpha=self.cfg.alpha)
            elif self.cfg.anti_method == 'm2':
                os_ln_fcs(self.cfg, norm_module, linear_modules, stats, os_k=self.cfg.os_k)
            elif self.cfg.anti_method == 'm3':
                weight_aware(self.cfg, norm_module, linear_modules, stats)
            elif self.cfg.anti_method == 'm4':
                iter_smooth(self.cfg, norm_module, linear_modules, stats, num_attention_heads)
