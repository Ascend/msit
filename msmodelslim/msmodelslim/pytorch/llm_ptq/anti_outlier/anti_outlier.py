#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import absolute_import, division, print_function

import os
import gc
import stat
import copy
import functools
from typing import OrderedDict, Mapping, Optional, Tuple, List
import inspect
from collections import OrderedDict as OrderedDict_CHECK
from easydict import EasyDict

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ascend_utils import ResListToRelease
from ascend_utils.common.security import get_valid_write_path, check_type
from msmodelslim.pytorch.llm_ptq.hooks.hook_def import ProcessHook
from msmodelslim.pytorch.llm_ptq.hooks.factory import get_process_hooks
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import enabled_adapter
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

try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import migration, migration_vit
except ImportError:
    migration, migration_vit = None, None
    msmodelslim_logger.warning(
        "The current CANN version does not support importing the migration and migration_vit packages."
    )
 
try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import attach_op, Multiplier
except ImportError:
    attach_op, Multiplier = None, None
    msmodelslim_logger.warning(
        "The current CANN version does not support importing the attach_op and Multiplier packages."
    )

from msmodelslim.pytorch.llm_ptq.accelerate_adapter import (PrepareWeight,
                                                            move_update_weight_hook_if_need,
                                                            check_model_compatible,
                                                            get_offloaded_dataset,
                                                            MemoryStateDictConfig,
                                                            DiskStateDictConfig,
                                                            copy_offloaded_state_dict,
                                                            enable_adapter,
                                                            replace_device_align_hook_if_needed)

STAT_KEY_MAX = "max"
STAT_KEY_MIN = "min"
STAT_KEY_SHIFT = "shift"
STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
STAT_KEY_THRESHOLD_TENSOR = "thres_t"
STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
STAT_KEY_SMOOTH_SCALE = "smooth_scale"
STAT_KEY_VARIANCE = "std"
TENSOR = 'tensor'
SCALE_MIN_LLM = 1e-5
SCALE_MIN_SD3 = 1e-3

SD3_CONTEXT_EMBEDDER = "context_embedder"

_PREDEFINED_FUSIONS = {
    tuple(): ("context_embedder",)
}

_PREDEFINED_FUSION_KWARGS = {
    "check_group_fusions": False
}

STATE_DICT_COPY_DIR = "msmodelslim_copy"


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
            msmodelslim_logger.info("Transfering meta model to cpu device...", e)
            if hasattr(mod, "_hf_hook"):
                mod._hf_hook.detach_hook(mod)


def model_to_org_device_with_buffer(model, device_org='cpu'):
    if not judge_model_with_accelerate(model):
        model.to(device_org)

    # 将原模型的权重恢复到GPU（或meta）上
    for _, mod in model.named_modules():
        if hasattr(mod, '_hf_hook'):
            mod._hf_hook.init_hook(mod)
    for name, mod in model.named_modules():
        # 需要将之前在cpu上可能产生的buffer同步转移到module所在的设备上
        if not hasattr(mod, '_buffers'):
            continue
        if not judge_model_with_accelerate(model):
            device = model.device
        elif hasattr(model, 'hf_device_map') and name in model.hf_device_map:
            device = f"npu:{model.hf_device_map[name]}" if "npu" in model.device.type else model.hf_device_map[name]
        elif hasattr(mod, 'device'):
            device = mod.device
        else:
            continue
        for buffer_name, buffer in mod._buffers.items():
            if buffer is not None:
                mod.register_buffer(buffer_name, buffer.to(device))


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

    model_to_org_device_with_buffer(model, device_org)

    return new_model


def replace_rms_norm(model: nn.Module, norm_class_name: str):
    for name, module in model.named_modules():
        if module.__class__.__name__.lower() == 'layernorm':
            pass
        elif norm_class_name != 'layernorm' and module.__class__.__name__.lower() == norm_class_name.lower():
            if enabled_adapter():
                with PrepareWeight(module):
                    new_module = NormBias(module)
                    new_module.to(module.weight.data.device)
                    move_update_weight_hook_if_need(module, new_module, as_submodule=True)
                    GraphOpt.set_module(model, name, new_module)
            else:
                new_module = NormBias(module)

                if hasattr(module, '_hf_hook'):
                    module._hf_hook.weights_map = None
                    new_module.old_hook = module._hf_hook
                else:
                    new_module.to(module.weight.data.device)

                GraphOpt.set_module(model, name, new_module)


def copy_state_dict(model: torch.nn.Module, typ: str = 'disk') -> Mapping:
    if check_model_compatible(model):
        if typ == 'disk':
            dataset = get_offloaded_dataset(model)
            if dataset is None:
                # 如果没有加载到 disk，则保持一致，保存 state_dict 到 cpu
                config = MemoryStateDictConfig()
            else:
                # 否则在现有的 offload 路径下新建 copy 文件夹，然后保存到里面
                save_folder = os.path.join(dataset.save_folder, STATE_DICT_COPY_DIR)
                config = DiskStateDictConfig().save_folder(save_folder)
        elif typ == 'memory':
            config = MemoryStateDictConfig()
        else:
            raise ValueError("state dict type must be disk or memory")
        return copy_offloaded_state_dict(model, config)

    states_dic = {}
    for key, value in model.state_dict().items():
        states_dic[key] = copy.deepcopy(value).to('cpu')
    return states_dic


def is_model_multimodal(model):
    if not hasattr(model.config, 'architectures'):
        return False
    if (model.config.architectures[0] == 'LlavaForConditionalGeneration' or 
        (model.config.architectures[0] == 'QWenLMHeadModel' and 
        hasattr(model.config, 'visual'))):
        return True
    return False


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
        self.calib_data = [] if calib_data is None else self.check_calib_data(calib_data)

        if is_model_multimodal(model):
            if cfg.anti_method != 'm2':
                raise ValueError("anti_method must be m2 while using multimodal qwen-vl or llava-v1.5")
            else:
                self.cfg = cfg
                self.model = model
                self.calib_data = calib_data
                return

        check_type(cfg, AntiOutlierConfig, param_name="config")
        self.with_accelerate = judge_model_with_accelerate(model)

        self.cfg = cfg
        self.device = self.cfg.device

        self.is_context_embedder_model = False
        if SD3_CONTEXT_EMBEDDER + ".weight" in model.state_dict().keys():
            self.is_context_embedder_model = True

        # 开启低显存低内存模式
        if self.cfg.is_adapter_enabled:
            enable_adapter()

        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device

        if not check_model_compatible(model) and not same_device:
            self.logger.warning("Model is not on the deivce indicated in `AntiOutlierConfig`, "
                                "Model is on the device `{}` while `AntiOutlierConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transfering model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `AntiOutlierConfig` indicates.")

        replace_device_align_hook_if_needed(model)

        self.norm_class_name = norm_class_name
        if not enabled_adapter():
            self.org_model = model
        
        self.hooks = get_process_hooks(model)

        # 非m4或m5场景下，保存anti_outlier处理前的原始权重，为避免显存的额外占用，原始权重放在内存上
        if self.cfg.anti_method not in ['m4', 'm5']:
            states_dic = copy_state_dict(model, self.cfg.offload_type)

        try:
            self.model_with_accelerate = judge_model_with_accelerate(model)
        except Exception as e:
            raise Exception("Please check the model and configuration.", e) from e
        if not self.model_with_accelerate:
            self.device_org = next(model.parameters()).device
        else:
            self.device_org = None
        
         # 如果手动指定NORM_LINEAR结构，就无需拷贝模型了
        if ProcessHook.GET_NORM_LINEAR_SUBGRAPH in self.hooks and self.hooks[
            ProcessHook.GET_NORM_LINEAR_SUBGRAPH] is not None:
            self.norm_linear_subgraph = self.hooks[ProcessHook.GET_NORM_LINEAR_SUBGRAPH](model)
            self.model = model
        else:
            self.model = deepcopy_model(model,
                                        self.logger,
                                        device_org=self.device_org,
                                        model_with_accelerate=self.model_with_accelerate).float()
            self.norm_linear_subgraph = None

        if not enabled_adapter() and model.device.type != 'cpu':
            self.model = deepcopy_model(model,
                                        self.logger,
                                        device_org=self.device_org,
                                        model_with_accelerate=self.model_with_accelerate).float()
        else:
            self.model = model

        if calib_data is None:
            calib_data = []
        else:
            self.calib_data = calib_data

        try:
            self.init_dag()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

        # 非m4或m5场景下，保存anti_outlier处理前的原始权重，作为属性存入model中
        if self.cfg.anti_method not in ['m4', 'm5']:
            setattr(self.model, 'ori_state_dict', states_dic)

    def init_dag(self):
        if self.is_context_embedder_model:
            self.norm_linear_subgraph = _PREDEFINED_FUSIONS
        else:
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
            if ProcessHook.GET_NORM_LINEAR_SUBGRAPH not in self.hooks or self.hooks[
                ProcessHook.GET_NORM_LINEAR_SUBGRAPH] is None:
                # 不要保存为成员变量，内部会引用模型以及模型的子模块，导致这些模块的参数正常无法释放
                dag = extract_dag(self.model, dummy_input,
                                hook_nodes=norm_class, anti_method=self.cfg.anti_method)
                self.norm_linear_subgraph = dag.get_norm_linear_subgraph()
                if self.cfg.anti_method == 'm4':
                    self.linear_linear_subgraph = dag.get_linear_linear_subgraph()
                    self.norm_linear_subgraph.update(self.linear_linear_subgraph)
                del dag

        if not enabled_adapter():
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
            if not self.is_context_embedder_model:
                act_stats[name][TENSOR] = tensor

        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]
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
            if not self.is_context_embedder_model:
                input_dict = self.trans_to_dict(inputs)
                self.model(**input_dict)
            else:
                self.model(**inputs)

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
        check_type(self.model.config, (PretrainedConfig, OrderedDict_CHECK), param_name="model.config")

        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise ValueError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or moddify the config file"
            )
        return num_attention_heads

    def check_calib_data(self, calib_data):
        check_type(calib_data, list, param_name='calib_data')
        for i, calib_data_item in enumerate(calib_data):
            element_not_tensor = False
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            if isinstance(calib_data_item, list):
                for item in calib_data_item:
                    if not isinstance(item, torch.Tensor):
                        element_not_tensor = True
                        break
            elif isinstance(calib_data_item, dict):
                for _, value in calib_data_item.items():
                    if not isinstance(value, torch.Tensor):
                        element_not_tensor = True
                        break
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")

        return calib_data

    def anti_for_multimodal(self, cfg, model):
        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)
            
        block_dict = {
            "QWenBlock" : QuantV2QwenBlock,
            "VisualAttentionBlock" : QuantVisualAttentionBlock,
            "LlamaDecoderLayer" : LlavaQuantDecoder,
            "CLIPEncoderLayer" : LlavaClipVision,
            }

        for name, mod in model.named_modules():
            mod_name = mod.__class__.__name__
            if (mod_name in block_dict):
                quant_mod = block_dict[mod_name](mod, cfg, name)
                self.logger.info(
                    "multimodal model replace block `{}` to `{}`".format(mod_name, block_dict[mod_name].__name__)
                )
                if hasattr(mod, '_hf_hook'):
                    add_hook_to_module(quant_mod, mod._hf_hook)
                    remove_hook_from_module(mod)
                _set_module(model, name, quant_mod)
                del mod

    def _process(self):
        if is_model_multimodal(self.model):
            if self.cfg.anti_method != 'm2':
                raise ValueError("anti_method must be m2 while using multimodal qwen-vl or llava-v1.5")
            else:
                self.anti_for_multimodal(self.cfg, self.model)
                data = self.calib_data[0]
                if isinstance(data, tuple) or isinstance(data, list):
                    self.model(*data)
                elif isinstance(data, dict):
                    self.model(**data)
                return
        fusion_kwargs = {}
        act_stats = self.os_stats()
        if self.cfg.anti_method == 'm4':
            num_attention_heads = self.get_num_attention_heads()
        scale_min = SCALE_MIN_LLM
        
        if self.is_context_embedder_model:
            fusion_kwargs = _PREDEFINED_FUSION_KWARGS
            scale_min = SCALE_MIN_SD3
        
        for norm_name_group in tqdm(self.norm_linear_subgraph.keys()):
            linear_names = self.norm_linear_subgraph[norm_name_group]
            if isinstance(norm_name_group, str):
                norm_module = PatternProcess.get_module_by_name(self.model, norm_name_group)
            elif len(norm_name_group) > 0:
                norm_module = PatternProcess.get_module_by_name(self.model, norm_name_group[0])
            else:
                norm_module = None

            linear_modules = []
            linear_name = linear_names[0]

            if linear_name not in act_stats.keys():
                raise RuntimeError(f"key {linear_name} not in act_stats")

            stats = act_stats[linear_name]

            is_expert = any("expert" in name.lower() for name in linear_names)
            if (is_expert):
                continue

            self.logger.debug(f"smooth {norm_name_group} -> {linear_names}")

            for name in linear_names:
                mod = PatternProcess.get_module_by_name(self.model, name)
                linear_modules.append(mod)

            args = []
            if ProcessHook.MODIFY_SMOOTH_ARGS in self.hooks and self.hooks[
                ProcessHook.MODIFY_SMOOTH_ARGS] is not None:
                args, fusion_kwargs = self.hooks[ProcessHook.MODIFY_SMOOTH_ARGS](
                    self.cfg, 
                    norm_name_group, 
                    linear_names, args, 
                    fusion_kwargs
                )

            if Multiplier is not None and norm_module is None:
                norm_module = Multiplier(
                    torch.ones_like(stats[STAT_KEY_SMOOTH_SCALE]).to(linear_modules[0].weight.device)
                )

            prepare_list = [PrepareWeight(norm_module, post_force=True, post_recurse=True)]
            prepare_list += [PrepareWeight(mod, post_force=True) for mod in linear_modules]

            with ResListToRelease(*prepare_list):
                if self.cfg.anti_method == 'm1' or self.cfg.anti_method == 'm5':
                    smooth_ln_fcs(self.cfg, norm_module, linear_modules, stats, alpha=self.cfg.alpha)
                elif self.cfg.anti_method == 'm2':
                    os_ln_fcs(self.cfg, norm_module, linear_modules, stats, os_k=self.cfg.os_k)
                elif self.cfg.anti_method == 'm3':
                    weight_aware(self.cfg, norm_module, linear_modules, stats)
                elif self.cfg.anti_method == 'm4':
                    if 'scale_min' in inspect.signature(iter_smooth).parameters:
                        fusion_kwargs.update({"scale_min": scale_min})
                    if 'check_group_fusions' not in inspect.signature(iter_smooth).parameters:
                        fusion_kwargs.pop("check_group_fusions", None)
                    self.logger.debug(f"fusion_kwargs is {fusion_kwargs}")
                    iter_smooth(
                        self.cfg, norm_module, linear_modules, stats, num_attention_heads, **fusion_kwargs
                        )
                    if attach_op is not None and Multiplier is not None and isinstance(norm_module, Multiplier):
                        attach_op(self.model, norm_module, linear_modules, linear_names)


def get_config():
    a_qconfig = {
        'quantizer': 'FixedFakeQuantize',
        'bit': 8,
        'symmetric': False,
        'ch_axis': -1,

    }
    w_qconfig = {
        'quantizer': 'FixedFakeQuantize',
        'bit': 8,
        'symmetric': True,
        'ch_axis': 0,
    }
    return EasyDict(a_qconfig), EasyDict(w_qconfig)


class QuantV2QwenBlock(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.self_attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.input_layernorm = org_layer.ln_1
        self.post_attention_layernorm = org_layer.ln_2
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

        self.layername = layername

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = args[0]
        rotary_pos_emb = kwargs.pop("rotary_pos_emb")
        registered_causal_mask = kwargs.pop("registered_causal_mask")
        layer_past = kwargs.pop("layer_past")
        attention_mask = kwargs.pop("attention_mask")
        head_mask = kwargs.pop("head_mask")
        encoder_hidden_states = kwargs.pop("encoder_hidden_states")
        encoder_attention_mask = kwargs.pop("encoder_attention_mask")
        use_cache = kwargs.pop("use_cache")
        output_attentions = kwargs.pop("output_attentions")

        org_hidden_states = copy.deepcopy(hidden_states)
        layernorm_output = self.input_layernorm(org_hidden_states)

        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is QuantV2QwenBlock , layername:`{}` ".format(self.layername))
            weight_all = self.self_attn.c_attn.weight
            bias_list = None
            if self.self_attn.c_attn.bias is not None:
                bias_list = torch.cat([self.self_attn.c_attn.bias])

            extra_dict = {
                'split_size': self.self_attn.split_size,
                'num_heads': self.self_attn.num_heads,
                'head_dim': self.self_attn.head_dim,
                'scale_attn_weights': self.self_attn.scale_attn_weights,
                'head_mask': head_mask,
                'observation_mask': None,
                'attention_mask': attention_mask,
            }
            # update scale
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(layernorm_output, weight_all, a_qconfig, w_qconfig, 'qkv', extra_dict, bias=bias_list)
            layernorm_output /= best_scale
            self.self_attn.c_attn.weight.data *= best_scale

            self.input_layernorm.weight.data = self.input_layernorm.weight.data.to('npu')
            self.input_layernorm.weight.data /= best_scale
            self.cac_migrate_attn = False
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            registered_causal_mask=registered_causal_mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs_tmp = attn_outputs[1:]
        residual = hidden_states

        layernorm_input = residual + attn_output
        
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_input

        if self.cac_migrate_mlp:            
            weight_list = torch.cat([self.mlp.w2.weight,  # gate_proj
                                     self.mlp.w1.weight]) # up_proj
            extra_dict = {
                'observation_mask': None, 
            }

            a_qconfig, w_qconfig = get_config()

            best_scale = \
                migration(layernorm_output, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            layernorm_output /= best_scale
            self.mlp.w1.weight.data *= best_scale
            self.mlp.w2.weight.data *= best_scale
            self.post_attention_layernorm.weight.data /= best_scale
            self.cac_migrate_mlp = False

        mlp_output = self.mlp(layernorm_output)
        
        hidden_states = residual + mlp_output
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += outputs_tmp
        else:
            outputs += outputs_tmp[1:]
        return outputs
    

class QuantVisualAttentionBlock(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.ln_1 = org_layer.ln_1
        self.ln_2 = org_layer.ln_2
        self.attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.ln_1_kv = org_layer.ln_1_kv if hasattr(org_layer, 'ln_1_kv') else None

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True
        self.layername = layername

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        post_ln_1 = self.ln_1(q_x)
         
        if self.cac_migrate_attn:
            msmodelslim_logger.info(
                f"current block is QuantVisualAttentionBlock , "
                f"layername:`{self.layername}` "
            )
            channel_max = post_ln_1.max(0)[0].max(0)[0]
            channel_min = post_ln_1.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_1 -= shift
            if (self.attn.in_proj.bias is None):
                msmodelslim_logger.warning("attn.in_proj.bias is None")
            self.attn.in_proj.bias.data += shift @ self.attn.in_proj.weight.data.T
                
            # calculate scale
            weight_list = torch.cat([self.attn.in_proj.weight])
            extra_dict = {
                'hidden_size_per_partition': self.attn.hidden_size_per_partition,
                'norm_factor': self.attn.norm_factor,
                'hidden_size_per_attention_head': self.attn.hidden_size_per_attention_head,
                'num_attention_heads_per_partition': self.attn.num_attention_heads_per_partition,
                'attn_mask': attn_mask,
                'bias': torch.cat([self.attn.in_proj.bias]),
                'shift': shift,
            }

            a_qconfig, w_qconfig = get_config()

            # update scale
            best_scale = \
                migration_vit(post_ln_1, weight_list, a_qconfig, w_qconfig, 'vit_qkv_function', extra_dict)
            post_ln_1 /= best_scale
            ## linear and ln
            self.attn.in_proj.weight.data *= best_scale
            self.ln_1.bias.data -= shift
            self.ln_1.weight.data /= best_scale
            self.ln_1.bias.data /= best_scale
            self.cac_migrate_attn = False
            

        x = q_x + self.attention(q_x=post_ln_1, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        
        post_ln_2 = self.ln_2(x)
        
        if self.cac_migrate_mlp:
            channel_max = post_ln_2.max(0)[0].max(0)[0]
            channel_min = post_ln_2.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_2 -= shift
            if (self.mlp.c_fc.bias is None):
                msmodelslim_logger.warning("mlp.c_fc.bias is None")
            self.mlp.c_fc.bias.data += shift @ self.mlp.c_fc.weight.data.T
            # calculate scale
            weight_list = torch.cat([self.mlp.c_fc.weight])
            extra_dict = {
                'bias': torch.cat([self.mlp.c_fc.bias]),
                'shift': shift,
                'observation_mask': None  # test
                }

            a_qconfig, w_qconfig = get_config()

            # update scale
            best_scale = \
                migration_vit(post_ln_2, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)
            post_ln_2 /= best_scale
            ## linear and ln
            self.mlp.c_fc.weight.data *= best_scale
            self.ln_2.bias.data -= shift
            self.ln_2.weight.data /= best_scale
            self.ln_2.bias.data /= best_scale
            self.cac_migrate_mlp = False

        x = x + self.mlp(post_ln_2)
        return x
       

class LlavaQuantDecoder(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.self_attn = org_layer.self_attn 
        self.mlp = org_layer.mlp
        self.input_layernorm = org_layer.input_layernorm 
        self.post_attention_layernorm = org_layer.post_attention_layernorm
        self.act_fn = org_layer.mlp.act_fn

        self.layername = layername

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = args[0]
        attention_mask = kwargs.pop("attention_mask")
        position_ids = kwargs.pop("position_ids")
        past_key_value = kwargs.pop("past_key_value")
        output_attentions = kwargs.pop("output_attentions")
        use_cache = kwargs.pop("use_cache")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is LlavaQuantDecoder , layername:`{}` ".format(self.layername))
            weight_list = torch.cat([self.self_attn.q_proj.weight,
                                     self.self_attn.k_proj.weight,
                                     self.self_attn.v_proj.weight])
            bias_list = None

            extra_dict = {
                'num_heads': self.self_attn.num_heads,
                'num_key_value_heads': self.self_attn.num_key_value_heads,
                'num_key_value_groups': self.self_attn.num_key_value_groups,
                'cos_cached': self.self_attn.rotary_emb.cos_cached,
                'sin_cached': self.self_attn.rotary_emb.sin_cached,
                'head_dim': self.self_attn.head_dim,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'observation_mask': None

            }
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'llama_qkv', extra_dict, bias=bias_list)
            hidden_states /= best_scale
            self.self_attn.q_proj.weight.data *= best_scale
            self.self_attn.k_proj.weight.data *= best_scale
            self.self_attn.v_proj.weight.data *= best_scale
            self.input_layernorm.weight.data = self.input_layernorm.weight.data.to('npu')
            self.input_layernorm.weight.data /= best_scale
            self.cac_migrate_attn = False

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.cac_migrate_mlp:

            weight_list = torch.cat([self.mlp.gate_proj.weight,
                                     self.mlp.up_proj.weight])
            
            extra_dict = {
                'observation_mask': None, 
                'act_fn': self.act_fn
            }

            a_qconfig, w_qconfig = get_config()

            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            hidden_states /= best_scale
            self.mlp.gate_proj.weight.data *= best_scale
            self.mlp.up_proj.weight.data *= best_scale
            self.post_attention_layernorm.weight.data /= best_scale
            self.cac_migrate_mlp = False

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
     
        return outputs


class LlavaClipVision(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.layer_norm1 = org_layer.layer_norm1
        self.layer_norm2 = org_layer.layer_norm2
        self.self_attn = org_layer.self_attn 
        self.mlp = org_layer.mlp
        self.act_fn = org_layer.mlp.activation_fn
        self.layername = layername

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is LlavaClipVision , layername:`{}` ".format(self.layername))
            channel_max = hidden_states.max(0)[0].max(0)[0]
            channel_min = hidden_states.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            hidden_states -= shift

            if hasattr(self.self_attn.q_proj, 'bias') and self.self_attn.q_proj.bias is not None:
                self.self_attn.q_proj.bias.data += shift @ self.self_attn.q_proj.weight.data.T
            if hasattr(self.self_attn.k_proj, 'bias') and self.self_attn.k_proj.bias is not None:
                self.self_attn.k_proj.bias.data += shift @ self.self_attn.k_proj.weight.data.T
            if hasattr(self.self_attn.v_proj, 'bias') and self.self_attn.v_proj.bias is not None:
                self.self_attn.v_proj.bias.data += shift @ self.self_attn.v_proj.weight.data.T

            # calculate scale
            weight_list = torch.cat([
                self.self_attn.q_proj.weight, 
                self.self_attn.k_proj.weight, 
                self.self_attn.v_proj.weight]
            )
            bias_list = torch.cat([
                self.self_attn.q_proj.bias, 
                self.self_attn.k_proj.bias, 
                self.self_attn.v_proj.bias]
            )
            

            extra_dict = {
                'split_size': self.self_attn.embed_dim,
                'num_heads': self.self_attn.num_heads,
                'head_dim': self.self_attn.head_dim,
                'causal_attention_mask': None,
                'observation_mask': None,
                'attention_mask': attention_mask,
                'bias': bias_list,
            }

            a_qconfig, w_qconfig = get_config()
            

            # update scale
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'llava_vit_qkv', extra_dict)
            hidden_states /= best_scale
            ## linear and ln
            self.self_attn.q_proj.weight.data *= best_scale
            self.self_attn.k_proj.weight.data *= best_scale
            self.self_attn.v_proj.weight.data *= best_scale

            self.layer_norm1.bias.data -= shift
            self.layer_norm1.weight.data /= best_scale
            self.layer_norm1.bias.data /= best_scale
            self.cac_migrate_attn = False

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        if self.cac_migrate_mlp:
            channel_max = hidden_states.max(0)[0].max(0)[0]
            channel_min = hidden_states.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            hidden_states -= shift
            if (self.mlp.fc1.bias is None):
                msmodelslim_logger.warning("mlp.fc1.bias is None")
            self.mlp.fc1.bias.data += shift @ self.mlp.fc1.weight.data.T

            weight_list = torch.cat([self.mlp.fc1.weight])


            extra_dict = {
                'bias': torch.cat([self.mlp.fc1.bias]),
                'act_fn': self.act_fn,
                'observation_mask': None,
            }

            a_qconfig, w_qconfig = get_config()


            # update scale
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)

            hidden_states /= best_scale
            self.mlp.fc1.weight.data *= best_scale

            self.layer_norm2.bias.data -= shift
            self.layer_norm2.weight.data /= best_scale
            self.layer_norm2.bias.data /= best_scale
            
            self.cac_migrate_mlp = False

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs