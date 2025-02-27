# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
from unittest.mock import patch

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from msmodelslim.tools.preload_model import tensor_memory_page_preload
from msmodelslim.tools.convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.tools.logger import set_logger_level


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),
    parser.add_argument('--save_path', type=str, help="The path to save quant model"),
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument('--anti_dataset', type=str, default="./anti_prompt.json",
                        help="The calib data for anti outlier")
    parser.add_argument('--calib_dataset', type=str, default="./calib_prompt.json",
                        help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for anti and calibration")
    parser.add_argument('--fp8', action="store_true", help="Enable fp8 convert")
    parser.add_argument('--bf16', action="store_true", help="Enable bf16 convert")
    parser.add_argument('--as_fp16', action="store_true", help="Load model as fp16")
    parser.add_argument('--as_bf16', action="store_true", help="Load model as bf16")
    return parser.parse_args()


def custom_hook(model_config, torch_dtype, layer_count):
    model_config["mla_quantize"] = "w8a8"
    model_config["quantize"] = "w8a8_dynamic"
    model_config["model_type"] = "deepseekv2"
    model_config["torch_dtype"] = "float16" if torch_dtype == torch.float16 else "bfloat16"
    model_config["num_hidden_layers"] = layer_count


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
    return calib_dataset


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    state_dict_keys = set(state_dict_keys)
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        if module_name == "":
            # When checking if the root module is loaded there's no need to prepend module_name.
            module_keys = set(module.state_dict())
        else:
            module_keys = {f"{module_name}.{k}" for k in module.state_dict()}
        if module_keys.issubset(state_dict_keys):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules


args = parse_args()
set_logger_level("info")
pbar = tqdm(total=4, position=0, desc="Total Process")
model_path = args.model_path
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
config.model_type = "deepseekv2"
config.use_cache = False
config.torch_dtype = torch.float16 if args.as_fp16 else config.torch_dtype
config.torch_dtype = torch.bfloat16 if args.as_bf16 else config.torch_dtype

# This patch is for speeding up transformer loading big model
patch("transformers.modeling_utils.set_initialized_submodules", new=set_initialized_submodules).start()
# This patch is for speeding up accelerator inference big model
patch("accelerate.utils.modeling.clear_device_cache", new=lambda garbage_collection=False: None).start()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                          config=config,
                                          trust_remote_code=True,
                                          use_fast=True,
                                          add_eos_token=True)

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                             config=config,
                                             trust_remote_code=True,
                                             device_map={
                                                 "model.embed_tokens": 0,
                                                 "model.layers": "cpu",
                                                 "model.norm": "cpu",
                                                 "lm_head": 0,
                                             },
                                             torch_dtype="auto",
                                             attn_implementation='eager')

auto_convert_model_fp8_to_bf16(model, model_path, OpsType.get_ops_type(args.bf16, args.fp8))
tensor_memory_page_preload(model)

pbar.update(1)

with open(args.anti_dataset, "r") as file:
    anti_prompt = json.load(file)
with open(args.calib_dataset, "r") as file:
    calib_prompt = json.load(file)

anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, args.batch_size, model.device)
dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, args.batch_size, model.device)

with torch.no_grad():
    anti_config = AntiOutlierConfig(w_bit=8,
                                    a_bit=8,
                                    anti_method='m4',
                                    dev_type='npu',
                                    dev_id=model.device.index)
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()

pbar.update(1)

disable_names = []
for ids in range(config.num_hidden_layers):
    disable_names.append("model.layers." + str(ids) + ".self_attn.kv_b_proj")

quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=True
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level="L0")
calibrator.run()
pbar.update(1)

calibrator.save(args.save_path,
                json_name="quant_model_description_w8a8_dynamic.json",
                safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
                save_type=["safe_tensor"],
                part_file_size=4)

custom_hooks = {
    'config.json': functools.partial(modify_config_json,
                                     custom_hook=functools.partial(custom_hook,
                                                                   torch_dtype=config.torch_dtype,
                                                                   layer_count=config.num_hidden_layers))
}
copy_config_files(input_path=args.model_path, output_path=args.save_path, quant_config=quant_config,
                  custom_hooks=custom_hooks)
pbar.update(1)
