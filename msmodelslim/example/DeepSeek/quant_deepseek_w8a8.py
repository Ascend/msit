# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import sys
import argparse
import functools
import json
from unittest.mock import patch

import torch
import torch_npu

from tqdm import tqdm

from convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType
from add_safetensors import add_safetensors

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from ascend_utils.common.security import get_valid_read_path, get_write_directory, check_number
from example.common.utils import SafeGenerator
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
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for anti and calibration")
    parser.add_argument('--from_fp8', action='store_true', help="Origin model is of fp8")
    parser.add_argument('--from_bf16', action='store_true', help="Origin model is of bf16")
    return parser.parse_args()


def custom_hook(model_config):
    model_config["mla_quantize"] = "w8a8"
    model_config["quantize"] = "w8a8_dynamic"
    model_config["model_type"] = "deepseekv2"


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


def main():
    args = parse_args()
    set_logger_level("info")
    # 显示整个量化过程各个步骤的进度条
    pbar = tqdm(total=5, position=0, desc="Total Process")

    model_path = args.model_path
    batch_size = args.batch_size

    save_path = get_write_directory(args.save_path, write_mode=0o750)
    check_number(batch_size, int, 1, 16, "batch_size")

    safe_generator = SafeGenerator()

    config = safe_generator.get_config_from_pretrained(model_path=model_path, 
                                                       trust_remote_code=True)
    # Set layer count to 0 means use all layers, otherwise it will only use the first layer_count layers
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
    # Set model type to deepseekv2 because we only support deepseekv2 now,
    # but v3's architecture is same as v2 without mtp layers so that we can reuse
    config.model_type = "deepseekv2"
    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

    tokenizer = safe_generator.get_tokenizer_from_pretrained(model_path=model_path,
                                                             config=config,
                                                             trust_remote_code=True,
                                                             use_fast=True,
                                                             add_eos_token=True)

    model = safe_generator.get_model_from_pretrained(model_path=model_path,
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

    auto_convert_model_fp8_to_bf16(model, model_path, OpsType.get_ops_type(args.from_bf16, args.from_fp8))

    pbar.update(1)

    anti_dataset_path = get_valid_read_path(args.anti_dataset, "json", is_dir=False)
    calib_dataset_path = get_valid_read_path(args.calib_dataset, "json", is_dir=False)

    with open(anti_dataset_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_dataset_path, "r") as file:
        calib_prompt = json.load(file)

    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, model.device)

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
    )

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level="L0")
    calibrator.run()
    pbar.update(1)

    calibrator.save(save_path,
                    json_name="quant_model_description_w8a8_dynamic.json",
                    safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
                    save_type=["safe_tensor"],
                    part_file_size=4)

    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
    }
    copy_config_files(input_path=model_path, output_path=save_path, quant_config=quant_config,
                      custom_hooks=custom_hooks)
    pbar.update(1)
    add_safetensors(org_paths=model_path, target_dir=save_path, safetensors_prefix="mtp_float",
                    max_file_size_gb=5, prefix="model.layers.61.")
    pbar.update(1)


if __name__ == "__main__":
    # torch_npu will fork a new process to init,
    # it's lazy_init will fail after we load a big model,so we need to init it here
    torch_npu.npu.init()
    # This patch is for speeding up transformer loading big model
    patch("transformers.modeling_utils.set_initialized_submodules", new=set_initialized_submodules).start()
    # Invoke main process
    main()
