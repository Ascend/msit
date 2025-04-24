# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
import copy
from unittest.mock import patch

import torch
import torch_npu

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType
from add_safetensors import add_safetensors

from ascend_utils.common.security import get_valid_read_path, get_write_directory, check_number
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
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for anti and calibration")
    parser.add_argument('--from_fp8', action='store_true', help="Origin model is of fp8")
    parser.add_argument('--from_bf16', action='store_true', help="Origin model is of bf16")
    return parser.parse_args()


def custom_hook(model_config):
    model_config["mla_quantize"] = "w8a8"
    model_config["quantize"] = "w8a8_dynamic"
    model_config["moe_quantize"] = "w4a8_dynamic"
    model_config["model_type"] = "deepseekv2"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, max_len=512, device="npu"):
    calib_dataset = []

    def truncate_strings(strings: list[str], max_len=max_len) -> list[str]:
        result = []
        for s in strings:
            current = s
            while True:
                # 截断前 512 字符作为一个块
                chunk = current[:max_len]
                result.append(chunk)
                # 剩余部分继续处理
                current = current[max_len:]
                # 没有剩余内容时结束循环
                if not current:
                    break
        return result
    

    calib_list = truncate_strings(calib_list)
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
    return calib_dataset


def main():
    args = parse_args()
    set_logger_level("info")
    # 显示整个量化过程各个步骤的进度条
    pbar = tqdm(total=5, position=0, desc="Total Process")
    model_path = args.model_path
    save_path = args.save_path
    anti_path = args.anti_dataset
    calib_path = args.calib_dataset
    batch_size = args.batch_size

    model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=False)
    save_path = get_write_directory(save_path, write_mode=0o750)
    anti_path = get_valid_read_path(anti_path, is_dir=False, check_user_stat=False)
    calib_path = get_valid_read_path(calib_path, is_dir=False, check_user_stat=False)
    check_number(batch_size, int, 1, 16, "batch_size")
    
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
    num_layer = config.num_hidden_layers
    if args.layer_count < 0 or args.layer_count > num_layer:
        raise ValueError(
            f"Invalid value for parameter layer_count: {args.layer_count}."
            f"Must be between 0 and {num_layer}."
        )
    # Set layer count to 0 means use all layers, otherwise it will only use the first layer_count layers
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
    if config.num_hidden_layers < 0:
        raise ValueError("model num_hidden_layers is invalid, please check it.")

    # Set model type to deepseekv2 because we only support deepseekv2 now,
    # but v3's architecture is same as v2 without mtp layers so that we can reuse
    config.model_type = "deepseekv2"
    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

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

    auto_convert_model_fp8_to_bf16(model, model_path, OpsType.get_ops_type(args.from_bf16, args.from_fp8))

    pbar.update(1)

    with open(anti_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_path, "r") as file:
        calib_prompt = json.load(file)

    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, device=model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, device=model.device)

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

    w4a8_pertoken_config = QuantConfig(
        a_bit=8,
        w_bit=4,
        w_sym=True,
        dev_id=model.device.index,
        dev_type='npu',
        is_lowbit=True,
        mm_tensor=False,
        group_size=256,
        is_dynamic=True,
        do_smooth=False,
        use_sigma=True,
        open_outlier=False,
        disable_names=disable_names,
    )

    # w4a8_dynamic 和 w8a8 & w8a8_dynamic 混合量化配置
    mix_cfg = {
        "model.layers.[012].mlp.gate_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.down_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.up_proj": "w8a8_dynamic",
        "model.layers.[0-9]*.self_attn.*": "w8a8",
        "model.layers.[0-9]*.mlp.shared_experts.*": "w8a8_dynamic",
        "model.layers.[0-9]*.mlp.experts.*": "w4a8_dynamic", 
    }

    calibrator = Calibrator(model, w4a8_pertoken_config, calib_data=dataset_calib, disable_level="L0", mix_cfg=mix_cfg)
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
    copy_config_files(input_path=model_path, output_path=save_path, quant_config=w4a8_pertoken_config,
                      custom_hooks=custom_hooks)
    pbar.update(1)
    add_safetensors(org_paths=model_path, target_dir=save_path, safetensors_prefix="mtp_float",
                    max_file_size_gb=5, prefix="model.layers.61.")
    pbar.update(1)


if __name__ == "__main__":
    # torch_npu will fork a new process to init,
    # it's lazy_init will fail after we load a big model,so we need to init it here
    torch_npu.npu.init()
    main()
