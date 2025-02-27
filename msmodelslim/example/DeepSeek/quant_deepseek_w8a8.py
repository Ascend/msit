#Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.tools.convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.tools.logger import set_logger_level

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path"),
    parser.add_argument('--save_path', type=str, help="save path"),
    parser.add_argument('--layer_count', type=int, default=0)
    parser.add_argument('--anti_dataset', type=str, default="./anti_prompt.json")
    parser.add_argument('--calib_dataset', type=str, default="./calib_prompt.json")
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    return parser.parse_args()

def custom_hook(model_config):
    model_config["mla_quantize"] = "w8a8"

args = parse_args()
set_logger_level("warning")
pbar = tqdm(total=4, position=0, desc="Total Process")
model_path = args.model_path
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                          config=config,
                                          trust_remote_code=True,
                                          use_fast=True,
                                          add_eos_token=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                             config=config,
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype="auto",
                                             max_memory={
                                                 0: "50GiB",
                                                 "cpu": "1500GiB"
                                             },
                                             attn_implementation='eager')

auto_convert_model_fp8_to_bf16(model, model_path, OpsType.get_ops_type(args.bf16, args.fp8))

pbar.update(1)


def get_anti_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append(inputs.data['input_ids'].to(device))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    for i in range(len(calib_dataset)):
        calib_dataset[i] = F.pad(calib_dataset[i], (0, max_len - calib_dataset[i].size(1)), value=0)
    return torch.cat(calib_dataset)

def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to(device)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


with open(args.anti_dataset, "r") as file:
    anti_prompt = json.load(file)
with open(args.calib_dataset, "r") as file:
    calib_prompt = json.load(file)

anti_data = []
for i in range(len(anti_prompt)):
    tmp = get_anti_dataset(tokenizer, anti_prompt[i])
    anti_data.append(tmp)

anti_dataset = []
for data in anti_data:
    anti_dataset.append([data])

dataset_calib = []
for i in range(len(calib_prompt)):
    tmp = get_calib_dataset(tokenizer,calib_prompt[i])
    dataset_calib += (tmp)

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
calibrator.save(args.save_path, save_type=["safe_tensor"], part_file_size=4)

custom_hooks = {
    'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
}
copy_config_files(input_path=args.model_path, output_path=args.save_path, quant_config=quant_config, custom_hooks=custom_hooks)
pbar.update(1)