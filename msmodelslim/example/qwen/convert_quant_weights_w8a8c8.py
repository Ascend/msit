# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from tools.copy_config_files import copy_config_files
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights")
    parser.add_argument("--model_path", type=str,
                        help="The path to model float weights")
    parser.add_argument("--save_path", type=str,
                        help="The path to save quant weights")
    return parser.parse_args()

args = parse_args()
IN_MODEL_PATH = args.model_path
OUT_MODEL_PATH = args.save_path
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                             trust_remote_code=True,
                                             torch_dtype="auto",
                                             device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

model.eval()

with open("example/qwen/calib_data/anti_prompt.json", "r") as file:
    anti_prompt = json.load(file)
with open("example/qwen/calib_data/calib_prompt.json", "r") as file:
    calib_prompt = json.load(file)


def get_anti_dataset(tokenizer_, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer_(calib_data, return_tensors='pt')
        calib_dataset.append(inputs.data['input_ids'].to(device))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    for i, dataset in enumerate(calib_dataset):
        calib_dataset[i] = F.pad(dataset, (0, max_len - dataset.size(1)), value=0)
    return torch.cat(calib_dataset)


def get_calib_dataset(tokenizer_, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer_(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device)])
    return calib_dataset

anti_data = []
for a_prompt in anti_prompt:
    tmp = get_anti_dataset(tokenizer, a_prompt)
    anti_data.append(tmp)

anti_dataset = []
for data in anti_data:
    anti_dataset.append([data])

dataset_calib = []
for c_prompt in calib_prompt:
    tmp = get_calib_dataset(tokenizer, c_prompt)
    dataset_calib += (tmp)

disable_names = [f"model.layers.{i}.mlp.down_proj" for i in range(0, 10)]
disable_names += [f"model.layers.{i}.mlp.down_proj" for i in range(30, 40)]
disable_names += [f"model.layers.{i}.mlp.down_proj" for i in range(70, 80)]

anti_config = AntiOutlierConfig(anti_method="m4", dev_type="npu")
anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
anti_outlier.process()

quant_config = QuantConfig(
                w_bit=8,
                a_bit=8,
                disable_names=disable_names,
                dev_type='npu',
                dev_id=model.device.index,
                act_method=1,
                pr=1.0,
                w_sym=True,
                mm_tensor=False,
                is_dynamic=False,
                use_kvcache_quant=True
               )
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])

SEQ_OUT_LEN = 100
TEST_PROMPT = "What is deep learning?\n"
test_input = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
model.eval()
generate_ids = model.generate(test_input.input_ids, attention_mask=test_input.attention_mask,
                              max_new_tokens=SEQ_OUT_LEN)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH, quant_config=quant_config)