import subprocess

import os
import json
import time
import argparse
import torch
import torch_npu
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_type", type=str, default="qwen", help="The path to save quant weights")
    parser.add_argument("--model_name", type=str, default="qwen2.5_72b", help="The path to save quant weights")
    parser.add_argument("--data_type", type=str, default="bf16", help="The path to save quant weights")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="The path to save quant weights")
    parser.add_argument("--quantize", type=str, default="w8a8_pdmix", help="The path to save quant weights")
    parser.add_argument("--model_path", type=str,
                        default="",
                        help="The path to model float weights")
    parser.add_argument("--save_path", type=str, default="./save_quant", help="The path to save quant weights")
    parser.add_argument("--anti_method", type=str, default="m6", help="help anti_method")
    parser.add_argument("--anti_prompt", type=str, default="", help="The path to load jsonl")
    parser.add_argument("--calib_prompt", type=str, default="", help="The path to load jsonl")
    parser.add_argument("--act_method", type=int, default=1, help="The path to save quant weights")
    parser.add_argument("--layer_count", type=int, default=0, help="Layer count for quant")
    parser.add_argument("--msmodelslim_path", type=str, default="", help="The path of msmodelsim to use")
    parser.add_argument("--disable_name_path", type=str, default="", help="The path of msmodelsim to use")
    parser.add_argument("--use_kv_quant", type=bool, default=True, help="The path of msmodelsim to use")
    parser.add_argument("--best_alpha", type=float, default=0.6, help="The path of msmodelsim to use")
    parser.add_argument("--best_beta", type=float, default=0.3, help="The path of msmodelsim to use")
    parser.add_argument("--is_dynamic", type=bool, default=False, help="The path of msmodelsim to use")
    return parser.parse_args()


args = parse_args()

print(args)

os.environ['BEST_ALPHA'] = str(args.best_alpha)
os.environ['BEST_BETA'] = str(args.best_beta)

if args.msmodelslim_path:
    print(f'Use msmodelslim in {args.msmodelslim_path}')
    sys.path.insert(0, args.msmodelslim_path)

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

import torch.nn.functional as F

IN_MODEL_PATH = args.model_path
OUT_MODEL_PATH = args.save_path

os.system(f'mkdir -p {args.save_path}')
assert os.path.exists(args.save_path)
os.system(f'cp {args.model_path}/*.json {args.save_path}')
assert os.path.exists(os.path.join(args.save_path, 'config.json'))
os.system(f'rm {args.save_path}/*.index.json')
os.system(f'rm {args.save_path}/quant_model_weight*.safetensors -f')
os.system(f'rm {args.save_path}/quant_model_description*.json -f')

t1 = time.time()
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
config.num_hidden_layers = config.num_hidden_layers if args.layer_count == 0 else args.layer_count
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, trust_remote_code=True,
                                             config=config, torch_dtype='auto', device_map='auto')

tokenizer.pad_token = tokenizer.eos_token

model.eval()

print('>>>model:', model)
print('>>> model load cost time:', time.time() - t1)


def get_anti_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append(
            inputs.data['input_ids'].to(device))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    for i in range(len(calib_dataset)):
        calib_dataset[i] = F.pad(calib_dataset[i], (0, max_len - calib_dataset[i].size(1)), value=0)
    return torch.cat(calib_dataset)


def get_calib_dataset(tokenizer, calib_list, device=model.device):  # device="npu:0" 如果需要使用npu进行量化
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to(device)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


print("calib data from flex")
with open(args.anti_prompt, "r") as file:
    anti_prompt = json.load(file)
with open(args.calib_prompt, "r") as file:
    calib_prompt = json.load(file)

anti_data = []
for i in range(len(anti_prompt)):
    tmp = get_anti_dataset(tokenizer, anti_prompt[i])
    anti_data.append(tmp)

anti_dataset = []
for data in anti_data:
    anti_dataset.append([data])

calib_dataset = []
for i in range(len(calib_prompt)):
    tmp = get_calib_dataset(tokenizer, calib_prompt[i])
    calib_dataset += (tmp)

## ========== anti ===================== ##
# msmodelslim量化
t1 = time.time()
anti_config = AntiOutlierConfig(anti_method=args.anti_method, dev_type='npu')
print('=' * 36)
print('anti config: ', anti_config.__dict__)
print('=' * 36)
anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
anti_outlier.process()
print('>>> anti cost time:', time.time() - t1)

# ========== quant ===================== ##
t1 = time.time()
disable_names = []
disable_names.append("lm_head")

with open(args.disable_name_path, "r") as file:
    disable_names_json = json.load(file)
    for disable_name in disable_names_json:
        disable_names.append(disable_name)

print("disable_names: ")
_ = [print(name) for name in disable_names]

quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=args.act_method,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=args.is_dynamic,
    use_kvcache_quant=args.use_kv_quant
)
print('=' * 36)
print('quant config: ', quant_config.__dict__)
print('=' * 36)
calibrator = Calibrator(model, quant_config, calib_data=calib_dataset, disable_level='L0')
calibrator.run()
calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])
print('>>> quant cost time:', time.time() - t1)

with open(os.path.join(args.save_path, 'config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)

if args.quantize:
    config['quantize'] = args.quantize

if args.torch_dtype:
    config['torch_dtype'] = args.torch_dtype

if args.use_kv_quant:
    config['quantization_config'] = {}
    config['quantization_config']['kv_quant_type'] = "C8"

with open(os.path.join(args.save_path, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
