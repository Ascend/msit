# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import argparse
import torch
import json

from torch import nn
import torch.nn.functional as F  

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
    dataset = []
    with open(dataset_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data.get(key_name, line)
            dataset.append(text)
    return dataset

def get_calib_dataset(tokenizer, calib_list, device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to(device)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset

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
                                             torch_dtype=torch.float16,
                                             device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

model.eval()

dataset = load_jsonl("example/common/boolq.jsonl")
calib_data = get_calib_dataset(tokenizer, dataset, model.device)
disable_names = []

anti_config = AntiOutlierConfig(anti_method="m4", dev_type="npu")
anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
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
calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
calibrator.run()
calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])

SEQ_OUT_LEN = 100
TEST_PROMPT = "What is deep learning?\n"
test_input = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
model.eval()
generate_ids = model.generate(test_input.input_ids, attention_mask=test_input.attention_mask,
                              max_new_tokens=SEQ_OUT_LEN)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)