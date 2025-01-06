# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import json
import logging
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file
from safetensors.torch import save_file
from msmodelslim.tools.copy_config_files import copy_config_files
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, help="The path to save quant weights")
    return parser.parse_args()


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append([inputs.data['input_ids'].to(device)])
    return calib_dataset


def get_anti_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append(
            inputs.data['input_ids'].to(device))
        max_len = max(max_len, inputs.data['input_ids'].size(1)) 
    new_calib_dataset = []
    for inputs in calib_dataset:
        new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
        new_calib_dataset.append(new_inputs)
    return torch.cat(new_calib_dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, 
                                                trust_remote_code=True, 
                                                torch_dtype='auto', 
                                                device_map='auto').bfloat16()

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    with open("example/llama/calib_data/anti_prompt_llama8b.json", "r") as file:
        anti_prompt = json.load(file)

    anti_data = []
    for prompt in anti_prompt:
        tmp = get_anti_dataset(tokenizer, prompt)
        anti_data.append(tmp)
        
    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])

    # msmodelslim量化
    # 启动flex smooth功能
    keys = ['.o_proj']
    disable_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear): 
            for key in keys:
                if key in name:
                    disable_names.append(name)
    anti_config = AntiOutlierConfig(anti_method="m6", dev_type='npu', disable_anti_names=disable_names)
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()
    model.anti_method = 'm1'

    disable_names = ['model.layers.31.mlp.down_proj', 
                    'model.layers.0.self_attn.v_proj', 
                    'model.layers.0.self_attn.q_proj', 
                    'model.layers.0.self_attn.k_proj', 
                    'model.layers.1.mlp.down_proj', 
                    'model.layers.0.mlp.down_proj', 
                    'model.layers.27.mlp.down_proj', 
                    'model.layers.2.mlp.down_proj', 
                    'model.layers.25.mlp.down_proj', 
                    'model.layers.24.mlp.down_proj', 
                    'model.layers.30.mlp.down_proj', 
                    'model.layers.21.mlp.down_proj', 
                    'model.layers.23.mlp.down_proj', 
                    'model.layers.18.mlp.down_proj', 
                    'model.layers.22.mlp.down_proj', 
                    'model.layers.29.mlp.down_proj', 
                    'model.layers.19.mlp.down_proj', 
                    'model.layers.26.mlp.down_proj', 
                    'model.layers.20.mlp.down_proj', 
                    'model.layers.3.mlp.down_proj', 
                    'model.layers.28.mlp.down_proj', 
                    'model.layers.16.mlp.down_proj', 
                    'model.layers.23.self_attn.o_proj', 
                    'model.layers.17.mlp.down_proj', 
                    'model.layers.4.mlp.down_proj', 
                    'model.layers.5.mlp.down_proj', 
                    'model.layers.10.mlp.down_proj', 
                    'model.layers.14.mlp.down_proj', 
                    'model.layers.15.mlp.down_proj']

    with open("example/llama/calib_data/calib_prompt_llama8b.json", "r") as file:
        calib_prompt = json.load(file)
    dataset_calib = []
    for calib_prompt_item in calib_prompt:
        tmp = get_calib_dataset(tokenizer, calib_prompt_item)
        dataset_calib += (tmp)
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
                    is_dynamic=False
                )
    # 当disable_level是dict类型时，启动层间混精，支持使用threshold设置阈值，或者使用disable_number直接设置按照从大到小选层数量
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()
    calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])

    # 伪量化对话
    SEQ_LEN_OUT = 100
    logging.info("testing quant weights...")
    TEST_PROMPT = "What is deep learning?\n"
    test_input = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    logging.info("model is inferring...")
    model.eval()
    generate_ids = model.generate(test_input.input_ids,
                                  attention_mask=test_input.attention_mask,
                                  max_new_tokens=SEQ_LEN_OUT)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logging.info(res)
    for result in res:
        logging.info(result)

    safetensor_path = os.path.join(OUT_MODEL_PATH, 'quant_model_weight_w8a8.safetensors')
    weight_1 = load_file(safetensor_path)

    for key in weight_1.keys():
        if 'deq_scale' in key and weight_1[key].dtype != torch.int64:
            weight_1[key] = deqscale2int64(weight_1[key])


    save_file(weight_1, safetensor_path)
    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH, quant_config=quant_config)