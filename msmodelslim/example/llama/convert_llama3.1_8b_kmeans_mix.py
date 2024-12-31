# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import functools
import operator
import json
import logging
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file

from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
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


def custom_hook(model_config):
    model_config['torch_dtype'] = 'float16'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                                 config=config,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto',
                                                 device_map='auto')

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    with open("anti_prompt_llama8b_kmeans_mix.json", "r") as file:
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
        if isinstance(mod, torch.nn.Linear):
            for key in keys:
                if key in name:
                    disable_names.append(name)
    anti_config = AntiOutlierConfig(anti_method="m6", dev_type='npu',
                                    disable_anti_names=disable_names,
                                    flex_config={'alpha': 0.8, 'beta': 0.1})
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()

    disable_names = []

    with open("calib_prompt_llama8b_kmeans_mix.json", "r") as file:
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
        is_dynamic=False,
        w_method="KMeans",
        # 聚类类数
        lut_len=40
    )
    mix_cfg = {
        'mix_method': 'auto',
        'mix_types': ['w8a8s_static_kmeans', 'w8a8', 'float'],
        'threshold1': 0.5,
        'threshold2': 0.75,
    }
    # 当disable_level是dict类型时，启动层间混精，支持使用threshold设置阈值，或者使用disable_number直接设置按照从大到小选层数量
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, mix_cfg=mix_cfg)
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

    safetensor_path = os.path.join(OUT_MODEL_PATH, 'quant_model_weight_w8a8s.safetensors')
    weight_1 = load_file(safetensor_path)

    for key, item in weight_1.items():
        if 'deq_scale' in key:
            weight_1[key] = deqscale2int64(item)

        if key.split(".")[-1] in ["weight_scale"]:
            zeros = torch.zeros_like(item)
            weight_1[key] = torch.stack((item, zeros), dim=1).reshape(-1, 1)

    save_file(weight_1, safetensor_path)

    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook),
    }
    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH,
                      quant_config=quant_config, custom_hooks=custom_hooks)
