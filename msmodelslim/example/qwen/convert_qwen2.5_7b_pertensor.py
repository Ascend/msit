# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import json
import logging
import argparse

import numpy as np
import torch
import functools
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, default="",
                        help="The path to model float weights")
    parser.add_argument("--save_path", type=str,
                        default="./",
                        help="The path to save quant weights")
    parser.add_argument("--anti_dataset", type=str, default="./anti_prompt.json", help="The prompts for anti outlier")
    parser.add_argument("--calib_dataset", type=str, default="./calib_prompt.json")
    parser.add_argument("--best_alpha", type=float, default=0.6, help="The best alpha of flex smooth")
    parser.add_argument("--best_beta", type=float, default=0.3, help="The best beta of flex smooth")
    parser.add_argument("--use_flex", type=bool, default=False)
    parser.add_argument("--kv_quant", default=False, action='store_true')
    parser.add_argument('--mix_select_layer', default=False, action="store_true")
    parser.add_argument('--mix_layer_alpha', type=float, default=0.5)
    parser.add_argument('--mix_layer_beta', type=float, default=1.0)
    parser.add_argument("--no_disable", action='store_true', help="If true, no layer will be disabled")
    parser.add_argument("--test_mode", action='store_true', help="If true, only 3 layer will be used")

    return parser.parse_args()


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


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


def get_calib_dataset(tokenizer, calib_list, device="npu"):  # device="npu:0" 如果需要使用npu进行量化
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to(device)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


def custom_hook(model_config):
    model_config['torch_dtype'] = 'float16'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.num_hidden_layers = 3 if args.test_mode else config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                                 config=config,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto',
                                                 device_map='auto')

    if 'qwen' in model.config.model_type:
        logging.info("Setup pad token for qwen model")
        tokenizer.pad_token = "<|endoftext|>"
    else:
        logging.info("Setup pad token for other model")
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 处理校准集
    with open(args.anti_dataset, "r") as file:
        anti_prompt = json.load(file)
    with open(args.calib_dataset, "r") as file:
        calib_prompt = json.load(file)

    anti_data = []
    for prompt in anti_prompt:
        tmp = get_anti_dataset(tokenizer, prompt)
        anti_data.append(tmp)

    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])

    dataset_calib = []
    for calib_prompt_item in calib_prompt:
        tmp = get_calib_dataset(tokenizer, calib_prompt_item)
        dataset_calib += (tmp)

    # msmodelslim量化
    # 启动flex smooth功能
    keys = ['.o_proj']
    disable_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            for key in keys:
                if key in name:
                    disable_names.append(name)

    anti_config = AntiOutlierConfig(anti_method="m6",
                                    dev_type='npu',
                                    use_kvcache_quant=False,
                                    disable_anti_names=disable_names,
                                    flex_config={'alpha': args.best_alpha, 'beta': args.best_beta,
                                                 'use_flex': args.use_flex})
    anti_outlier = AntiOutlier(model,
                               calib_data=anti_dataset,
                               cfg=anti_config)
    anti_outlier.process()

    # ========== quant ===================== #
    # get disable layer_names
    layers_name = []

    quant_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        disable_names=layers_name,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=False,
        w_method="KMeans",
        # 聚类类数
        lut_len=40,
        use_kvcache_quant=args.kv_quant,
    )

    if args.mix_select_layer:
        calibrator = None
        if args.mix_select_layer:
            logging.info(f"mix_select_layer, threshold1:{args.mix_layer_alpha}, threshold2{args.mix_layer_beta}")
            mix_cfg = {
                'mix_method': 'auto',
                'mix_types': ['w8a8s_static_kmeans', 'w8a8', 'float'],
                'threshold1': args.mix_layer_alpha,
                'threshold2': args.mix_layer_beta
            }
            calibrator = Calibrator(model,
                                    quant_config,
                                    calib_data=dataset_calib,
                                    mix_cfg=mix_cfg)
    else:
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

    # 转FP16
    safetensor_path = os.path.join(OUT_MODEL_PATH, 'quant_model_weight_w8a8s.safetensors')
    weight_1 = load_file(safetensor_path)

    for key in weight_1.keys():
        if 'deq_scale' in key:
            weight_1[key] = deqscale2int64(weight_1[key])

    save_file(weight_1, safetensor_path)

    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook),
    }
    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH,
                      quant_config=quant_config, custom_hooks=custom_hooks)

    # FP
    afetensor_path = os.path.join(OUT_MODEL_PATH, 'quant_model_weight_w8a8s.safetensors')
    safetensor_weight = load_file(safetensor_path)

    for key in safetensor_weight.keys():
        if 'deq_scale' in key:
            safetensor_weight[key] = deqscale2int64(safetensor_weight[key])

    save_file(safetensor_weight, safetensor_path)