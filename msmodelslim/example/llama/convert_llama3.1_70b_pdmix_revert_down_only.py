import argparse
import json
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tools.copy_config_files import copy_config_files
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Create quant weights.")
    parser.add_argument("--model_path", type=str, help="The path of float model")
    parser.add_argument("--save_path", type=str, help="The path of quant model to save")
    parser.add_argument("--anti_prompt", type=str, default="./calib_data/anti_prompt_revert_down_only.json",
                        help="The prompts for anti outlier")
    parser.add_argument("--calib_prompt", type=str, default="./calib_data/calib_prompt_revert_down_only.json",
                        help="The prompts for anti outlier")
    parser.add_argument("--best_alpha", type=float, default=0.75, help="The best alpha of flex smooth")
    parser.add_argument("--best_beta", type=float, default=0.1, help="The best beta of flex smooth")
    parser.add_argument("--use_flex", type=bool, default=False, help="If true, will enable auto flex")
    parser.add_argument("--no_disable", action='store_true', help="If true, no layer will be disabled")
    parser.add_argument("--test_mode", action='store_true', help="If true, only 1 layer will be use")
    return parser.parse_args()


def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset


def get_anti_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append(
            inputs.data['input_ids'].to(device)
        )
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    new_calib_dataset = []
    for inputs in calib_dataset:
        new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
        new_calib_dataset.append(new_inputs)
    return torch.cat(new_calib_dataset)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    args = parse_args()
    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.num_hidden_layers = 1 if args.test_mode else config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path,
                                              trust_remote_code=True
                                              )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_path,
                                                 trust_remote_code=True,
                                                 config=config,
                                                 torch_dtype='auto',
                                                 device_map='auto')
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.anti_prompt, 'r', encoding='utf-8') as f:
        anti_prompt = json.load(f)

    anti_data = []
    for prompt in anti_prompt:
        tmp = get_anti_dataset(tokenizer, prompt)
        anti_data.append(tmp)

    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])

    with open(args.calib_prompt, 'r') as f:
        calib_prompt = json.load(f)

    calib_dataset = []
    for prompt in calib_prompt:
        tmp = get_calib_dataset(tokenizer, prompt)
        calib_dataset += (tmp)

    disable_names = []
    anti_config = AntiOutlierConfig(anti_method='m6', dev_type='npu', dev_id=model.device.index,
                                    disable_anti_names=disable_names,
                                    flex_config={'alpha': args.best_alpha, 'beta': args.best_beta,
                                                 'use_flex': args.use_flex})
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()

    disable_names = [
        'model.layers.3.mlp.down_proj',
        'model.layers.1.mlp.down_proj',
        'model.layers.6.mlp.down_proj',
        'model.layers.0.mlp.down_proj',
        'model.layers.79.mlp.down_proj',
        'model.layers.69.mlp.down_proj',
        'model.layers.56.mlp.down_proj',
        'model.layers.76.mlp.down_proj',
        'model.layers.4.mlp.down_proj',
        'model.layers.68.mlp.down_proj',
        'model.layers.43.mlp.down_proj',
        'model.layers.71.mlp.down_proj',
        'model.layers.45.mlp.down_proj',
        'model.layers.41.mlp.down_proj',
        'model.layers.14.mlp.down_proj',
        'model.layers.2.mlp.down_proj',
        'model.layers.47.mlp.down_proj',
    ]

    if args.no_disable:
        disable_names = []

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
        use_kvcache_quant=True,
    )

    calibrator = Calibrator(model, quant_config, calib_data=calib_dataset, disable_level='L0')
    calibrator.run()
    calibrator.save(OUT_MODEL_PATH, save_type=['safe_tensor'])
    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH, quant_config=quant_config)
