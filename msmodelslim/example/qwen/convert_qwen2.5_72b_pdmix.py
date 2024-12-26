import os
import json
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tools.copy_config_files import copy_config_files
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, help="The path to save quant weights")
    parser.add_argument("--anti_dataset", type=str, default="./anti_dataset.json", help="The prompts for anti outlier")
    parser.add_argument("--calib_dataset", type=str, default="./calib_dataset.json")
    parser.add_argument("--best_alpha", type=float, default=0.6000000238418579, help="The best alpha of flex smooth")
    parser.add_argument("--best_beta", type=float, default=0.30000001192092896, help="The best beta of flex smooth")
    parser.add_argument("--use_flex", type=bool, default=False, help="The best beta of flex smooth")
    parser.add_argument("--kv_quant", action='store_true')
    parser.add_argument("--no_disable", action='store_true', help="If true, no layer will be disabled")
    parser.add_argument("--test_mode", action='store_true', help="If true, only 1 layer will be used")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.num_hidden_layers = 1 if args.test_mode else config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                            trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH,
                                                trust_remote_code=True,
                                                config=config,
                                                torch_dtype='auto',
                                                device_map='auto')

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

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


    def get_calib_dataset(tokenizer, calib_list, device="npu"):  # device="npu:0" 如果需要使用npu进行量化
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
        tmp = get_calib_dataset(tokenizer, calib_prompt[i])
        dataset_calib += (tmp)

    # msmodelslim量化
    # 启动flex smooth功能
    keys = ['.o_proj']
    disable_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            for key in keys:
                if key in name:
                    disable_names.append(name)
    anti_config = AntiOutlierConfig(anti_method='m6',
                                    dev_type='npu',
                                    dev_id=model.device.index,
                                    use_kvcache_quant=args.kv_quant,
                                    disable_anti_names=disable_names,
                                    flex_config={'alpha': args.best_alpha, 'beta': args.best_beta, 'use_flex' : args.use_flex})
    anti_outlier = AntiOutlier(model,
                            calib_data=anti_dataset,
                            cfg=anti_config,
                            )
    anti_outlier.process()

    # ========== quant ===================== ##
    # get disable layer_names
    disable_names = ['model.layers.0.mlp.down_proj',
                        'model.layers.1.mlp.down_proj',
                        'model.layers.2.mlp.down_proj',
                        'model.layers.79.mlp.down_proj',
                        'model.layers.5.mlp.down_proj',
                        'model.layers.6.mlp.down_proj',
                        'model.layers.8.mlp.down_proj',
                        'model.layers.4.mlp.down_proj',
                        'model.layers.9.mlp.down_proj',
                        'model.layers.78.mlp.down_proj',
                        'model.layers.11.mlp.down_proj',
                        'model.layers.23.mlp.down_proj',
                        'model.layers.7.mlp.down_proj',
                        'model.layers.76.mlp.down_proj',
                        'model.layers.31.mlp.down_proj',
                        'model.layers.3.mlp.down_proj',
                        'model.layers.10.mlp.down_proj']
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
        use_kvcache_quant=args.kv_quant,
        lut_len=22
    )

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()

    # 伪量化对话
    SEQ_LEN_OUT = 100
    print("testing quant weights...")
    TEST_PROMPT = "What is deep learning?\n"
    test_input = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    print("model is inferring...")
    model.eval()
    generate_ids = model.generate(test_input.input_ids,
                                  attention_mask=test_input.attention_mask,
                                  max_new_tokens=SEQ_LEN_OUT)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(res)
    for result in res:
        print(result)
    calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])

    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH, quant_config=quant_config)