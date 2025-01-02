# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import functools
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, help="The path to save quant weights")
    parser.add_argument("--anti_dataset", type=str, default="./anti_prompt_72b_pdmix.json",
                        help="The prompts for anti outlier")
    parser.add_argument("--calib_dataset", type=str, default="./calib_prompt_72b_pdmix.json",
                        help="The prompts for calibrator")
    parser.add_argument("--best_alpha", type=float, default=0.75, help="The best alpha of flex smooth")
    parser.add_argument("--best_beta", type=float, default=0.1, help="The best beta of flex smooth")
    parser.add_argument("--use_flex", type=bool, default=False, help="If true, will find flex alpha and beta automatic")
    parser.add_argument("--no_osmooth", type=bool, default=True,
                        help="The true, will disable o_proj smooth when doing anti outlier")
    parser.add_argument("--kv_quant", type=bool, default=True, help="The true, will use kvcache quant")
    parser.add_argument("--kv_smooth", type=bool, default=True, help="The true, will enable kvcache smooth")
    parser.add_argument("--act_method", type=int, default=2,
                        help="The activation observer type, 1 is min-max, 2 is histogram, 3 is auto")
    parser.add_argument("--run_fake_quant", action='store_true', help="If true, will run fake quant after save")
    parser.add_argument("--no_disable", action='store_true', help="If true, no layer will be disabled")
    parser.add_argument("--test_mode", action='store_true', help="If true, only {layer_count} layer will be used")
    parser.add_argument("--layer_count", type=int, default=1, help="Layer to load for test mode")
    parser.add_argument("--fp16", action='store_true', help="If true, will load as fp16 model and save as fp16 model")
    return parser.parse_args()


def run_fake_quant(model, tokenizer):
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


if __name__ == "__main__":
    args = parse_args()

    IN_MODEL_PATH = args.model_path
    OUT_MODEL_PATH = args.save_path

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if args.fp16:
        config.torch_dtype = torch.float16

    config.num_hidden_layers = args.layer_count if args.test_mode else config.num_hidden_layers
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
    anti_disable_names = []

    if args.no_osmooth:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                for key in keys:
                    if key in name:
                        anti_disable_names.append(name)

    anti_config = AntiOutlierConfig(
        anti_method='m6',
        dev_type='npu',
        dev_id=model.device.index,
        use_kvcache_quant=args.kv_smooth,
        disable_anti_names=anti_disable_names,
        flex_config={'alpha': args.best_alpha,
                     'beta': args.best_beta,
                     'use_flex': args.use_flex}
    )
    anti_outlier = AntiOutlier(
        model,
        calib_data=anti_dataset,
        cfg=anti_config,
    )
    anti_outlier.process()

    # ========== quant ===================== ##
    # get disable layer_names
    disable_names = [
        'model.layers.0.mlp.down_proj',
        'model.layers.1.mlp.down_proj',
        'model.layers.2.mlp.down_proj',
        'model.layers.79.mlp.down_proj',
    ]

    if args.no_disable:
        disable_names = []

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
        is_dynamic=False,
        use_kvcache_quant=args.kv_quant,
    )

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()
    calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])

    if args.run_fake_quant:
        run_fake_quant(model, tokenizer)

    if args.fp16:
        def modify_torch_dtype_float16(model_config):
            model_config['torch_dtype'] = 'float16'


        custom_hooks = {
            'config.json': functools.partial(modify_config_json, custom_hook=modify_torch_dtype_float16),
        }

    copy_config_files(input_path=IN_MODEL_PATH, output_path=OUT_MODEL_PATH, quant_config=quant_config)
