import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, help="The path to save quant weights")
    parser.add_argument("--anti_prompt", type=str, default="./anti_prompt_c8.json", help="The prompts for anti outlier")
    parser.add_argument("--calib_prompt", type=str, default="./calib_prompt_c8.json",
                        help="The prompts for anti outlier")
    parser.add_argument("--best_alpha", type=float, default=0.6000000238418579, help="The best alpha of flex smooth")
    parser.add_argument("--best_beta", type=float, default=0.30000001192092896, help="The best beta of flex smooth")
    parser.add_argument("--no_disable", action='store_true', help="If true, no layer will be disabled")
    parser.add_argument("--test_mode", action='store_true', help="If true, only 1 layer will be used")
    return parser.parse_args()


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


def get_calib_dataset(tokenizer, calib_list, device=model.device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to(device)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


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

anti_config = AntiOutlierConfig(anti_method='m6',
                                dev_type='npu',
                                dev_id=model.device.index,
                                flex_config={'alpha': args.best_alpha, 'beta': args.best_beta})
anti_outlier = AntiOutlier(model,
                           calib_data=anti_dataset,
                           cfg=anti_config,
                           )
anti_outlier.process()

# ========== quant ===================== ##
disable_names = [
    "model.layers.1.self_attn.v_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.k_proj",
    "model.layers.2.self_attn.v_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.5.self_attn.v_proj",
    "model.layers.5.self_attn.q_proj",
    "model.layers.5.self_attn.k_proj",
    "model.layers.2.mlp.down_proj",
    "model.layers.4.self_attn.v_proj",
    "model.layers.4.self_attn.q_proj",
    "model.layers.4.self_attn.k_proj",
    "model.layers.9.self_attn.v_proj",
    "model.layers.9.self_attn.q_proj",
    "model.layers.9.self_attn.k_proj",
    "model.layers.79.mlp.down_proj",
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
calibrator.save(OUT_MODEL_PATH, save_type=["safe_tensor"])
