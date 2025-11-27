# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import argparse
import sys
import json
import shutil
import torch

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from safetensors.torch import load_file, save_file

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(parent_directory)

from example.common.utils import cmd_bool
from example.common.security.path import get_valid_read_path, get_write_directory
from example.common.vlm_utils import VlmSafeGenerator, ModifyConfigParams, CopyTokenizerParams
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


CPU = "cpu"
NPU = "npu"

KEY_MAPPING = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}


def convert_ckpt_panguvl(quant_path, model_dir):
    """Convert checkpoint for PanguVL model."""
    config_path = os.path.join(quant_path, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {quant_path}")

    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Modify field
    config['quantize'] = 'w8a8_dynamic'

    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    description_path = os.path.join(quant_path, 'quant_model_description.json')
    with open(description_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        new_data = {}
        for key, value in data.items():
            if "visual" in key and "kv_cache" in key:
                continue
            new_key = key.replace("model.", "")
            new_key = new_key.replace("language_layers", "openpangu.language_model.model.layers")
            new_data[new_key] = value
        new_data[".weight"] = "FLOAT"
        new_data["openpangu.language_model.lm_head.weight"] = "FLOAT"
        new_data['openpangu.language_model.model.embed_tokens.weight'] = "FLOAT"

        with open(description_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

    if os.path.exists(os.path.join(quant_path, "quant_model_weight_w8a8.safetensors")):
        ckpt_path = os.path.join(quant_path, "quant_model_weight_w8a8.safetensors")
    elif os.path.exists(os.path.join(quant_path, "quant_model_weight_w8a8_dynamic.safetensors")):
        ckpt_path = os.path.join(quant_path, "quant_model_weight_w8a8_dynamic.safetensors")
    else:
        raise FileNotFoundError("No quantized model weights found in the specified directory")

    state_dict = load_file(ckpt_path)
    keys_to_delete = []
    for k, v in state_dict.items():
        if ("norm" in k or 'ln' in k) and k.endswith(".bias"):
            if torch.all(v == 0):
                keys_to_delete.append(k)

    for k in keys_to_delete:
        state_dict.pop(k)

    save_path = ckpt_path.replace(".safetensors", "_clean.safetensors")
    if save_path.endswith(".safetensors"):
        save_file(state_dict, save_path)

    os.remove(ckpt_path)

    if os.path.exists(os.path.join(model_dir, "tokenizer.model")):
        filename = 'tokenizer.model'
    else:
        tiktoken_files = [f for f in os.listdir(model_dir) if f.endswith(".tiktoken")]

        if tiktoken_files:
            filename = tiktoken_files[0]
        else:
            raise FileNotFoundError("No tokenizer files found in the model directory")

    src_filepath = os.path.join(model_dir, filename)
    dest_filepath = os.path.join(quant_path, filename)
    shutil.copyfile(src_filepath, dest_filepath)
    os.chmod(dest_filepath, 0o600)


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--calib_images', type=str, default='/cache/cali_images')
    parser.add_argument('--save_directory', type=str, default='')
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=True)
    parser.add_argument('--anti_method', type=str, default='m2')
    parser.add_argument('--act_method', type=int, default=2)
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=False)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--group_size', type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument('--mindie_format', action="store_true", 
                       help="Compatible with quantization formats supported by MindIE")
    parser.add_argument('--use_kvcache_quant', type=cmd_bool, default=False)
    args = parser.parse_args()

    # check args
    args.model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    args.calib_images = get_valid_read_path(args.calib_images, is_dir=True, check_user_stat=True)
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    # 1. Load model
    device_map = CPU if args.device_type == CPU else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map=device_map, 
        trust_remote_code=True,
        torch_dtype="auto", 
        local_files_only=True,
        key_mapping=KEY_MAPPING
    ).eval()
    config = model.config

    # 2. Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        local_files_only=True, 
        trust_remote_code=True
    )

    # 3. Set fallback layers
    disable_names = []
    vision_name = [
        'model.visual.vision_projection.fc1', 
        'model.visual.merger.0.mlp.0',
        'model.visual.merger.0.mlp.2', 
        'model.visual.merger.1.mlp.0', 
        'model.visual.merger.1.mlp.2'
    ]
    llm_name = []
    disable_names.extend(vision_name)
    disable_names.extend(llm_name)

    # 4. Load calibration dataset
    images_list = os.listdir(args.calib_images)
    calib_data = []
    for i in images_list:
        image_path = os.path.join(args.calib_images, i)
        image_path = get_valid_read_path(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": "Please describe this picture in detail."
                    },
                ]
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors='pt'
        ).to(args.device_type)

        calib_data.append([
            inputs['input_ids'], 
            inputs['attention_mask'],
            None, None, None, None, None, None, None, 
            inputs['pixel_values'], 
            None, 
            inputs['image_grid_thw'], 
            None, None, None, None
        ])

    # 5. Outlier suppression
    anti_config = AntiOutlierConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        anti_method=args.anti_method,
        dev_type=args.device_type,
        dev_id=model.device.index,
    )
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
    anti_outlier.process()

    # 6. Model quantization
    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=model.device.index,
        act_method=args.act_method,
        mm_tensor=False,
        open_outlier=args.open_outlier,
        use_kvcache_quant=args.use_kvcache_quant,
        is_dynamic=args.is_dynamic,
        is_lowbit=args.is_lowbit,
        group_size=args.group_size
    )
    calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
    calibrator.run()

    # 7. Save weights
    calibrator.save(args.save_directory, save_type=["ascendV1"], part_file_size=args.part_file_size)

    quant_type = quant_config.model_quant_type.lower()
    checker = VlmSafeGenerator()
    auto_config = checker.get_config_from_pretrained(args.model_path, trust_remote_code=True)

    # Use dataclass parameters
    modify_params = ModifyConfigParams(
        model_dir=args.model_path,
        dest_dir=args.save_directory,
        torch_dtype=auto_config.torch_dtype,
        quantize_type=quant_type,
        args=args,
        quantize_config_parts=['vision_config']
    )
    checker.modify_config(modify_params)

    copy_params = CopyTokenizerParams(
        model_dir=args.model_path,
        dest_dir=args.save_directory
    )
    checker.copy_tokenizer_files(copy_params)
    convert_ckpt_panguvl(args.save_directory, args.model_path)


if __name__ == '__main__':
    main()