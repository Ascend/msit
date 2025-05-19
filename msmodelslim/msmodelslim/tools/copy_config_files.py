# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import argparse
import os
import glob
from unittest.mock import MagicMock

from ascend_utils.common.security import safe_copy_file, json_safe_dump, json_safe_load, get_valid_read_path, \
    get_valid_write_path


def copy_file(src_path: str, dst_path: str, quant_config, mindie_format: bool):
    safe_copy_file(src_path, dst_path)


def modify_config_json(src_path: str, dst_path: str, quant_config, mindie_format: bool, custom_hook=None):
    """
    复制config.json
    @param src_path: 源目录
    @param dst_path: 目标目录
    @param quant_config: 量化配置
    @param custom_hook: 自定义修改 custom_hook(model_config: dict)
    """
    model_config = json_safe_load(src_path)
    model_config['quantize'] = str(quant_config.model_quant_type.value).lower()
    
    if mindie_format:
        matched_files = glob.glob(os.path.join(dst_path.split("config.json")[0], \
                                    f"quant_model_description*.json"))
        dest_quant_description_filepath = matched_files[0]
    else:
        dest_quant_description_filepath = os.path.join(dst_path.split("config.json")[0], \
                                                        f"quant_model_description.json")
    dest_quant_description_filepath = get_valid_write_path(dest_quant_description_filepath, is_dir=False)
    quant_description_data = json_safe_load(dest_quant_description_filepath, check_user_stat=False)

    quantization_config = quant_description_data
    if quant_config.use_kvcache_quant:
        quantization_config['kv_quant_type'] = "C8"
    else:
        quantization_config['kv_quant_type'] = None
    
    if quant_config.use_fa_quant:
        quantization_config['use_fa_quant'] = "FAQuant"
    else:
        quantization_config['use_fa_quant'] = None

    quantization_config['group_size'] = quant_config.group_size

    if mindie_format:
        model_config['quantization_config'] = quantization_config
    else:
        json_safe_dump(quantization_config, dest_quant_description_filepath, indent=4)
    
    if custom_hook:
        custom_hook(model_config)
    json_safe_dump(model_config, dst_path, indent=4)


EXCLUDING_SUBFIX_LIST = (
    'index.json',
)

FILE_HOOKS = {
    'config.json': modify_config_json,
}

DEFAULT_FILE_HOOKS = copy_file


def copy_config_files(input_path, output_path, quant_config, mindie_format, custom_hooks=None):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    @param quant_config: 量化配置
    @param custom_hooks: 自定义处理函数字典 {'file_name': hook} hook(src_path: str, dst_path: str, quant_config: QuantConfig)
    """
    for file in os.listdir(input_path):
        if not (file.endswith('.json') or file.endswith('.py')):
            continue

        if any((file.endswith(subfix) for subfix in EXCLUDING_SUBFIX_LIST)):
            continue

        src_path = get_valid_read_path(os.path.join(input_path, file), extensions=['.json', '.py'])
        dst_path = get_valid_write_path(os.path.join(output_path, file))

        if custom_hooks and file in custom_hooks:
            hook = custom_hooks[file]
        else:
            hook = FILE_HOOKS.get(file, DEFAULT_FILE_HOOKS)
        hook(src_path, dst_path, quant_config, mindie_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy Config Files')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--quant_type', type=str, help='quant type')
    parser.add_argument('--use_kvcache_quant', action='store_true', help='use kvcache quant')
    args = parser.parse_args()

    model_path = get_valid_read_path(args.model_path, is_dir=True)
    save_path = get_valid_write_path(args.save_path, is_dir=True)
    quant_type = MagicMock()
    quant_type.value = args.quant_type
    quant_config = MagicMock()
    quant_config.model_quant_type = quant_type
    quant_config.use_kvcache_quant = args.use_kvcache_quant

    copy_config_files(input_path=model_path, output_path=save_path, quant_config=quant_config)
