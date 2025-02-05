# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import argparse
import os
from unittest.mock import MagicMock

from ascend_utils.common.security import safe_copy_file, json_safe_dump, json_safe_load, get_valid_read_path, \
    get_valid_write_path


def copy_json(src_path: str, dst_path: str, quant_config):
    safe_copy_file(src_path, dst_path)


def modify_config_json(src_path: str, dst_path: str, quant_config, custom_hook=None):
    """
    复制config.json
    @param src_path: 源目录
    @param dst_path: 目标目录
    @param quant_config: 量化配置
    @param custom_hook: 自定义修改 custom_hook(model_config: dict)
    """
    model_config = json_safe_load(src_path)
    model_config['quantize'] = str(quant_config.model_quant_type.value).lower()
    model_config['quantization_config'] = {}
    if quant_config.use_kvcache_quant:
        model_config['quantization_config']['kv_quant_type'] = 'C8'
    if custom_hook:
        custom_hook(model_config)
    json_safe_dump(model_config, dst_path, indent=4)


EXCLUDING_SUBFIX_LIST = (
    'index.json',
)

FILE_HOOKS = {
    'config.json': modify_config_json,
}

DEFAULT_FILE_HOOKS = copy_json


def copy_config_files(input_path, output_path, quant_config, custom_hooks=None):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    @param quant_config: 量化配置
    @param custom_hooks: 自定义处理函数字典 {'file_name': hook} hook(src_path: str, dst_path: str, quant_config: QuantConfig)
    """
    for file in os.listdir(input_path):
        if not file.endswith('.json'):
            continue

        if any((file.endswith(subfix) for subfix in EXCLUDING_SUBFIX_LIST)):
            continue

        src_path = get_valid_read_path(os.path.join(input_path, file), extensions='.json')
        dst_path = get_valid_write_path(os.path.join(output_path, file))

        if custom_hooks and file in custom_hooks:
            hook = custom_hooks[file]
        else:
            hook = FILE_HOOKS.get(file, DEFAULT_FILE_HOOKS)
        hook(src_path, dst_path, quant_config)


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
