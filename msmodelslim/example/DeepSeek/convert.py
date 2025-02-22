#Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
from easy_quant import DataFreeConverter
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to quant model")
    parser.add_argument("--model_path", type=str, help="The path of the model to be converted")
    parser.add_argument("--save_path", type=str, help="The path to save the converted model")
    return parser.parse_args()


def main():
    args = parse_args()
    disable_names = ["lm_head"]
    disable_names += [f"model.layers.{i}.self_attn.q_a_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.q_b_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.kv_a_proj_with_mqa" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.kv_b_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.o_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.mlp.gate" for i in range(62)]
    quant_config = QuantConfig(
        w_bit=4,
        a_bit=16,
        disable_names=disable_names,
        w_sym=True,
        mm_tensor=False,
        group_size=64,
        is_lowbit=True,
        open_outlier=False,
        dev_type="npu",
    )

    converter = DataFreeConverter(quant_config)
    converter.convert(args.model_path, args.save_path)

if __name__ == "__main__":
    main()
