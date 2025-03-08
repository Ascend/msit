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

    # MLA
    disable_names += [f"model.layers.{i}.self_attn.q_a_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.q_b_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.kv_a_proj_with_mqa" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.kv_b_proj" for i in range(62)]
    disable_names += [f"model.layers.{i}.self_attn.o_proj" for i in range(62)]
    
    # MOE gate
    disable_names += [f"model.layers.{i}.mlp.gate" for i in range(62)]

    # Dense
    disable_names += [f"model.layers.{i}.mlp.up_proj" for i in range(3)]
    disable_names += [f"model.layers.{i}.mlp.down_proj" for i in range(3)]
    disable_names += [f"model.layers.{i}.mlp.gate_proj" for i in range(3)]

    # shared_experts
    disable_names += [f"model.layers.{i}.mlp.shared_experts.down_proj" for i in range(61)]

    default_quant_config = QuantConfig(
        w_bit=4,
        a_bit=16,
        disable_names=disable_names,
        w_sym=True,
        mm_tensor=False,
        group_size=64,
        is_lowbit=True,
        open_outlier=False,
        w_method="HQQ",
        dev_type="npu",
    )
    # MTP
    disable_names = ["model.layers.61.mlp.gate", "model.layers.61.shared_head.head", "model.layers.61.eh_proj"]
    mtp_quant_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        disable_names=disable_names,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=True,
        dev_type="npu",
    )
    cfg_dict = {"default": default_quant_config, "model.layers.61.": mtp_quant_config}
    
    converter = DataFreeConverter(cfg_dict)
    converter.convert(args.model_path, args.save_path)

if __name__ == "__main__":
    main()
