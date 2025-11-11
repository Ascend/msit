#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import sys
from typing import List
import torch
import torch.utils.data
import torch.nn as nn

from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.iter_smooth import IterSmoothProcessor, IterSmoothProcessorConfig
from msmodelslim.core.graph.adapter_types import MappingConfig, AdapterConfig
from msmodelslim.quant.processor.anti_outlier.smooth_interface import IterSmoothInterface

SEQ_LEN_OUT = 32
KEY_INPUT_IDS = "input_ids"
KEY_ATTENTION_MASK = "attention_mask"
STR_TEST_PROMPT = "Hello world"
RETURN_TENSOR_TYPE = "pt"


def test_iter_smooth_processor_with_hooks():
    """
    测试 IterSmoothProcessor 功能，包括统计钩子的收集和比较处理前后的区别
    """

    try:
        # 获取模型和分词器
        model, tokenizer = get_fake_llama_model_and_tokenizer()

        # 将模型设置为评估模式并禁用梯度计算
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # 创建测试提示词
        test_prompt = tokenizer(STR_TEST_PROMPT, return_tensors=RETURN_TENSOR_TYPE, padding=True, truncation=True)

        # 创建 IterSmoothProcessorConfig
        iter_smooth_config = IterSmoothProcessorConfig(
            type="iter_smooth",
            alpha=0.9,
            enable_subgraph_type=["norm-linear", "linear-linear", "ov", "up-down"],
            include=[],
            exclude=[],
            subgraph_priority={
                "up-down": 1,
                "ov": 2,
                "norm-linear": 3,
                "linear-linear": 4
            }
        )

        # 创建 ModelAdapter
        class IterSmoothModelAdapter(IterSmoothInterface):
            def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
                adapter_config = []
                for layer_idx in range(1):
                    # Norm-Linear融合的映射配置1：输入层归一化到QKV投影
                    norm_linear_mapping_config1 = MappingConfig(
                        source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                        targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                                 f"model.layers.{layer_idx}.self_attn.q_proj",
                                 f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
                    )

                    # Norm-Linear融合的映射配置2：后注意力层归一化到MLP投影
                    norm_linear_mapping_config2 = MappingConfig(
                        source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                        targets=[f"model.layers.{layer_idx}.mlp.gate_proj",
                                 f"model.layers.{layer_idx}.mlp.up_proj"]  # MLP层的门控和上投影
                    )

                    # OV融合的映射配置（QKV到输出投影）
                    ov_mapping_config = MappingConfig(
                        source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                        targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
                    )

                    # Up-Down融合的映射配置
                    up_down_mapping_config = MappingConfig(
                        source=f"model.layers.{layer_idx}.mlp.up_proj",  # 上投影层
                        targets=[f"model.layers.{layer_idx}.mlp.down_proj"]  # 下投影层
                    )

                    # 为当前layer添加4个配置
                    adapter_config.extend([
                        AdapterConfig(
                            subgraph_type="norm-linear",
                            mapping=norm_linear_mapping_config1
                        ),
                        AdapterConfig(
                            subgraph_type="norm-linear",
                            mapping=norm_linear_mapping_config2
                        ),
                        AdapterConfig(
                            subgraph_type="ov",
                            mapping=ov_mapping_config
                        ),
                        AdapterConfig(
                            subgraph_type="up-down",
                            mapping=up_down_mapping_config
                        )
                    ])
                return adapter_config

        adapter = IterSmoothModelAdapter()
        # 创建 IterSmoothProcessor 实例
        iter_smooth_processor = IterSmoothProcessor(model, iter_smooth_config, adapter)

        msmodelslim_logger.info("IterSmoothProcessor 创建成功！")
        msmodelslim_logger.info(f"处理器类型: {iter_smooth_processor.config.type}")
        msmodelslim_logger.info(f"Alpha 值: {iter_smooth_processor.config.alpha}")
        msmodelslim_logger.info(f"支持的子图类型: {iter_smooth_processor.config.enable_subgraph_type}")

        # 检查模型是否设置了 anti_method 属性
        if hasattr(model, 'anti_method'):
            msmodelslim_logger.info(f"模型 anti_method 属性: {model.anti_method}")
        else:
            msmodelslim_logger.warning("模型未设置 anti_method 属性")

        # 获取校准数据集
        dataset_calib = [[test_prompt[KEY_INPUT_IDS], test_prompt.data[KEY_ATTENTION_MASK]]]

        # 测试pre_run阶段 - 加载全局子图配置
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("开始pre_run阶段 - 加载全局子图配置")
        msmodelslim_logger.info("=" * 50)

        # 调用pre_run加载全局适配器配置
        iter_smooth_processor.pre_run()
        msmodelslim_logger.info("全局子图配置加载完成")

        # 测试preprocess阶段 - 安装统计钩子
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("开始preprocess阶段 - 安装统计钩子")
        msmodelslim_logger.info("=" * 50)

        # 创建BatchProcessRequest - 使用与helper_test_anti_outlier_numeric相同的测试数据
        request = BatchProcessRequest(
            module=model,
            name="model.layers.0",
            datas=dataset_calib  # 使用相同的测试数据格式
        )

        # 调用preprocess安装统计钩子
        iter_smooth_processor.preprocess(request)
        msmodelslim_logger.info("统计钩子安装完成")

        # 在preprocess之后再次禁用所有参数的梯度（preprocess会创建新的RMSNormBias模块）
        for param in model.parameters():
            param.requires_grad = False

        # 检查是否成功安装了钩子
        hook_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, '_forward_hooks') and module._forward_hooks:
                    hook_count += len(module._forward_hooks)
                    msmodelslim_logger.info(f"模块 {name} 已安装 {len(module._forward_hooks)} 个前向钩子")

        msmodelslim_logger.info(f"总共安装了 {hook_count} 个统计钩子")

        # 测试前向推理 - 触发统计钩子收集统计信息
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("开始前向推理 - 触发统计钩子收集统计信息")
        msmodelslim_logger.info("=" * 50)

        output_logits_before_anti = model(test_prompt[KEY_INPUT_IDS]).logits
        msmodelslim_logger.info(f"处理前输出形状: {output_logits_before_anti.shape}")
        msmodelslim_logger.info(f"处理前输出统计: 均值={output_logits_before_anti.mean().item():.6f}, "
                                f"标准差={output_logits_before_anti.std().item():.6f}")

        # 检查收集的统计信息
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("检查收集的统计信息")
        msmodelslim_logger.info("=" * 50)

        if hasattr(iter_smooth_processor, 'act_stats') and iter_smooth_processor.act_stats:
            msmodelslim_logger.info(f"成功收集了 {len(iter_smooth_processor.act_stats)} 个模块的统计信息")
            for module_name, stats in iter_smooth_processor.act_stats.items():
                msmodelslim_logger.info(f"模块 {module_name} 的统计信息:")
                for stat_key, stat_value in stats.items():
                    if isinstance(stat_value, torch.Tensor):
                        msmodelslim_logger.info(f"  {stat_key}: 形状={stat_value.shape}, 设备={stat_value.device}")
                    else:
                        msmodelslim_logger.info(f"  {stat_key}: {stat_value}")
        else:
            msmodelslim_logger.warning("未收集到统计信息")

        # 测试postprocess阶段 - 应用平滑处理
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("开始postprocess阶段 - 应用平滑处理")
        msmodelslim_logger.info("=" * 50)

        # 调用postprocess应用平滑处理
        iter_smooth_processor.postprocess(request)
        msmodelslim_logger.info("平滑处理完成")

        # 检查处理后的统计信息是否被清理
        if hasattr(iter_smooth_processor, 'act_stats'):
            if iter_smooth_processor.act_stats:
                msmodelslim_logger.info(f"平滑处理后剩余统计信息: {len(iter_smooth_processor.act_stats)} 个模块")
            else:
                msmodelslim_logger.info("平滑处理后统计信息已清理")

        # 获取处理后的模型输出并比较
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("比较 IterSmoothProcessor 处理前后的模型输出")
        msmodelslim_logger.info("=" * 50)

        output_logits_after_anti = model(test_prompt[KEY_INPUT_IDS]).logits
        msmodelslim_logger.info(f"处理后输出形状: {output_logits_after_anti.shape}")
        msmodelslim_logger.info(f"处理后输出统计: 均值={output_logits_after_anti.mean().item():.6f}, "
                                f"标准差={output_logits_after_anti.std().item():.6f}")

        # 比较处理前后的输出差异
        # 首先检查形状是否匹配
        if output_logits_before_anti.shape != output_logits_after_anti.shape:
            error_msg = (f"处理前后输出形状不匹配: "
                         f"处理前={output_logits_before_anti.shape}, "
                         f"处理后={output_logits_after_anti.shape}")
            msmodelslim_logger.error(error_msg)
            raise AssertionError(error_msg)

        # 计算绝对差异
        abs_diff = torch.abs(output_logits_before_anti - output_logits_after_anti)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        std_diff = abs_diff.std().item()

        msmodelslim_logger.info(f"输出差异统计:")
        msmodelslim_logger.info(f"  最大绝对差异: {max_diff:.15f}")
        msmodelslim_logger.info(f"  平均绝对差异: {mean_diff:.15f}")
        msmodelslim_logger.info(f"  绝对差异标准差: {std_diff:.15f}")

        # 检查是否在可接受范围内（对称模式应该保持数值一致性）
        tolerance = 1e-5
        if torch.allclose(output_logits_before_anti, output_logits_after_anti, atol=tolerance):
            msmodelslim_logger.info(f"✓ IterSmoothProcessor 处理前后输出差异在可接受范围内 (atol={tolerance})")
        else:
            msmodelslim_logger.error(f"✗ IterSmoothProcessor 处理前后输出差异超出可接受范围 (atol={tolerance})")

            # 计算相对差异
            relative_diff = abs_diff / (torch.abs(output_logits_before_anti) + 1e-8)
            max_rel_diff = relative_diff.max().item()
            mean_rel_diff = relative_diff.mean().item()

            msmodelslim_logger.info(f"相对差异统计:")
            msmodelslim_logger.info(f"  最大相对差异: {max_rel_diff:.6f}")
            msmodelslim_logger.info(f"  平均相对差异: {mean_rel_diff:.6f}")

            error_msg = (f"IterSmoothProcessor对称模式处理前后输出差异过大：\n"
                         f"  最大差异={max_diff:.6e}, 平均差异={mean_diff:.6e}, "
                         f"容差={tolerance}\n"
                         f"  注意：对称模式下应该保持数值一致性（纯线性变换）")
            msmodelslim_logger.error(error_msg)
            raise AssertionError(error_msg)

        # 检查模型权重是否发生变化
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("检查模型权重变化")
        msmodelslim_logger.info("=" * 50)

        return iter_smooth_processor

    except Exception as e:
        msmodelslim_logger.error(f"测试 IterSmoothProcessor 失败: {e}")
        import traceback
        traceback.print_exc()
        # 重新抛出异常，让pytest能够识别测试失败
        raise


if __name__ == "__main__":
    msmodelslim_logger.info("=" * 60)
    msmodelslim_logger.info("LLaMA2-7B 模型 IterSmoothProcessor 功能测试开始")
    msmodelslim_logger.info("=" * 60)

    try:
        # 测试1: 完整的IterSmoothProcessor功能测试
        msmodelslim_logger.info("\n1. 完整测试 IterSmoothProcessor 功能（包含统计钩子收集）...")
        iter_smooth_processor = test_iter_smooth_processor_with_hooks()

        if iter_smooth_processor:
            msmodelslim_logger.info("✓ IterSmoothProcessor 完整功能测试成功")
        else:
            msmodelslim_logger.error("✗ IterSmoothProcessor 完整功能测试失败")

        msmodelslim_logger.info("\n" + "=" * 60)
        msmodelslim_logger.info("所有测试完成")
        msmodelslim_logger.info("=" * 60)

    except Exception as e:
        msmodelslim_logger.error(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        raise e
