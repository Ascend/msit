# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import os.path
import shutil
from dataclasses import dataclass, field
from logging import Logger
from typing import Optional, Union

import torch
import torch.distributed as dist

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import SAVE_TYPE_SAFE_TENSOR
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer import BufferedSafetensorsWriter, JsonDescriptionWriter, \
    SafetensorsWriter


@dataclass
class SafetensorsSaverConfig:
    logger: Logger = field(init=False, default=msmodelslim_logger)

    output_dir: str
    model_quant_type: str
    use_kvcache_quant: bool = False
    use_fa_quant: bool = False

    safetensors_name: Optional[str] = 'quant_model_weight.safetensors'
    json_name: Optional[str] = 'quant_model_description.json'
    part_file_size: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.safetensors_name, str):
            default_safetensors_name = f"quant_model_weight_{self.model_quant_type.lower()}.safetensors"
            self.logger.warning(f"invalid `safetensors_name`, defaulting to `{default_safetensors_name}`")
            self.safetensors_name = default_safetensors_name
        if not isinstance(self.json_name, str):
            default_json_name = f"quant_model_description_{self.model_quant_type.lower()}.json"
            self.logger.warning(f"invalid `json_name`, defaulting to `{default_json_name}`")
            self.json_name = default_json_name

    @staticmethod
    def from_dict(d: dict):
        if isinstance(d, SafetensorsSaverConfig):
            return d
        if not isinstance(d, dict):
            raise TypeError(f'Safetensors save config must be an instance of dict, but got {type(d).__name__}')
        return SafetensorsSaverConfig(**d)

    def get_saver(self):
        if dist.is_initialized() and dist.get_world_size() > 1:
            return DistributedSafetensorsSaver(self)
        return SafetensorsSaver(self)


class SafetensorsSaver(BaseSaver):
    type_ = SAVE_TYPE_SAFE_TENSOR

    def __init__(self, cfg: Union[SafetensorsSaverConfig, dict]):
        super().__init__()

        cfg = SafetensorsSaverConfig.from_dict(cfg)
        self.logger = cfg.logger

        if cfg.part_file_size is None:
            file_path = os.path.join(cfg.output_dir, cfg.safetensors_name)
            self.weight_writer = SafetensorsWriter(logger=self.logger, file_path=file_path)
        else:
            file_name_prefix = cfg.safetensors_name.replace('.safetensors', '')
            self.weight_writer = BufferedSafetensorsWriter(logger=self.logger,
                                                           max_gb_size=cfg.part_file_size,
                                                           save_directory=cfg.output_dir,
                                                           save_prefix=file_name_prefix)
        self.meta_writer = JsonDescriptionWriter(logger=self.logger,
                                                 model_quant_type=cfg.model_quant_type,
                                                 json_name=cfg.json_name,
                                                 save_directory=cfg.output_dir,
                                                 use_kvcache_quant=cfg.use_kvcache_quant,
                                                 use_fa_quant=cfg.use_fa_quant)

    def pre_process(self) -> None:
        pass

    def save(self, name, meta, data) -> None:
        self.weight_writer.write(name, data)
        self.meta_writer.write(name, meta)

    def post_process(self) -> None:
        self.weight_writer.close()
        self.meta_writer.close()

        self.logger.info(f'Safetensors weight saved successfully')


class DistributedSafetensorsSaver(SafetensorsSaver):
    def __init__(self, cfg: Union[SafetensorsSaverConfig, dict]):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.cfg = SafetensorsSaverConfig.from_dict(cfg)
        # 保存原始输出目录
        self.original_output_dir = self.cfg.output_dir
        # 设置rank特定的输出目录
        self.rank_output_dir = os.path.join(self.original_output_dir, f"rank_{self.rank}")
        self.cfg.output_dir = self.rank_output_dir
        # 初始化每个rank的文件映射列表
        self.file_mappings = [{} for _ in range(self.world_size)]

        # 调用父类初始化
        super().__init__(cfg)

    def pre_process(self) -> None:
        pass

    def save(self, name, meta, data) -> None:
        self.weight_writer.write(name, data)
        self.meta_writer.write(name, meta)

    def post_process(self) -> None:
        self.weight_writer.close()
        self.meta_writer.close()
        self.logger.info(f'Safetensors weight saved successfully for rank {self.rank}')

        # 等待所有rank完成保存
        os.sync()
        dist.barrier()
        self._merge_rank_files()
        dist.barrier()

    def _merge_rank_files(self):
        """合并所有rank保存的文件"""
        # 统计所有rank的文件数量
        file_counts = [torch.zeros(1, dtype=torch.int64)] * self.world_size
        local_count = len([f for f in os.listdir(self.rank_output_dir) if f.endswith('.safetensors')])
        msmodelslim_logger.debug(f"Rank {dist.get_rank()} has {local_count} files")
        dist.all_gather_object(file_counts, local_count)

        if self.rank != 0:
            return

        # 合并各类文件
        self._merge_safetensors_files(file_counts)
        self._merge_index_files()
        self._merge_json_files()
        self._cleanup_rank_dirs()

    def _merge_safetensors_files(self, file_counts):
        """合并safetensors文件"""
        msmodelslim_logger.debug(f"Merge {sum(file_counts)} safetensors files")
        for rank in range(self.world_size):
            rank_dir = os.path.join(self.original_output_dir, f"rank_{rank}")
            safetensors_files = sorted([f for f in os.listdir(rank_dir) if f.endswith('.safetensors')])

            # 计算当前rank的文件偏移
            offset = sum(file_counts[:rank])

            # 重命名并移动文件
            for i, file in enumerate(safetensors_files):
                src = os.path.join(rank_dir, file)
                # 保持原有的文件命名格式
                if self.cfg.part_file_size is None:
                    dst = os.path.join(self.original_output_dir, self.cfg.safetensors_name)
                else:
                    file_name_prefix = self.cfg.safetensors_name.replace('.safetensors', '')
                    dst = os.path.join(self.original_output_dir,
                                       f"{file_name_prefix}-{offset + i + 1:05d}-of-{sum(file_counts):05d}.safetensors")
                # 记录文件映射关系
                self.file_mappings[rank][file] = os.path.basename(dst)
                msmodelslim_logger.debug(f"{src} -> {dst}")
                shutil.move(src, dst)

    def _merge_index_files(self):
        """合并index文件"""
        index_files = []
        for rank in range(self.world_size):
            rank_dir = os.path.join(self.original_output_dir, f"rank_{rank}")
            index_file = os.path.join(rank_dir, f"{self.cfg.safetensors_name}.index.json")
            if os.path.exists(index_file):
                index_files.append((rank, index_file))

        if index_files:
            # 合并index文件
            merged_index = {"metadata": {"total_size": 0}, "weight_map": {}}
            for rank, index_file in index_files:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    merged_index["metadata"]["total_size"] += index_data["metadata"]["total_size"]
                    # 更新weight_map中的文件路径
                    for key, value in index_data["weight_map"].items():
                        # 从原始路径中提取文件名
                        original_file = os.path.basename(value)
                        # 使用对应rank的文件映射查找新文件名
                        new_file = self.file_mappings[rank].get(original_file)
                        if new_file is None:
                            self.logger.warning(f"找不到rank {rank}中文件 {original_file} 的映射关系")
                            continue
                        merged_index["weight_map"][key] = new_file

            # 保存合并后的index文件
            final_index_path = os.path.join(self.original_output_dir, f"{self.cfg.safetensors_name}.index.json")
            msmodelslim_logger.info(f'Save merged index file to {final_index_path}')
            with open(final_index_path, 'w') as f:
                json.dump(merged_index, f, indent=2)

    def _merge_json_files(self):
        """合并json描述文件"""
        json_files = []
        for rank in range(self.world_size):
            rank_dir = os.path.join(self.original_output_dir, f"rank_{rank}")
            json_file = os.path.join(rank_dir, self.cfg.json_name)
            if os.path.exists(json_file):
                json_files.append(json_file)

        if json_files:
            # 合并json文件
            merged_meta = {}
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    merged_meta.update(meta)

            # 对键进行排序
            sorted_meta = dict(sorted(merged_meta.items()))

            # 保存合并后的json文件
            final_json_path = os.path.join(self.original_output_dir, self.cfg.json_name)
            with open(final_json_path, 'w') as f:
                json.dump(sorted_meta, f, indent=2)

    def _cleanup_rank_dirs(self):
        """清理rank目录"""
        for rank in range(self.world_size):
            rank_dir = os.path.join(self.original_output_dir, f"rank_{rank}")
            msmodelslim_logger.info(f'Rmtree {rank_dir}')
            shutil.rmtree(rank_dir)
