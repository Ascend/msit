#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import abc
from typing import Optional, List, Any, Generator

import torch.nn
from torch.utils.data import DataLoader, Subset

from msmodelslim import logger
from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest, ProcessRequest


class ProcessUnit:

    def __init__(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None):
        self.processor = processor
        self.input_datas = input_datas
        self.generators = self.build_generators()
        self.batch_outputs = [None for _ in self.generators]

    @staticmethod
    def _create_dataloader(dataset, rank, world_size, batch_size):
        total_size = len(dataset)
        per_rank_size = total_size // world_size
        start = rank * per_rank_size
        end = start + per_rank_size if rank != world_size - 1 else total_size
        subset = Subset(dataset, range(start, end))
        return DataLoader(subset, batch_size=batch_size, shuffle=False)

    @abc.abstractmethod
    def build_generators(self) -> List[Generator[ProcessRequest, Any, None]]:
        pass

    def make_progress(self) -> bool:
        requests = []

        for gen, output in zip(self.generators, self.batch_outputs):
            try:
                request = gen.send(output)
                requests.append(request)
            except StopIteration:
                return False

        batch_request = BatchProcessRequest(requests[0].name,
                                            requests[0].module,
                                            [(request.args, request.kwargs,) for request in requests],
                                            [None for _ in requests])
        self.processor.process(batch_request)
        self.batch_outputs = batch_request.outputs
        return True


@torch.no_grad()
def generated_schedule(process_unit: List[ProcessUnit]):
    """
    使用生成式前向函数运行模型。
    
    该函数从处理单元列表中提取输入数据，每个处理单元包含一个处理器和可选的输入数据。
    函数会交错调度各个处理单元，即先调度unit1的第一步，然后调度unit2的第一步，以此类推。
    当某个unit完成时，会将其从调度队列中删除。
    
    参数:
        process_unit: 处理单元列表，每个元素包含一个处理器和可选的输入数据
    """

    logger.info(f"Scheduler {len(process_unit)} unit")

    unit_list = [unit for unit in process_unit]

    _ = [unit.processor.pre_run() for unit in process_unit]

    while unit_list:
        for unit in unit_list:
            if not unit.make_progress():
                unit_list.remove(unit)

    _ = [unit.processor.post_run() for unit in process_unit]
