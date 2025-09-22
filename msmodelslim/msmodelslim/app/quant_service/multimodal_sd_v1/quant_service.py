#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools
import os
from pathlib import Path
from typing import Optional, Any

from msmodelslim.app import DeviceType
from msmodelslim.app.base.const import RunnerType
from msmodelslim.app.quant_service.base import BaseQuantConfig, BaseQuantService
from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.utils.cache import load_cached_data, to_device
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from .pipeline_interface import MultimodalPipelineInterface
from .quant_config import MultimodalSDModelslimV1QuantConfig


@logger_setter(prefix='msmodelslim.app.quant_service.multimodal_sd_v1')
class MultimodalSDModelslimV1QuantService(BaseQuantService):
    backend_name: str = "multimodal_sd_modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)

    def quantize(self, quant_config: BaseQuantConfig, model_adapter: Any, save_path: Optional[Path] = None,
                 device: DeviceType = DeviceType.NPU):
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask")
        if not isinstance(model_adapter, MultimodalPipelineInterface):
            raise SchemaValidateError("model must be a MultimodalPipelineInterface")
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError("device must be a DeviceType")

        return self.quant_process(MultimodalSDModelslimV1QuantConfig.from_base(quant_config), model_adapter, save_path,
                                  device)

    def quant_process(self, quant_config: MultimodalSDModelslimV1QuantConfig,
                      model_adapter: MultimodalPipelineInterface,
                      save_path: Optional[Path], device: DeviceType = DeviceType.NPU):

        # 覆盖配置
        model_adapter.set_model_args(quant_config.spec.multimodal_sd_config.model_extra['model_config'])
        # 加载模型
        model_adapter.load_pipeline()

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")

        config_dump_data_dir = quant_config.spec.multimodal_sd_config.dump_config.dump_data_dir
        if config_dump_data_dir:
            pth_file_path = os.path.join(config_dump_data_dir, "calib_data.pth")
        else:
            # 默认在保存路径下
            pth_file_path = os.path.join(save_path, "calib_data.pth")
        calib_data = load_cached_data(
            pth_file_path=pth_file_path,
            generate_func=model_adapter.run_calib_inference,
            model=model_adapter.init_model(device),
            dump_config=quant_config.spec.multimodal_sd_config.dump_config
        )

        calib_data = to_device(calib_data, device.value)

        get_logger().info(f"prepare calib_data from {pth_file_path} success")

        final_process_cfg = quant_config.spec.process

        if save_path is not None:
            get_logger().info(f"==========QUANTIZATION: Prepare Save Path==========")
            for save_cfg in quant_config.spec.save:
                save_cfg.set_save_directory(save_path)
            final_process_cfg += quant_config.spec.save
            get_logger().info(f"prepare Persistence to {save_path} success")

        get_logger().info(f"==========QUANTIZATION: Run Quantization==========")

        if quant_config.spec.runner != "layer_wise":
            get_logger().warning(f"runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")
        
        runner = LayerWiseRunner(adapter=model_adapter)

        for process_cfg in final_process_cfg:
            runner.add_processor(processor_cfg=process_cfg)

        # 多模态每个模型都不同处理，在模型内实现
        model_adapter.apply_quantization(functools.partial(runner.run, calib_data=calib_data, device=device))
        get_logger().info(f"quantization with runner {RunnerType.LAYER_WISE} success")
        get_logger().info(f"==========QUANTIZATION: END==========")
