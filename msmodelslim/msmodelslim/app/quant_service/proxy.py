#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from pathlib import Path
from typing import Optional, Any

from msmodelslim.app import DeviceType
from msmodelslim.app.base import BaseQuantConfig
from msmodelslim.app.quant_service import BaseQuantService, DatasetLoaderInterface, load_plugins, load_quant_service_cls
from msmodelslim.infra import VLMDatasetLoader
from msmodelslim.utils.logging import logger_setter, get_logger


@logger_setter(prefix='msmodelslim.app.quant_service.proxy')
class QuantServiceProxy(BaseQuantService):

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)
        self.quant_service: Optional[BaseQuantService] = None

    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: Any,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
    ) -> None:
        load_plugins()
        
        # Determine the appropriate dataset loader based on apiversion
        self.dataset_loader = self._get_dataset_loader_for_service(quant_config.apiversion)
        
        self.quant_service = load_quant_service_cls(quant_config.apiversion)(self.dataset_loader)
        self.quant_service.quantize(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=device,
        )
    
    def _get_dataset_loader_for_service(self, apiversion: str) -> DatasetLoaderInterface:
        """
        Get the appropriate dataset loader for a specific service.
        
        For services that require specialized dataset loaders, create and return
        the appropriate loader instance. For other services, return the default
        FileDatasetLoader.
        
        Args:
            apiversion: The API version string (e.g., "multimodal_vlm_modelslim_v1")
        
        Returns:
            Dataset loader instance
        """
        # Map services to their specialized dataset loaders
        if apiversion == 'multimodal_vlm_modelslim_v1':
            get_logger().info(f"Creating VLMDatasetLoader for service '{apiversion}'")
            return VLMDatasetLoader()
        
        # Other services use the default FileDatasetLoader
        return self.dataset_loader
