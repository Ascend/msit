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

import os
from pathlib import Path
from typing import List, Dict, Union, Optional

from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import get_valid_read_path


@logger_setter('msmodelslim.infra.vlm_dataset_loader')
class VLMDatasetLoader(DatasetLoaderInterface):
    """
    Dataset loader for multimodal vision-language models.
    
    Supports:
    - Image directory (automatic discovery)
    - Image list with custom prompts
    - JSONL format (image path + text)
    
    Like FileDatasetLoader, supports lab_calib short names:
        - "calibImages" -> searches in lab_calib/calibImages
        - "calib_data.jsonl" -> searches in lab_calib/calib_data.jsonl
    
    Examples:
        # Method 1: Image directory with short name (lab_calib)
        dataset = loader.get_dataset_by_name('calibImages')  # -> lab_calib/calibImages
        
        # Method 2: Image directory with absolute path
        dataset = loader.get_dataset_by_name('/path/to/images')
        
        # Method 3: JSONL file with short name
        dataset = loader.get_dataset_by_name('calib_data.jsonl')  # -> lab_calib/calib_data.jsonl
    """
    
    def __init__(self, dataset_dir: Optional[Path] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_dir: Optional directory to search for relative paths (like lab_calib).
                        If None, will try to auto-detect lab_calib directory.
        """
        super().__init__()
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        # Set dataset directory (for short names like "calibImages")
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir
        else:
            # Try to auto-detect lab_calib directory
            self.dataset_dir = self._get_default_dataset_dir()
    
    def get_dataset_by_name(self, dataset_name: str) -> Union[str, List[Dict[str, str]]]:
        """
        Load dataset by name or path.
        
        Args:
            dataset_name: Can be:
                - Short name (e.g., "calibImages") -> searches in lab_calib/
                - Path to image directory (absolute or relative)
                - Path to JSONL file (absolute or relative)
                - Dataset identifier
        
        Returns:
            Dataset in format suitable for model adapter
        """
        dataset_path = Path(dataset_name)
        
        # Strategy (similar to FileDatasetLoader):
        # 1. If absolute path, use it directly
        # 2. If relative path exists, use it
        # 3. Otherwise, try to combine with dataset_dir (lab_calib)
        
        if dataset_path.is_absolute():
            # Case 1: Absolute path
            resolved_path = dataset_path
            get_logger().info(f"Using absolute path: {resolved_path}")
        elif dataset_path.exists():
            # Case 2: Relative path that exists from current directory
            resolved_path = dataset_path.resolve()
            get_logger().info(f"Using existing relative path: {dataset_name} -> {resolved_path}")
        elif self.dataset_dir is not None:
            # Case 3: Try to combine with dataset_dir (lab_calib)
            resolved_path = self.dataset_dir / dataset_name
            if os.path.exists(resolved_path):
                get_logger().info(f"Resolved short name: {dataset_name} -> {resolved_path}")
            else:
                # Path doesn't exist even after combining with dataset_dir
                # Try to resolve the original relative path
                try:
                    resolved_path = dataset_path.resolve()
                    if resolved_path.exists():
                        get_logger().info(f"Resolved relative path: {dataset_name} -> {resolved_path}")
                    else:
                        # Path still doesn't exist, raise error
                        get_logger().error(
                            f"Dataset path not found: {dataset_name}"
                        )
                        raise InvalidDatasetError(
                            f"Dataset path does not exist: {dataset_name}",
                            action=(
                                f"Please check if the path '{dataset_name}' is correct "
                                f"or if it exists in {self.dataset_dir}"
                            )
                        )
                except InvalidDatasetError:
                    # Re-raise our own exception
                    raise
                except Exception as e:
                    get_logger().error(f"Failed to resolve path {dataset_name}: {e}")
                    raise InvalidDatasetError(
                        f"Failed to resolve dataset path: {dataset_name}",
                        action=f"Please check if the path is valid and accessible"
                    ) from e
        else:
            # Case 4: No dataset_dir, try to resolve as relative path
            try:
                resolved_path = dataset_path.resolve()
                if resolved_path.exists():
                    get_logger().info(f"Resolved relative path: {dataset_name} -> {resolved_path}")
                else:
                    get_logger().error(f"Dataset path not found: {dataset_name}")
                    raise InvalidDatasetError(
                        f"Dataset path does not exist: {dataset_name}",
                        action=f"Please check if the path '{dataset_name}' is correct or provide an absolute/valid path"
                    )
            except InvalidDatasetError:
                # Re-raise our own exception
                raise
            except Exception as e:
                get_logger().error(f"Failed to resolve path {dataset_name}: {e}")
                raise InvalidDatasetError(
                    f"Failed to resolve dataset path: {dataset_name}",
                    action=f"Please check if the path is valid and accessible"
                ) from e

        # 新增：兜底检查路径是否存在，避免走到类型判断分支才报错
        if not resolved_path.exists():
            get_logger().error(f"Dataset path not found: {resolved_path}")
            raise InvalidDatasetError(
                f"Dataset path does not exist: {resolved_path}",
                action=f"Please check if the path '{resolved_path}' is correct and exists"
            )

        # Now check what type of resource resolved_path is
        if resolved_path.is_dir():
            resolved_path = get_valid_read_path(str(resolved_path), is_dir=True, check_user_stat=True)
            get_logger().info(f"Loading images from directory: {resolved_path}")
            return self._load_images_from_directory(resolved_path)
        elif resolved_path.is_file() and resolved_path.suffix == '.jsonl':
            resolved_path = get_valid_read_path(str(resolved_path), is_dir=False, check_user_stat=True)
            get_logger().info(f"Loading dataset from JSONL: {resolved_path}")
            return self._load_from_jsonl(resolved_path)
        else:
            # Unknown type or unsupported file format
            get_logger().error(f"Unsupported dataset type: {resolved_path}")
            raise InvalidDatasetError(
                f"Dataset path exists but is not a valid type: {resolved_path}",
                action="Please provide either a directory containing images or a .jsonl file"
            )
    
    def _get_default_dataset_dir(self) -> Optional[Path]:
        """
        Try to auto-detect lab_calib directory.
        
        Returns:
            Path to lab_calib directory, or None if not found
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../lab_calib'))
        lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
        return Path(lab_calib_dir)

    def _load_images_from_directory(self, directory: Union[Path, str]) -> List[Dict[str, str]]:
        """
        Load all images from a directory.
        
        Args:
            directory: Path to image directory (Path object or string)
        
        Returns:
            List of dicts with 'image' and 'text' keys
        """
        # Convert to string for security validation, then back to Path for operations
        directory = Path(get_valid_read_path(str(directory), is_dir=True, check_user_stat=True))
        
        image_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No images found in directory: {directory}")
        
        get_logger().info(f"Found {len(image_files)} images in {directory}")
        
        return [
            {
                'image': str(img_path),
                'text': 'Describe this image in detail.'  # Default prompt
            }
            for img_path in sorted(image_files)
        ]
    
    def _load_from_jsonl(self, jsonl_path: Union[Path, str]) -> List[Dict[str, str]]:
        """
        Load dataset from JSONL file.
        
        JSONL format:
            {"image": "/path/to/img1.jpg", "text": "Describe this."}
            {"image": "/path/to/img2.jpg", "text": "What is this?"}
        
        Args:
            jsonl_path: Path to JSONL file (Path object or string)
        
        Returns:
            List of dicts with 'image' and 'text' keys
        """
        import json
        
        # Convert to string for security validation
        jsonl_path = get_valid_read_path(str(jsonl_path))
        
        dataset = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'image' not in item:
                        get_logger().warning(f"Line {line_num}: Missing 'image' field, skipping")
                        continue
                    
                    # Default text if not provided
                    if 'text' not in item:
                        item['text'] = 'Describe this image in detail.'
                    
                    # Validate image path
                    image_path = get_valid_read_path(item['image'])
                    item['image'] = str(image_path)
                    
                    dataset.append(item)
                    
                except json.JSONDecodeError as e:
                    get_logger().warning(f"Line {line_num}: Invalid JSON - {e}, skipping")
                    continue
                except Exception as e:
                    get_logger().warning(f"Line {line_num}: Error - {e}, skipping")
                    continue
        
        if not dataset:
            raise ValueError(f"No valid entries found in JSONL file: {jsonl_path}")
        
        get_logger().info(f"Loaded {len(dataset)} samples from JSONL")
        return dataset

