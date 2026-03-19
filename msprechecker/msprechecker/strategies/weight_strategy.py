# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

from .base_strategy import CollectStrategy
from ..utils import Utils, LOGGER, PathUtil


class Weight(CollectStrategy):
    """Collect strategy that computes SHA-256 hashes of safetensor weight files."""

    # Matches the shard index in filenames like ``model-00003-of-00010.safetensors``
    _TENSOR_ID_RE = re.compile(r"(\d{5})-of-\d{5}")

    def __init__(
            self,
            name: str = "weight",
            *,
            weight_dir: str = "",
            tensor_suffix: str = ".safetensors",
            max_size: int = 10 * 1024 ** 3,  # 10 GiB – skip files larger than this
            chunk_size: int = 256 * 1024 ** 2,  # 256 MiB read buffer
            max_hash_workers: int = 4,
    ):
        super().__init__(name)
        self._weight_dir = weight_dir
        self._tensor_suffix = tensor_suffix
        self._max_size = max_size
        self._chunk_size = chunk_size
        self._max_hash_workers = max_hash_workers

    def _validate_weight_dir(self) -> bool:
        LOGGER.info("Validating weight directory: {}".format(self._weight_dir))
        if not os.path.isdir(self._weight_dir):
            Utils.log_error_and_exit(
                "Expected {} to be a directory. Weight strategy failed".format(self._weight_dir),
            )
        return True

    @staticmethod
    def _merge_hashes(hash_map):
        sorted_hashes = sorted(hash_map.values())
        combined = "".join(sorted_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _calculate_hash256(self, tensor_file):
        sha256_hash = hashlib.sha256()
        with open(tensor_file, "rb") as f:
            while True:
                data = f.read(self._chunk_size)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()

    def _is_valid_tensor_file(self, path: str) -> bool:
        if os.path.islink(path):
            Utils.log_error_and_exit(
                "Expected {} to be a regular file. Weight strategy failed".format(path)
            )
            return False

        if not os.path.isfile(path) or not path.endswith(self._tensor_suffix):
            return False

        file_size = os.path.getsize(path)
        if file_size > self._max_size:
            LOGGER.warning(
                "Tensor file %r (%d bytes) exceeds max_size (%d bytes), skipping",
                path,
                file_size,
                self._max_size,
            )
            return False
        return True

    def _filter_valid_tensor_files(self) -> List[str]:
        result = []
        for filename in os.listdir(self._weight_dir):
            full_path = os.path.join(self._weight_dir, filename)
            if self._is_valid_tensor_file(full_path):
                result.append(full_path)
        return result

    def _get_tensor_id(self, tensor_file: str) -> str:
        """Return the zero-padded shard index, or the basename as fallback."""
        m = self._TENSOR_ID_RE.search(os.path.basename(tensor_file))
        return m.group(1) if m else os.path.basename(tensor_file)

    def _calculate_hash256(self, tensor_file: str) -> str:
        sha256 = hashlib.sha256()
        with open(tensor_file, "rb") as f:
            while True:
                chunk = f.read(self._chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parallel_hash_calculation(
            self, tensor_files: List[str]
    ) -> Dict[str, Optional[str]]:
        max_workers = min(len(tensor_files), self._max_hash_workers)
        results: Dict[str, Optional[str]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self._calculate_hash256, f): self._get_tensor_id(f)
                for f in tensor_files
            }
            for future in as_completed(future_to_id):
                tensor_id = future_to_id[future]
                try:
                    results[tensor_id] = future.result()
                except Exception:
                    Utils.log_error_and_exit(
                        "Failed to calculate hash for tensor id {}", tensor_id
                    )
                    results[tensor_id] = None

        return results

    def _get_weights_from_resource(self) -> dict:
        resources_dir_path = PathUtil.get_resources_root_dir_path()
        resources_path = os.path.join(resources_dir_path, "weights.json")
        if not os.path.exists(resources_path):
            Utils.log_error_and_exit(f"Resources file {resources_path} not exist.")
        return Utils.load_json(resources_path)

    def _check_weight(self):
        target_hash = self._merge_hashes(self._target)
        if self._weight_dir:
            current = self.execute()
            current_hash = self._merge_hashes(current)
            if current_hash != target_hash:
                LOGGER.warning(f"Weight hash {current_hash} not match weight hash {target_hash} from dumped "
                               f"file, please check your weight dir.")
            else:
                LOGGER.info(f"Weight check passed.")
                return

        weight_data = self._get_weights_from_resource()
        LOGGER.debug("Target hash: {}".format(target_hash))
        if target_hash not in weight_data:
            Utils.log_error_and_exit("Weights from dumped file not found in resource, please update it.")
        weight_map = weight_data.get(target_hash)

        name = weight_map.get("name", "")
        url = weight_map.get("modelscope", "")
        Utils.log_error_and_exit(f"Please download {name} from {url} "
                                 f"After downloading, please rerun the command using the [-w/--weight-dir] option.")

    def get_weight_dir(self):
        return self._weight_dir

    def execute(self) -> Optional[Dict[str, Optional[str]]]:
        if not self._validate_weight_dir():
            return None

        tensor_files = self._filter_valid_tensor_files()
        if not tensor_files:
            LOGGER.warning("No valid tensor files found in the specified directory")
            return None

        return self._parallel_hash_calculation(tensor_files)

    def sync(self, target_data: dict) -> Any:
        super().sync(target_data)
        self._check_weight()
