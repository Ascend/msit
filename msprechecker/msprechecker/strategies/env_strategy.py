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
import os

from .base_strategy import CollectStrategy
from ..utils import LOGGER


class Env(CollectStrategy):
    ENV_FILTERS = [
        "ASCEND",
        "MINDIE",
        "ATB_",
        "HCCL_",
        "MIES",
        "RANKTABLE",
        "GE_",
        "TORCH",
        "ACL_",
        "NPU_",
        "LCCL_",
        "LCAL_",
        "OPS",
        "INF_",
    ]

    def __init__(self, name: str = "env", ascend_only: bool = False):
        super().__init__(name)
        self._ascend_only = ascend_only
        self.env_path = None

    def execute(self):
        env_items = os.environ.items()

        if self._ascend_only:
            return {
                k: v
                for k, v in env_items
                if any(item in k for item in self.ENV_FILTERS)
            }
        return dict(env_items)

    def sync(self, target_data: dict) -> None:
        super().sync(target_data)
        env_file_name = f"sync-{target_data.get('image', {}).get('image_type', '')}-{target_data.get('timestamp', '')}.env"
        with open(env_file_name, "w") as f:
            for k, v in self._target.items():
                f.write(f"{k}={v}\n")
        LOGGER.info(f"Env file {env_file_name} generated, you may source it to set environment variables.")
        self.env_path = os.path.abspath(env_file_name)
