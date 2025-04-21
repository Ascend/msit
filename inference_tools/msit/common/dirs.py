# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime

from msit.utils.log import get_current_timestamp, logger
from msit.utils.path import DirSafeHandler, get_basename_from_path, join_path
from msit.utils.toolkits import get_rank, timestamp_sync


class DirPool:
    msit_path = model_dir = step_dir = rank_dir = input_dir = tensor_dir = None

    @classmethod
    def make_msit_dir(cls, path: str):
        timestamp = get_current_timestamp(used_for_log=False)
        timestamp = timestamp_sync(timestamp)
        formatted_date = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        cls.msit_path = DirSafeHandler.join_and_create(path, f"msit_{formatted_date}/")

    @classmethod
    def get_msit_dir(cls):
        return DirSafeHandler.get_or_raise(cls.msit_path, "Dump dir has not been set.")

    @classmethod
    def make_model_dir(cls):
        cls.model_dir = DirSafeHandler.join_and_create(cls.get_msit_dir(), "model")

    @classmethod
    def get_model_dir(cls):
        return DirSafeHandler.get_or_raise(cls.model_dir, "Model dir has not been set.")

    @classmethod
    def get_uninfer_model_path(cls, model_path: str):
        save_name = "inferential_" + get_basename_from_path(model_path)
        return join_path(cls.get_model_dir(), save_name)

    @classmethod
    def make_step_dir(cls, current_iter: int):
        cls.step_dir = DirSafeHandler.join_and_create(cls.get_msit_dir(), f"step{current_iter}")

    @classmethod
    def get_step_dir(cls):
        logger.info(f"Step dir has switched to {cls.step_dir}.")
        return DirSafeHandler.get_or_raise(cls.step_dir, "Step dir has not been set.")

    @classmethod
    def make_rank_dir(cls):
        cls.rank_dir = DirSafeHandler.join_and_create(cls.get_step_dir(), f"rank{get_rank()}")

    @classmethod
    def get_rank_dir(cls):
        return DirSafeHandler.get_or_raise(cls.rank_dir, "Rank dir has not been set.")

    @classmethod
    def make_input_dir(cls):
        cls.input_dir = DirSafeHandler.join_and_create(cls.get_rank_dir(), "input")

    @classmethod
    def get_input_dir(cls):
        return DirSafeHandler.get_or_raise(cls.input_dir, "Input dir has not been set.")

    @classmethod
    def make_tensor_dir(cls):
        cls.tensor_dir = DirSafeHandler.join_and_create(cls.get_rank_dir(), "dump_tensor_data")

    @classmethod
    def get_tensor_dir(cls):
        return DirSafeHandler.get_or_raise(cls.tensor_dir, "dump_tensor_data dir has not been set.")
