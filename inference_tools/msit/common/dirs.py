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

import os

from msit.utils.path import make_dir, MsitPath
from msit.common.exceptions import MsitException
from msit.common.dependencies import get_rank_for_pytorch
from msit.common.constants import DumpConst, PathConst, MsgConst


class DirPool:
    dump_path = input_dir = model_dir = tensor_dir = rank_dir = None

    @classmethod
    def make_dump_dir(cls, path):
        cls.dump_path = os.path.join(path, DumpConst.CURRENT_DUMP_FOLDER)
        make_dir(cls.dump_path)

    @classmethod
    def get_dump_path(cls):
        return cls.dump_path

    @classmethod
    def make_input_dir(cls):
        if cls.dump_path:
            cls.input_dir = os.path.join(cls.dump_path, PathConst.SUBDIR_INPUT)
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, "Dump dir has not been set.")
        make_dir(cls.input_dir)

    @classmethod
    def get_input_dir(cls):
        if not cls.input_dir:
            cls.make_input_dir()
        return cls.input_dir

    @classmethod
    def make_new_model_dir(cls):
        if cls.dump_path:
            cls.model_dir = os.path.join(cls.dump_path, PathConst.SUBDIR_MODEL)
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, "Dump dir has not been set.")
        make_dir(cls.model_dir)

    @classmethod
    def get_new_model_path(cls, model_path):
        save_name = DumpConst.NEW_ + os.path.basename(model_path)
        if not cls.model_dir:
            cls.make_new_model_dir()
        new_model_path = os.path.join(cls.model_dir, save_name)
        return new_model_path

    @classmethod
    def make_tensor_dir(cls):
        if cls.dump_path:
            cls.tensor_dir = os.path.join(cls.dump_path, PathConst.SUBDIR_TENSOR)
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, "Dump dir has not been set.")
        make_dir(cls.tensor_dir)

    @classmethod
    def get_tensor_dir(cls):
        if not cls.tensor_dir:
            cls.make_tensor_dir()
        return cls.tensor_dir

    @classmethod
    def make_rank_dir(cls):
        current_rank = get_rank_for_pytorch()
        if not cls.tensor_dir:
            cls.make_tensor_dir()
        cls.rank_dir = os.path.join(cls.tensor_dir, f"{PathConst.SUBDIR_RANK}{current_rank}")
        make_dir(cls.rank_dir)

    @classmethod
    def get_rank_dir(cls):
        if not cls.rank_dir:
            cls.make_rank_dir()
        return cls.rank_dir
