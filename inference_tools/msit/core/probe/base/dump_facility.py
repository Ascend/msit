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

import numpy as np

from msit.common.log import logger
from msit.common.dirs import DirPool
from msit.common.exceptions import MsitException
from msit.common.constants import DumpConst, PathConst, MsgConst
from msit.utils.io import save_npy, load_bin_to_ndarray, load_npy


class DataDumper:
    def __init__(self, args):
        self.args = args
        DirPool.make_dump_dir(args.dump_path)
        self.input_map = {}

    @staticmethod
    def _is_dynamic_shape(tensor_shape):
        for shape in tensor_shape:
            if shape is None or not isinstance(shape, int):
                return True
        return False

    @staticmethod
    def _check_input_shape(op_name, model_shape, input_shape):
        if len(model_shape) != len(input_shape):
            raise MsitException(MsgConst.INVALID_ARGU, "Unequal lengths for the shape of model and input. "
                                f"Model shape: {model_shape}, input shape: {input_shape}.")
        for index, value in enumerate(model_shape):
            if value is None or isinstance(value, str):
                continue
            if input_shape[index] != value:
                raise MsitException(MsgConst.INVALID_ARGU, \
                                    "The input shape does not match the model shape. "
                                    f"Tensor name: {op_name}, {str(input_shape)} v.s. {str(model_shape)}.")

    @staticmethod
    def _tensor2numpy_for_type(tensor_type):
        numpy_data_type = DumpConst.TYPE2DTYPE_MAP.get(tensor_type)
        if numpy_data_type:
            return numpy_data_type
        else:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f"Tensor type {tensor_type} not provided.")

    @staticmethod
    def _generate_random_input_data(save_dir, names, shapes, dtypes):
        input_map = {}
        for index, (tensor_name, tensor_shape, tensor_dtype) in enumerate(zip(names, shapes, dtypes)):
            input_data = np.random.random(tensor_shape).astype(tensor_dtype)
            input_map[tensor_name] = input_data
            shape_str = "_".join(list(map(str, tensor_shape)))
            file_name = "_".join([PathConst.INDEX + str(index), DumpConst.SHAPE, shape_str, PathConst.SUFFIX_NPY])
            save_npy(input_data, os.path.join(save_dir, file_name))
            logger.info(
                f"Save input file path: {os.path.join(save_dir, file_name)}, "
                f"shape: {input_data.shape}, dtype: {input_data.dtype}."
            )
        return input_map

    @staticmethod
    def _read_input_data(input_paths, names, shapes, dtypes):
        input_map = {}
        for input_path, name, shape, dtype in zip(input_paths, names, shapes, dtypes):
            if input_path.endswith(PathConst.SUFFIX_BIN):
                input_data = load_bin_to_ndarray(input_path, dtype, shape)
            elif input_path.endswith(PathConst.SUFFIX_NPY):
                input_data = load_npy(input_path)
            if np.prod(input_data.shape) != np.prod(shape):
                raise MsitException(MsgConst.INVALID_ARGU, \
                                    "The shape of the input data does not match the model's shape, "
                                    f"input path: {input_path}, input shape: {input_data.shape}, "
                                    f"model's shape: {shape}.")
            input_data = input_data.reshape(shape)
            input_map[name] = input_data
            logger.info(
                f"Load input file path: {input_path}, shape: {input_data.shape}, dtype: {input_data.dtype}."
            )
        return input_map

    @classmethod
    def _get_input_shape_info(cls, tensor_name, tensor_shape, input_shape, tensor_type):
        cls._check_input_shape(tensor_name, tensor_shape, input_shape)
        tensor_shape_info = {DumpConst.NAME: tensor_name, DumpConst.SHAPE: input_shape, DumpConst.TYPE: tensor_type}
        logger.info(f"The dynamic shape of {tensor_name} has been fixed to {input_shape}.")
        return tensor_shape_info

    def get_inputs_data(self, inputs_tensor_info):
        names, shapes, dtypes = [], [], []
        for x in inputs_tensor_info:
            names.append(x[DumpConst.NAME])
            shapes.append(x[DumpConst.SHAPE])
            dtypes.append(self._tensor2numpy_for_type(x[DumpConst.TYPE]))
        if not self.args.input_path:
            input_dir = DirPool.get_input_dir()
            self.input_map = self._generate_random_input_data(input_dir, names, shapes, dtypes)
        else:
            self.input_map = self._read_input_data(self.args.input_path, names, shapes, dtypes)
        return self.input_map

    def _process_tensor_shape(self, tensor_name, tensor_type, tensor_shape):
        tensor_shape_info_list = []
        if self._is_dynamic_shape(tensor_shape):
            if not self.args.input_shape and not self.args.dym_shape_range:
                raise MsitException(MsgConst.INVALID_ARGU,
                                    f"The dynamic shape {tensor_shape} are not supported. Please "
                                    "set `-inshape`, `--input-shape` or `-dshape`, `--dym-shape-range` "
                                    "to fix the dynamic shape.")
            if tensor_name not in self.args.input_shape and tensor_name not in self.args.dym_shape_range:
                raise MsitException(MsgConst.INVALID_ARGU,
                                    "`-inshape`, `--input-shape` or `-dshape`, `--dym-shape-range` does not "
                                    f"provide the shape of {tensor_name}.")
        if self.args.dym_shape_range:
            input_shapes = self.args.dym_shape_range.get(tensor_name)
            for input_shape in input_shapes:
                tensor_shape_info = self._get_input_shape_info(tensor_name, tensor_shape, input_shape, tensor_type)
                tensor_shape_info_list.append(tensor_shape_info)
        elif self.args.input_shape:
            input_shape = self.args.input_shape.get(tensor_name)
            tensor_shape_info = self._get_input_shape_info(tensor_name, tensor_shape, input_shape, tensor_type)
            tensor_shape_info_list.append(tensor_shape_info)
        else:
            tensor_shape_info = {
                DumpConst.NAME: tensor_name, DumpConst.SHAPE: tensor_shape, DumpConst.TYPE: tensor_type
            }
            tensor_shape_info_list.append(tensor_shape_info)
        return tensor_shape_info_list
