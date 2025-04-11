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

import numpy as np

from msit.common.dirs import DirPool
from msit.utils.constants import MsgConst, PathConst
from msit.utils.dependencies import dependent
from msit.utils.exceptions import MsitException
from msit.utils.io import load_bin_to_ndarray, load_npy, save_npy
from msit.utils.log import logger
from msit.utils.path import join_path


def get_tf_type2dtype_map():
    pons = dependent.get_tensorflow()
    if None not in pons:
        tf, _, _ = pons
        return {
            tf.float16: np.float16,
            tf.float32: np.float32,
            tf.float64: np.float64,
            tf.int8: np.int8,
            tf.int16: np.int16,
            tf.int32: np.int32,
            tf.int64: np.int64,
        }
    else:
        return {}


class OfflineModelActuator:
    def __init__(self, model_path: str, input_shape: dict, input_path: str, **kwargs):
        self.model_path = model_path
        self.input_shape = input_shape or {}
        self.input_path = input_path or ""
        self.kwargs = kwargs

    @staticmethod
    def _is_dynamic_shape(tensor_shape):
        for shape in tensor_shape:
            if shape is None or not isinstance(shape, int):
                return True
        return False

    @staticmethod
    def _tensor2numpy_for_type(tensor_type):
        base_type2dtype_map = {
            "tensor(int)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.double,
            "tensor(bool)": np.bool_,
            "tensor(complex64)": np.complex64,
            "tensor(complex128)": np.complex_,
            "float32": np.float32,
            "float16": np.float16,
        }
        numpy_data_type = {**base_type2dtype_map, **get_tf_type2dtype_map()}.get(tensor_type)
        if numpy_data_type:
            return numpy_data_type
        else:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f"Tensor type {tensor_type} not provided.")

    @staticmethod
    def _generate_random_input_data(save_dir, names, shapes, dtypes):
        input_map = {}
        for tensor_name, tensor_shape, tensor_dtype in zip(names, shapes, dtypes):
            input_data = np.random.random(tensor_shape).astype(tensor_dtype)
            input_map[tensor_name] = input_data
            shape_str = "_".join(list(map(str, tensor_shape)))
            file_name = "_".join([tensor_name, "shape", shape_str, PathConst.SUFFIX_NPY])
            save_npy(input_data, join_path(save_dir, file_name))
            logger.info(
                f"Save input file path: {join_path(save_dir, file_name)}, "
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
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    "The shape of the input data does not match the model's shape, "
                    f"input path: {input_path}, input shape: {input_data.shape}, "
                    f"model's shape: {shape}.",
                )
            input_data = input_data.reshape(shape)
            input_map[name] = input_data
            logger.info(f"Load input file path: {input_path}, shape: {input_data.shape}, dtype: {input_data.dtype}.")
        return input_map

    @staticmethod
    def _check_input_shape(op_name, model_shape, input_shape):
        if not input_shape:
            raise MsitException(
                MsgConst.REQUIRED_ARGU_MISSING,
                f"{op_name}'s input_shape is missing. "
                f'Please set `shape: [xxx]` in "input" according to {model_shape}.',
            )
        if len(model_shape) != len(input_shape):
            raise MsitException(
                MsgConst.INVALID_ARGU,
                f"Unequal lengths for the shape of {op_name}. "
                f"Model shape: {model_shape}, input shape: {input_shape}.",
            )
        for index, value in enumerate(model_shape):
            if value is None or isinstance(value, str):
                continue
            if input_shape[index] != value:
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    "The input shape does not match the model shape. "
                    f"Tensor name: {op_name}, {str(input_shape)} v.s. {str(model_shape)}.",
                )

    @classmethod
    def _get_input_shape_info(cls, tensor_name, tensor_shape, input_shape, tensor_type):
        cls._check_input_shape(tensor_name, tensor_shape, input_shape)
        tensor_shape_info = {"name": tensor_name, "shape": input_shape, "type": tensor_type}
        logger.info(f"The dynamic shape of {tensor_name} has been fixed to {input_shape}.")
        return tensor_shape_info

    def get_inputs_data(self, inputs_tensor_info):
        names, shapes, dtypes = [], [], []
        for x in inputs_tensor_info:
            names.append(x["name"])
            shapes.append(x["shape"])
            dtypes.append(self._tensor2numpy_for_type(x["type"]))
        if not self.input_path:
            DirPool.make_input_dir()
            input_dir = DirPool.get_input_dir()
            input_map = self._generate_random_input_data(input_dir, names, shapes, dtypes)
        else:
            input_map = self._read_input_data(self.input_path, names, shapes, dtypes)
        return input_map

    def process_tensor_shape(self, tensor_name, tensor_type, tensor_shape):
        tensor_shape_info_list = []
        if self._is_dynamic_shape(tensor_shape):
            if not self.input_shape:
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    f"The dynamic shape {tensor_shape} are not supported. Please "
                    f'set "shape" of {tensor_name} in "input" to fix the dynamic shape.',
                )
            if tensor_name not in self.input_shape:
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    f'{tensor_name} has a dynamic shape, but its shape is not defined in the "input".',
                )
        if self.input_shape:
            inshape = self.input_shape.get(tensor_name)
            tensor_shape_info = self._get_input_shape_info(tensor_name, tensor_shape, inshape, tensor_type)
            tensor_shape_info_list.append(tensor_shape_info)
        else:
            tensor_shape_info = {"name": tensor_name, "shape": tensor_shape, "type": tensor_type}
            tensor_shape_info_list.append(tensor_shape_info)
        return tensor_shape_info_list
