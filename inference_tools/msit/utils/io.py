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
import csv
import json
from functools import wraps

import yaml
import numpy as np
import pandas as pd
from onnx import load_model, save_model
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession

from msit.common.exceptions import MsitException
from msit.common.dependencies import import_tensorflow
from msit.common.constants import PathConst, MsgConst, DumpConst
from msit.utils.path import MsitPath, change_permission

tf = import_tensorflow()


class SafelyOpen:
    def __init__(self, file_path, mode, file_size_limitation=None, suffix=None, encoding="utf-8"):
        self.file_path = MsitPath(file_path, PathConst.FILE, mode, file_size_limitation, suffix).check()
        self.mode = mode
        self.encoding = encoding
        self._file = None

    def __enter__(self):
        if PathConst.BINARY_MODE not in self.mode:
            self._file = open(self.file_path, self.mode, encoding=self.encoding)
        else:
            self._file = open(self.file_path, self.mode)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()


def _load_file(mode, file_size, file_suffix, use_safely_open: bool, encoding="utf-8"):
    def decorator(func):
        @wraps(func)
        def wrapper(path, *args, **kwargs):
            try:
                if use_safely_open:
                    with SafelyOpen(path, mode, file_size, file_suffix, encoding) as f:
                        return func(f)
                else:
                    path = MsitPath(path, PathConst.FILE, mode, file_size, file_suffix).check()
                    return func(path, *args, **kwargs)
            except Exception as e:
                raise MsitException(MsgConst.IO_FAILURE, MsgConst.LOAD_ERROR.format(path, func.__name__)) from e
        return wrapper
    return decorator


def _load_dir(dir_size):
    def decorator(func):
        @wraps(func)
        def wrapper(path, *args, **kwargs):
            path = MsitPath(path, PathConst.DIR, "r", dir_size).check()
            try:
                return func(path, *args, **kwargs)
            except Exception as e:
                raise MsitException(MsgConst.IO_FAILURE, MsgConst.LOAD_ERROR.format(path, func.__name__)) from e
        return wrapper
    return decorator


def _save_file(mode, file_size, file_suffix, use_safely_open: bool):
    def decorator(func):
        @wraps(func)
        def wrapper(data, path, *args, **kwargs):
            try:
                if use_safely_open:
                    with SafelyOpen(path, mode, file_size, file_suffix) as f:
                        func(data, f, *args, **kwargs)
                else:
                    path = MsitPath(path, PathConst.FILE, mode, file_size, file_suffix).check()
                    func(data, path, *args, **kwargs)
            except Exception as e:
                raise MsitException(MsgConst.IO_FAILURE, \
                                    MsgConst.SAVE_ERROR.format(data.__class__.__name__, path, func.__name__)) from e
            change_permission(path, PathConst.AUTHORITY_FILE)
        return wrapper
    return decorator


def _save_dir(dir_size):
    def decorator(func):
        @wraps(func)
        def wrapper(data, path, *args, **kwargs):
            path = MsitPath(path, PathConst.DIR, "w", dir_size).check()
            try:
                func(data, path, *args, **kwargs)
            except Exception as e:
                raise MsitException(MsgConst.IO_FAILURE, \
                                    MsgConst.SAVE_ERROR.format(data.__class__.__name__, path, func.__name__)) from e
            change_permission(path, PathConst.AUTHORITY_DIR)
        return wrapper
    return decorator


@_load_file("r", PathConst.SIZE_20G, PathConst.SUFFIX_ONNX, use_safely_open=False)
def load_onnx_model(model_path):
    return load_model(model_path)


@_load_file("r", PathConst.SIZE_20G, PathConst.SUFFIX_ONNX, use_safely_open=False)
def load_onnx_session(model_path, onnx_fusion_switch=True, provider=PathConst.CPUEXECUTE):
    options = SessionOptions()
    if not onnx_fusion_switch:
        options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    return InferenceSession(model_path, sess_options=options, providers=[provider])


@_save_file("w", None, PathConst.SUFFIX_ONNX, use_safely_open=False)
def save_onnx_model(onnx_model, save_path):
    model_size = onnx_model.ByteSize()
    save_external_flag = model_size < 0 or model_size > DumpConst.MAX_PROTOBUF_2G
    save_model(onnx_model, save_path, save_as_external_data=save_external_flag)


@_load_file("r", PathConst.SIZE_10G, PathConst.SUFFIX_NPY, use_safely_open=False)
def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=False)


@_save_file("w", None, PathConst.SUFFIX_NPY, use_safely_open=False)
def save_npy(npy_data, save_path):
    np.save(save_path, npy_data)


@_save_file("w", None, PathConst.SUFFIX_BIN, use_safely_open=False)
def save_bin_from_ndarray(numpy_data: np.ndarray, save_path):
    numpy_data.tofile(save_path)


@_load_file("r", PathConst.SIZE_10G, PathConst.SUFFIX_BIN, use_safely_open=False)
def load_bin_to_ndarray(bin_path, dtype=np.float16, shape=None):
    if dtype == np.float32 and os.path.getsize(bin_path) == np.prod(shape) * 2:
        return np.fromfile(bin_path, dtype=np.float16).astype(np.float32)
    else:
        return np.fromfile(bin_path, dtype=dtype)


@_load_dir(PathConst.SIZE_50G)
def load_saved_model():
    pass


@_save_dir(PathConst.SIZE_50G)
def save_saved_model():
    pass


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_YAML, use_safely_open=True)
def load_yaml(f):
    return yaml.safe_load(f)


@_save_file("w", None, PathConst.SUFFIX_YAML, use_safely_open=True)
def save_yaml(yaml_data, f):
    yaml.dump(yaml_data, f)


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_JSON, use_safely_open=True)
def load_json(f):
    return json.load(f)


@_save_file("w", None, PathConst.SUFFIX_JSON, use_safely_open=True)
def save_json(json_data, f, indent=None):
    json.dump(json_data, f, indent=indent, default=str)


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_CSV, use_safely_open=True, encoding="utf-8-sig")
def load_csv_by_builtin(f, sep=","):
    csv_reader = csv.reader(f, delimiter=sep)
    return list(csv_reader)


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_CSV, use_safely_open=False)
def load_csv_by_pandas(csv_path, sep=","):
    return pd.read_csv(csv_path, sep=sep)


@_save_file("w", None, PathConst.SUFFIX_CSV, use_safely_open=False)
def save_csv_by_pandas(csv_data: pd.DataFrame, csv_path, sep=","):
    csv_data.to_csv(csv_path, sep=sep)
