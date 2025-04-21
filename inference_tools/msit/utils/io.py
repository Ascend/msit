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

import csv
import json
import pickle
from functools import wraps

import numpy as np
import pandas as pd
import yaml

from msit.utils.constants import MsgConst, PathConst
from msit.utils.dependencies import dependent
from msit.utils.exceptions import MsitException
from msit.utils.log import logger
from msit.utils.path import (
    AUTHORITY_DIR,
    AUTHORITY_FILE,
    MsitPath,
    change_permission,
    get_basename_from_path,
    get_file_size,
    join_path,
)
from msit.utils.toolkits import CHECK_CSV_LEVEL_STRICT, is_input_yes, sanitize_csv_value

_LOAD_ERROR = 'Failed to load the path "{}" using <{}>.'
_SAVE_ERROR = 'Failed to save {} to "{}" using <{}>. Please check permissions or disk space.'


class SafelyOpen:
    def __init__(self, file_path, mode, file_size_limitation=None, suffix=None, path_exist=True, encoding="utf-8"):
        self.file_path = MsitPath(file_path, PathConst.FILE, mode, file_size_limitation, suffix).check(
            path_exist=path_exist
        )
        self.mode = mode
        self.encoding = encoding
        self._file = None

    def __enter__(self):
        if "b" not in self.mode:
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
                raise MsitException(MsgConst.IO_FAILURE, _LOAD_ERROR.format(path, func.__name__)) from e

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
                raise MsitException(MsgConst.IO_FAILURE, _LOAD_ERROR.format(path, func.__name__)) from e

        return wrapper

    return decorator


def _save_file(mode, file_size, file_suffix, use_safely_open: bool):
    def decorator(func):
        @wraps(func)
        def wrapper(data, path, *args, **kwargs):
            try:
                if use_safely_open:
                    with SafelyOpen(path, mode, file_size, file_suffix, path_exist=False) as f:
                        func(data, f, *args, **kwargs)
                else:
                    path = MsitPath(path, PathConst.FILE, mode, file_size, file_suffix).check(path_exist=False)
                    func(data, path, *args, **kwargs)
            except Exception as e:
                raise MsitException(
                    MsgConst.IO_FAILURE, _SAVE_ERROR.format(data.__class__.__name__, path, func.__name__)
                ) from e
            change_permission(path, AUTHORITY_FILE)

        return wrapper

    return decorator


def _save_dir(dir_size):
    def decorator(func):
        @wraps(func)
        def wrapper(data, path, *args, **kwargs):
            path = MsitPath(path, PathConst.DIR, "w", dir_size).check(path_exist=False)
            try:
                func(data, path, *args, **kwargs)
            except Exception as e:
                raise MsitException(
                    MsgConst.IO_FAILURE, _SAVE_ERROR.format(data.__class__.__name__, path, func.__name__)
                ) from e
            change_permission(path, AUTHORITY_DIR)

        return wrapper

    return decorator


@_load_file("r", PathConst.SIZE_30G, PathConst.SUFFIX_ONNX, use_safely_open=False)
def load_onnx_model(model_path):
    onnx = dependent.get("onnx")
    return onnx.load_model(model_path)


@_load_file("r", PathConst.SIZE_30G, PathConst.SUFFIX_ONNX, use_safely_open=False)
def load_onnx_session(model_path, onnx_fusion_switch=True, provider="CPUExecutionProvider"):
    ort = dependent.get("onnxruntime")
    options = ort.SessionOptions()
    if not onnx_fusion_switch:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options, providers=[provider])


@_load_file("r", PathConst.SIZE_30G, PathConst.SUFFIX_OM, use_safely_open=False)
def load_om_model(model_path):
    om = dependent.get("msit.lib.msit_c")
    model_id, ret = om.acl.load_from_file(model_path)
    if ret != 0:
        raise MsitException(MsgConst.IO_FAILURE, f"Load model:{model_path} failed! ErrorCode = {ret}.")
    logger.info(f"Load model:{model_path} success!")
    return model_id


@_save_file("w", None, PathConst.SUFFIX_ONNX, use_safely_open=False)
def save_onnx_model(onnx_model, save_path):
    onnx = dependent.get("onnx")
    model_size = onnx_model.ByteSize()
    save_external_flag = model_size > PathConst.SIZE_2G
    onnx.save_model(onnx_model, save_path, save_as_external_data=save_external_flag)


@_load_file("r", PathConst.SIZE_30G, PathConst.SUFFIX_PROTOTXT, use_safely_open=False)
def load_caffe_model(model_path, weight_path):
    caffe = dependent.get("caffe")
    if caffe:
        caffe.set_mode_cpu()
        return caffe.Net(model_path, weight_path, caffe.TEST)
    return None


@_load_file("r", PathConst.SIZE_10G, PathConst.SUFFIX_NPY, use_safely_open=False)
def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=False)


def load_npy_from_buffer(raw_data, dtype, shape):
    try:
        return np.frombuffer(raw_data, dtype=dtype).reshape(shape)
    except Exception as e:
        raise MsitException(MsgConst.IO_FAILURE, "Failed to load npy data from buffer.") from e


@_save_file("w", None, PathConst.SUFFIX_NPY, use_safely_open=False)
def save_npy(npy_data, save_path):
    np.save(save_path, npy_data)


@_save_file("w", None, PathConst.SUFFIX_BIN, use_safely_open=False)
def save_bin_from_ndarray(numpy_data: np.ndarray, save_path):
    numpy_data.tofile(save_path)


@_load_file("r", PathConst.SIZE_10G, PathConst.SUFFIX_BIN, use_safely_open=False)
def load_bin_data(bin_path, dtype=np.float16, shape=None, is_byte_data=False):
    if is_byte_data:
        return np.fromfile(bin_path, dtype=np.int8)
    if dtype == np.float32 and get_file_size(bin_path) == np.prod(shape) * 2:
        return np.fromfile(bin_path, dtype=np.float16).astype(np.float32)
    else:
        return np.fromfile(bin_path, dtype=dtype)


@_load_dir(PathConst.SIZE_30G)
def load_saved_model(model_path, tag):
    pons = dependent.get_tensorflow()
    if None not in pons:
        tf, _, _ = pons
        tf.compat.v1.reset_default_graph()
        graph = tf.compat.v1.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        saved_model = tf.compat.v1.saved_model.loader.load(sess, set(tag), model_path)
        return saved_model, sess
    return None, None


@_load_file("rb", PathConst.SIZE_30G, PathConst.SUFFIX_PB, use_safely_open=False)
def load_pb_frozen_graph_model(model_path):
    pons = dependent.get_tensorflow()
    if None not in pons:
        tf, _, _ = pons
        data = tf.compat.v1.gfile.GFile(model_path, "rb").read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(data)
        tf.compat.v1.import_graph_def(graph_def, name="")
        return graph_def
    return None


@_save_file("wb", PathConst.SIZE_30G, PathConst.SUFFIX_PB, use_safely_open=False)
def save_pb_frozen_graph_model(frozen_graph, model_path):
    pons = dependent.get_tensorflow()
    if None not in pons:
        tf, _, _ = pons
        with tf.io.gfile.GFile(model_path, "wb") as f:
            f.write(frozen_graph)


def savedmodel2pb(model_path, tag, serve, pb_save_dir):
    """
    Converts a TensorFlow 1.x SavedModel to a frozen PB file.

    :param model_path: Path to the saved TensorFlow SavedModel directory
    :param tag: Tag used for loading the model
    :param serve: Signature key (e.g., "serving_default")
    :param pb_save_dir: Directory to save the PB file
    :return: Path to the converted PB file and net output nodes
    """
    pons = dependent.get_tensorflow()
    if None not in pons:
        _, _, sm2pb = pons
        meta_graph_def, sess = load_saved_model(model_path, tag)
        signature_def = meta_graph_def.signature_def.get(serve)
        if signature_def is None:
            raise MsitException(MsgConst.VALUE_NOT_FOUND, f'Signature "{serve}" not found in the model.')
        input_tensor_names = [t.name for t in signature_def.inputs.values()]
        output_tensor_names = [t.name for t in signature_def.outputs.values()]
        logger.info(f"Saved model input tensors: {input_tensor_names}.")
        logger.info(f"Saved model output tensors: {output_tensor_names}.")
        output_node_names = [t.split(":")[0] for t in output_tensor_names]
        frozen_graph_def = sm2pb(sess, sess.graph.as_graph_def(), output_node_names)
        pb_file_name = get_basename_from_path(model_path) + PathConst.SUFFIX_PB
        pb_file_path = join_path(pb_save_dir, pb_file_name)
        save_pb_frozen_graph_model(frozen_graph_def.SerializeToString(), pb_file_path)
        sess.close()
        logger.info(f"SavedModel has been successfully converted to a frozen PB file at {pb_file_path}.")
        return pb_file_path
    return ""


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_YAML, use_safely_open=True)
def load_yaml(f):
    return yaml.safe_load(f)


@_save_file("w", None, PathConst.SUFFIX_YAML, use_safely_open=True)
def save_yaml(yaml_data, f):
    yaml.dump(yaml_data, f)


@_load_file("r", PathConst.SIZE_2G, PathConst.SUFFIX_JSON, use_safely_open=True)
def load_json(f):
    return json.load(f)


@_save_file("w", None, PathConst.SUFFIX_JSON, use_safely_open=True)
def save_json(json_data, f, indent: int = None):
    json.dump(json_data, f, indent=indent, default=str)


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_CSV, use_safely_open=True, encoding="utf-8-sig")
def load_csv_by_builtin(f, sep=",", check=CHECK_CSV_LEVEL_STRICT):
    csv_reader = csv.reader(f, delimiter=sep)
    sanitized_rows = []
    for row in csv_reader:
        sanitized_row = [sanitize_csv_value(value, check) for value in row]
        sanitized_rows.append(sanitized_row)
    return sanitized_rows


@_load_file("r", PathConst.SIZE_500M, PathConst.SUFFIX_CSV, use_safely_open=False)
def load_csv_by_pandas(csv_path, sep=",", check=CHECK_CSV_LEVEL_STRICT):
    df = pd.read_csv(csv_path, sep=sep, dtype=str)
    df = df.applymap(lambda value: sanitize_csv_value(value, check))
    return df


@_save_file("w", None, PathConst.SUFFIX_CSV, use_safely_open=False)
def save_csv_by_pandas(csv_data: pd.DataFrame, csv_path, sep=",", check=CHECK_CSV_LEVEL_STRICT):
    sanitized_data = csv_data.applymap(lambda value: sanitize_csv_value(value, check))
    sanitized_data.to_csv(csv_path, sep=sep, index=False)


@_load_file("r", PathConst.SIZE_30G, None, use_safely_open=False)
def load_torch_obj(path, **kwargs):
    kwargs.setdefault("weights_only", True)
    try:
        torch = dependent.get("torch")
        return torch.load(path, **kwargs)
    except pickle.UnpicklingError:
        if kwargs["weights_only"]:
            prompt = """
            Weights only load failed. Re-running <torch.load> with `weights_only` set to `False` will likely succeed, 
            but it can result in arbitrary code execution. Do it only if you get the file from a trusted source. \n
            Please confirm your awareness of the risks associated with this action ([y]/n): """
            if not is_input_yes(prompt):
                return None
            kwargs["weights_only"] = False
            return torch.load(path, **kwargs)
        else:
            return None
