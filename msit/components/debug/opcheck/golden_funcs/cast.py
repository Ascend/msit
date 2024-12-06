# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

import numpy
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
    print("no need disable v2")
import numpy as np
import torch
from components.debug.opcheck.conversion.dtype_convert import numpy_to_torch_tensor, str_to_dtype
from components.debug.opcheck.graph_parser import OpInfo


def _cast(context: OpInfo):
    import tensorflow as tf
    input0 = context.param.get("input_arrays")[0]
    type = context.param.get("stc_input_dtypes")[0]
    dst_type = context.param.get("output_dtypes")[0]
    if type == "float32" and dst_type == "int64":
        out = torch.tensor(input0, dtype=torch.int64).numpy()
        return out
    if dst_type == "complex32":
        _shape = list(input0.shape)
        input0 = input0.reshape(_shape + [1])
        imag = numpy.zeros(_shape + [1], dtype=numpy.float16)
        res = numpy.concatenate((input0, imag), axis=-1)
        return res
    if type == "uint1":
        input0 = np.unpackbits(input0)
    if type == "float32" and dst_type == "float16":
        out = torch.tensor(input0, dtype=torch.float16).numpy()
        return out
    if dst_type == "complex64":
        out = tf.cast(input0, dtype=dst_type)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out)
        return res
    if type == "uint32":
        return input0.astype(dst_type)
    input0_tensor = numpy_to_torch_tensor(input0)
    out_dtype_torch = str_to_dtype(dst_type)
    out = input0_tensor.to(out_dtype_torch)
    if dst_type == "bfloat16":
        out = out.type(torch.float32)
        res = out.numpy().astype(tf.bfloat16.as_numpy_dtype)
    else:
        res = out.numpy()
    return res