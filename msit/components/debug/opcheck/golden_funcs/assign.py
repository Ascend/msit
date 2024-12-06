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
from components.debug.opcheck.graph_parser import OpInfo


def _assign_add(context: OpInfo):
    inputs = context.param.get("input_arrays")
    import tensorflow as tf
    #tf.compat.v1.disable_eager_execution()
    #ref = tf.Variable(inputs[0])
    #value = inputs[1]
    #output_data = tf.compat.v1.assign_add(ref, value, name="assignadd")
    #init = tf.compat.v1.global_variables_initializer()
    #with tf.compat.v1.Session() as sess:
    #    sess.run(init)
    #    result = sess.run(output_data)
    ref = inputs[0]
    value = inputs[1]
    ref_dtype = ref.dtype
    if ref_dtype == tf.bfloat16.as_numpy_dtype:
        ref = ref.astype("float32")
        value = value.astype("float32")
    x_dtype = ref.dtype
    result = numpy.add(ref, value)

    if context.param.get("output_dtypes")[0] == "bfloat16":
        return result.astype(tf.bfloat16.as_numpy_dtype)
    else:
        return result

def _assign_sub(context: OpInfo):
    var = context.param.get("input_arrays")[0]
    value = context.param.get("input_arrays")[1]
    dtype = value.dtype
    if dtype == tf.bfloat16.as_numpy_dtype:
        var = var.astype("float32")
        value = value.astype("float32")
    result = numpy.subtract(var, value)
    if context.param.get("output_dtypes")[0] == "bfloat16":
        return result.astype(tf.bfloat16.as_numpy_dtype)
    else:
        return result