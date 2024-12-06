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


def dsl_gather_v2(context: OpInfo):
    """
    A numpy implementation of Gather
    """
    params_data = context.param.get("input_arrays")[0]
    params_shape_len = len(params_data.shape)

    indices_data = context.param.get("input_arrays")[1]
    indices_shape_len = len(indices_data.shape)

    if "batch_dims" in context.param.get("other_runtime_params"):
        batch_dims = context.param.get("batch_dims")
    else:
        batch_dims = 0

    batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len

    if "axis_dict" in context.param.get("other_runtime_params"):
        axis = context.param.get("axis_dict")
    else:
        if "axis" in context.param.get("other_runtime_params"):
            axis = context.param.get("axis")
        else:
            axis = 0

    axis = axis if axis >= 0 else axis + params_shape_len

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # indices 1
    params_shape = params_data.shape
    indices_shape = indices_data.shape

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(params, indices, axis=axis, batch_dims=batch_dims, name=None)
        res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})

    return res