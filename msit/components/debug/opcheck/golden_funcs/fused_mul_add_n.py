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

import tensorflow as tf
from components.debug.opcheck.graph_parser import OpInfo


def _fused_mul_add_n(context: OpInfo):
    input0, input1, input2 = context.param.get("input_arrays")
    output_dtype = context.param.get("output_dtypes")[0]
    if str(output_dtype) == "bfloat16":
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
        input2 = input2.astype("float32")
    input0_holder = tf.placeholder(shape=input0.shape,
                                      dtype=input0.dtype)
    input1_holder = tf.placeholder(shape=input1.shape,
                                      dtype=input1.dtype)
    input2_holder = tf.placeholder(shape=input2.shape,
                                      dtype=input2.dtype)
    output_data1 = tf.multiply(input0_holder, input2_holder)

    output_data = tf.add(output_data1, input1_holder)

    feed_dict = {input0_holder: input0, input1_holder: input1, input2_holder: input2}

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run(output_data, feed_dict=feed_dict)
    if str(output_dtype) == "bfloat16":
        res = res.astype(tf.bfloat16.as_numpy_dtype)
    #if output_dtype == "float16":
    #    res = due_fp16_overflow(res)
    return res 