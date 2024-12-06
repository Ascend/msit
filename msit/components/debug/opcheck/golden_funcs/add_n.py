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

import numpy as np
from components.debug.opcheck.graph_parser import OpInfo


def _conv2d(context: OpInfo):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, _ = context.param.get("input_arrays")
    _, conv_filter_ori_shape, _, _ = context.param.get("dyn_ori_inputs")
    strides = context.param.get("strides")
    pads = context.param.get("pads")
    dilations = context.param.get("dilations")
    groups = context.param.get('groups') #('groups', 1)
    data_format = context.param.get("data_format") #("data_format", "NCHW")
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    C1KHKW, Cout1, Cout0, C0 = conv_filter.shape
    _, _, KH, KW = conv_filter_ori_shape
    ON = IN
    OC = Cout1 * Cout0
    OH = (IH + pad_top + pad_bottom - (dilationh * (KH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (KW - 1) + 1)) // stridew + 1
    # 5HD to HWCN
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC*C0).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    # x filter to NHWC
    conv_filter = conv_filter.transpose(0, 3, 1, 2).reshape(C0*C1KHKW, Cout1, Cout0)
    conv_filter = conv_filter.reshape(C0*IC, KH*KW, Cout1, Cout0)
    conv_filter = conv_filter.transpose(1, 0, 2, 3).reshape(KH, KW, C0*IC, Cout1, Cout0)
    conv_filter = conv_filter.reshape(KH, KW, C0*IC, Cout1*Cout0).astype(np.float32)
    if groups > 1:
        Cout_per_group = conv_filter.shape[3] // groups
        Cin_per_group = conv_filter.shape[2] // groups
        split_data = tf.split(value=tensor_x,
                              num_or_size_splits=groups,
                              axis=3,
                              name='split1')
        if bias is not None:
            bias = bias.astype(np.float32)
        split_per_group_conv = []
        split_group_index = 0
        for data in split_data:

            conv_filter_per_group = \
                conv_filter[:, :, 0:Cin_per_group,
                split_group_index * Cout_per_group: (split_group_index + 1) * Cout_per_group]

            tf_conv2d_result = tf.nn.conv2d(data, conv_filter_per_group,
                                            strides=(strideh, stridew),
                                            padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                            data_format="NHWC",
                                            use_cudnn_on_gpu=False,
                                            dilations=dilations)
            if bias is not None:
                bias_split_data = \
                    bias[split_group_index * Cout_per_group:(split_group_index + 1) * Cout_per_group]
                tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, bias_split_data)
            split_group_index += 1
            split_per_group_conv.append(tf_conv2d_result)

        output = tf.concat(split_per_group_conv, axis=3, name='concat1')
        init_op = tf.compat.v1.global_variables_initializer()
        feed_dict = {tensor_x: x}
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(output, feed_dict=feed_dict)
    else:
        # Group <= 1
        tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
        tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                        padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                        data_format="NHWC", dilations=dilations)
        feed_dict = {tensor_x: x, tensor_filter: conv_filter}
        if bias is not None:
            bias = bias.astype(np.float32)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
        fusion_mode = None
        if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
            out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).astype(np.float16)
    return output
