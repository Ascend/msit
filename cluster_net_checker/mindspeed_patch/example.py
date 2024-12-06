# Copyright (c) 2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

"""
1.导入文件：
    将group_info.py文件放入mindspeed代码仓，路径任意，例如 MindSpeed/mindspeed/core
2.打开动态profiling代码（若已有动态profiling，则无需打开）：
    在MindSpeed/mindspeed/core/training.py中的train_decorator函数，加入动态profiling开关，参考下面36-37行
3.设置动态profiling config：
    msprof_tx 需设置为 true （通信域落盘基于mstx）
"""


def train_decorator(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        from torch_npu.profiler import dynamic_profile as dp
        dp.init('./profiler_config')

        if hasattr(args_, 'profile_npu') and args_.profile_npu \
                and (torch.distributed.get_rank() in args_.profile_ranks):
            """
            ...
            """
        else:
            return train(*args, **kwargs)

    return wrapper


"""
1.插入拓扑信息打点代码：
    在MindSpeed/mindspeed/core/training.py中的train_step_decorator函数中加入拓扑信息打点代码，参考98-103行
2.动态profiling step（若已有动态profiling，则无需加）:
    加入动态profiling的step函数，参考106行
"""


def train_step_decorator(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        nonlocal train_step
        args_ = get_args()
        flop_count = None

        # 此处开始为通信域打点代码
        from group_info import GroupInfo
        mstx = torch_npu.npu.mstx()
        gp_info = GroupInfo()
        gp_info_list = gp_info.get_group_info()
        for gp_info in gp_info_list:
            mstx.mark(gp_info)

        dp.step()  # 动态profiling 一个step的结束，如果已经有的话就不需要加

        if args_.op_cal_tflops:
            flop_count = get_flops_counter()
            flop_count.start()
        """
        ...
        """

    return wrapper
