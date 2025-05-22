#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse

import torch
import torch.multiprocessing as mp

from msmodelslim.quant.entry import _main, _dist_main


def main():
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("-p", "--plugin_path", type=str, required=True, help="插件文件路径")
    parser.add_argument("-t", "--quant_type", type=str, help="量化类型", default="w8a8")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_path", type=str, help="保存路径", default="./save")
    parser.add_argument("--dev_type", type=str, help="设备类型", default="npu")
    parser.add_argument("--offload_dev_type", type=str, help="offload设备类型", default="meta")
    parser.add_argument("--debug", action="store_true", help="是否开启debug模式")
    parser.add_argument("--layer_count", type=int, help="层数", default=0)
    parser.add_argument("--dist", action="store_true", help="是否开启分布式量化")
    parser.add_argument("--master_port", type=int, help="master端口", default=29505)
    args = parser.parse_args()

    if args.dist:
        device_count = torch.cuda.device_count()
        if device_count < 2:
            raise ValueError("分布式量化需要至少2个设备")
        mp.spawn(
            _dist_main,
            args=(device_count, args),
            nprocs=device_count
        )
    else:
        _main(args)


if __name__ == "__main__":
    main()
