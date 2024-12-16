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
from sys import stdout

from time import strftime, localtime, time


def get_pid():
    return os.getpid()


def get_current_time(used_for_log=True, microsecond=False):
    if used_for_log:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        if microsecond:
            return round(time() * 1e6) % 10**10
        else:
            return strftime("%Y%m%d_%H%M%S", localtime())


def stdout_flush():
    stdout.flush()


def convert_bytes(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} Bytes"
    elif bytes_size < 1_048_576:  # 1024 * 1024
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1_073_741_824:  # 1024 * 1024 * 1024
        return f"{bytes_size / (1_048_576):.2f} MB"
    else:
        return f"{bytes_size / (1_073_741_824):.2f} GB"
