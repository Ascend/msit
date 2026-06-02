# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=duplicate-code

import os
from pathlib import Path


def is_in_container():
    def check_docker_env_file():
        docker_env_file = "/.dockerenv"
        return os.path.exists(docker_env_file)

    def check_first_process():
        first_proc = "/proc/1"
        schedule_file = os.path.join(first_proc, "sched")

        try:
            with Path(schedule_file).open(encoding="utf-8") as f:
                first_line = f.readlines(1)
        except Exception:
            return True

        if first_line and first_line[0] and first_line[0].startswith("systemd"):
            return False

        return True

    return check_docker_env_file() or check_first_process()


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
