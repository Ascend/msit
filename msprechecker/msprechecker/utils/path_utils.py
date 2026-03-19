# -*- coding: utf-8 -*-
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
import os
import sys


class PathUtil:

    @staticmethod
    def get_root_dir_path():
        """
        获取安装路径的根路径
        """
        if hasattr(sys, "_MEIPASS"):
            return os.path.dirname(sys.executable)
        current_dir_path = os.path.dirname(__file__)
        root_dir_index = os.path.abspath(os.path.join(current_dir_path, "..", ".."))
        return root_dir_index

    @staticmethod
    def get_output_root_dir_path():
        """
        获取输出报告根目录
        """
        return os.path.join(PathUtil.get_root_dir_path(), 'output')

    @staticmethod
    def get_log_root_dir_path():
        """
        获取运行日志文件根目录
        """
        return os.path.join(PathUtil.get_root_dir_path(), 'logs')

    @staticmethod
    def get_resources_root_dir_path():
        """
        获取资源文件根目录
        """
        return os.path.join(PathUtil.get_root_dir_path(), 'resources')

    @staticmethod
    def _get_meipass():
        return getattr(sys, '_MEIPASS', None)