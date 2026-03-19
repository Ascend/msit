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
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
from abc import ABCMeta
from datetime import timezone, timedelta, datetime
from enum import Enum
from functools import lru_cache

import unicodedata
from msguard.security import open_s

from .util import Framework
from .logger import LOGGER

TIME_OUT = 30


class Color:
    default_format = "\033[38m%s\033[0m"
    red_format = "\033[31m%s\033[0m"
    green_format = "\033[32m%s\033[0m"
    yellow_format = "\033[33m%s\033[0m"
    blue_format = "\033[34m%s\033[0m"
    bold_format = "\033[1m%s\033[0m"


class BoxChars:
    STYLES = {
        'default': {  # 默认实线框
            'UL': "┌", 'UR': "┐",
            'LL': "└", 'LR': "┘",
            'HZ': "─", 'VT': "│",
            'LT': "├", 'RT': "┤"
        },
        'double': {  # 双线框
            'UL': "╔", 'UR': "╗",
            'LL': "╚", 'LR': "╝",
            'HZ': "═", 'VT': "║",
            'LT': "╠", 'RT': "╣"
        },
        'rounded': {  # 圆角框
            'UL': "╭", 'UR': "╮",
            'LL': "╰", 'LR': "╯",
            'HZ': "─", 'VT': "│",
            'LT': "├", 'RT': "┤"
        }
    }

    def __init__(self, style: str = 'default', empty: bool = False):
        self._style = None
        self.empty = empty
        self.set_style(style if not empty else None)

    def set_style(self, style: str):
        if not style:
            self._disable_borders()
            return

        charset = self.STYLES.get(style, self.STYLES['default'])
        for key, val in charset.items():
            setattr(self, key, val)
        self._style = style

    def _disable_borders(self):
        for key in ('UL', 'UR', 'LL', 'LR', 'HZ', 'VT', 'LT', 'RT'):
            setattr(self, key, "")

    @property
    def current_style(self) -> str:
        """获取当前样式名称"""
        return self._style or 'empty'


class Utils:
    cmd_result_cache = {}

    @staticmethod
    def color_green(msg):
        return Color.green_format % msg

    @staticmethod
    def color_red(msg):
        return Color.red_format % msg

    @staticmethod
    def color_yellow(msg):
        return Color.yellow_format % msg

    @staticmethod
    def color_bold(msg):
        return Color.bold_format % msg

    @staticmethod
    def color_blue(msg):
        return Color.blue_format % msg

    @staticmethod
    def get_term_size(default_width=120, default_height=24):
        """
        安全获取终端大小
        """
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except (OSError, AttributeError):
            return default_width, default_height

    @staticmethod
    def is_valid_ip(ip: str):
        single_address = "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
        ip_pattern = re.compile(rf"\b{single_address}(?:\.{single_address}){{3}}\b")

        return bool(ip_pattern.match(ip))

    @staticmethod
    def print_line(style: str = '='):
        """打印分隔线"""
        width, _ = Utils.get_term_size()
        Output.message(style * width)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_time_stamp(connector: str = "-"):
        tz_utc = timezone(timedelta(hours=8))
        return datetime.now(tz_utc).strftime(f"%Y%m%d{connector}%H%M%S")

    @staticmethod
    @lru_cache(maxsize=None)
    def is_in_container():
        def check_docker_env_file():
            docker_env_file = "/.dockerenv"
            return os.path.exists(docker_env_file)

        def check_first_process():
            first_proc = "/proc/1"
            schedule_file = os.path.join(first_proc, "sched")

            try:
                with open_s(schedule_file) as f:
                    first_line = f.readlines(1)
            except (OSError, IOError):
                return True

            if first_line and first_line[0] and first_line[0].startswith("systemd"):
                return False

            return True

        return check_docker_env_file() or check_first_process()

    @staticmethod
    def singleton(cls):
        instances = {}

        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

        return get_instance

    @staticmethod
    def log_error_and_exit(message, message_args=(), code=2):
        if message_args:
            message = message.format(*message_args)
        LOGGER.error(message)
        raise CustomError(message, code)

    @staticmethod
    def split_text(help_text, width=120):
        lines = help_text.splitlines()
        wrapped_lines = []
        for line in lines:
            current_line = ""
            current_width = 0

            for char in line:
                # 计算当前字符的宽度
                char_width = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1

                # 如果加上当前字符会超出限制且当前行不为空，则换行
                if current_width + char_width > width and current_line:
                    wrapped_lines.append(current_line)
                    current_line = char
                    current_width = char_width
                else:
                    current_line += char
                    current_width += char_width

            # 添加最后一行
            if current_line:
                wrapped_lines.append(current_line)

        return wrapped_lines

    @staticmethod
    def color_cell(text, width, alignment=">"):
        text = str(text)
        # 计算填充后的文本宽度
        display_width = Utils.get_text_real_length(text)

        # 计算剩余宽度以进行填充
        remaining_width = width - display_width

        # 根据对齐方式生成填充文本
        if alignment == "<":
            formatted_text = text + " " * remaining_width
        elif alignment == ">":
            formatted_text = " " * remaining_width + text
        else:
            # 默认居中对齐
            left_padding = remaining_width // 2
            right_padding = remaining_width - left_padding
            formatted_text = " " * left_padding + text + " " * right_padding

        return formatted_text

    @staticmethod
    def load_json(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError:
            LOGGER.error("File {} is not JSON type".format(report_path))
            return {}
        except Exception as e:
            LOGGER.error("Error: while reading {}: {}".format(report_path, e))
            return {}

    @staticmethod
    def dump_file(file_path, data):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(data)
        except Exception as e:
            Utils.log_error_and_exit("Error: while writing {}: {}".format(file_path, e))
        LOGGER.info("Generate {} successfully".format(file_path))

    @staticmethod
    def exec_cmd_base(args, env=None, shell=False):
        cache_key = str(args) + str(env) + str(shell)
        if cache_key in Utils.cmd_result_cache:
            return Utils.cmd_result_cache.get(cache_key)
        try:
            # Run command to get module info
            LOGGER.debug("Execute cmd: %s", ' '.join(args))
            os_env = os.environ.copy()
            if hasattr(sys, "_MEIPASS") and "LD_LIBRARY_PATH" in os_env:
                # 打包后要去掉LD_LIBRARY_PATH环境变量
                del os_env["LD_LIBRARY_PATH"]
            if env:
                os_env.update(env)
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell,
                                    universal_newlines=True, env=os_env, timeout=TIME_OUT)
            Utils.cmd_result_cache[cache_key] = {"result": result, "err": ""}
            return Utils.cmd_result_cache.get(cache_key)
        except subprocess.CalledProcessError as e:
            error_message = "Fail to execute cmd: {}".format(e.stderr)
            LOGGER.error(error_message)
        except subprocess.TimeoutExpired as e:
            error_message = "Timeout {}s while execute cmd: {}".format(TIME_OUT, args[0])
            LOGGER.error(error_message)
        except FileNotFoundError:
            error_message = "Command {} not found".format(args[0])
            LOGGER.error(error_message)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            error_message = "Unknown error: {}".format(str(e))
        Utils.cmd_result_cache[cache_key] = {"result": "", "err": str(error_message)}
        return Utils.cmd_result_cache.get(cache_key)

    @staticmethod
    def exec_cmd(args, env=None, shell=False) -> (bool, str, str):
        result_dict = Utils.exec_cmd_base(args, env, shell)
        error_message = result_dict["err"]
        result = result_dict["result"]
        if error_message:
            return False, "--", str(error_message)
        # Check for errors
        if result.returncode != 0:
            LOGGER.error("Failed to retrieve info. Return code: %d", result.returncode)
            LOGGER.error("Stderr: %s", result.stderr)

        # Parse the output to get the Manufacturer and Product Name
        result_out = result.stdout
        if not result_out:
            LOGGER.warning("No stdout from command, record stderr.")
            return False, result.stderr, result.stderr
        else:
            value = result.stdout
        LOGGER.debug("Successfully execute cmd: {}".format(' '.join(args)))
        return True, value, result.stderr

    @staticmethod
    def collect_data(args, env=None, err_msg=None):
        flag, output, err = Utils.exec_cmd(args, env)
        if flag:
            return str(output)
        if isinstance(err_msg, dict):
            err_msg["msg"] = err
        return '--'

    @staticmethod
    def grep_lines(lines, keyword):
        for line in lines.strip().splitlines():
            if keyword in line:
                return line.split(":")[-1].strip()
        return ""

    @classmethod
    def get_text_real_length(cls, text):
        """
        计算文本长度（中文字符计为 2 个字符）
        """
        length = 0
        for char in str(text):
            char_width = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
            length += char_width
        return length

    @classmethod
    def get_max_string_length(cls, info_dict, level=0):
        """
        Recursively get the maximum length of string values in the most inner level of a nested dictionary.
        """
        max_length = 0

        for value in info_dict.values():
            if isinstance(value, dict):
                # 如果值是字典，递归调用
                max_length = max(max_length, cls.get_max_string_length(value, level + 1))
            elif isinstance(value, str):
                # 如果值是字符串，获取长度
                split_value_list = value.splitlines()
                value_length = max(Utils.get_text_real_length(str(value)) for value in split_value_list)
                max_length = max(max_length, value_length)

        return max_length + 4 * level


class Output(object):
    stderr_only = False
    is_debug = False
    color_format = Color.default_format

    @classmethod
    def format_severity(cls, severity):
        return '[%s] ' % severity if severity else ''

    @classmethod
    def debug_message(cls, *args):
        cls.color_format = Color.blue_format
        return cls.do_message(True, sys.stdout, 'DEBUG', *args)

    @classmethod
    def bold_message(cls, *args):
        cls.color_format = Color.bold_format
        return cls.do_message(True, sys.stdout, '', *args)

    @classmethod
    def message(cls, *args):
        cls.color_format = Color.default_format
        return cls.do_message(True, sys.stdout, '', *args)

    @classmethod
    def message_without_wrap(cls, *args):
        cls.color_format = Color.default_format
        return cls.do_message(False, sys.stdout, '', *args)

    @classmethod
    def info_message(cls, *args):
        cls.color_format = Color.green_format
        return cls.do_message(True, sys.stdout, 'INFO', *args)

    @classmethod
    def warning_message(cls, *args):
        cls.color_format = Color.yellow_format
        return cls.do_message(True, sys.stdout, 'WARNING', *args)

    @classmethod
    def error_message(cls, *args):
        cls.color_format = Color.red_format
        return cls.do_message(True, sys.stdout, 'ERROR', *args)

    @classmethod
    def do_message(cls, add_newline, destination, severity, *args):
        real_destination = sys.stderr if cls.stderr_only else destination
        out_format = ("%s%s\n" if add_newline else "%s%s")
        out_str = out_format % (cls.format_severity(severity), " ".join([str(a) for a in args]))
        out_msg = cls.color_format % out_str
        # Get the binary stream if it's a text stream
        binary_dest = real_destination.buffer if hasattr(real_destination, 'buffer') else real_destination
        binary_dest.write(out_msg.encode('utf-8', errors='ignore'))
        binary_dest.flush()


class CustomError(Exception):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.message = message
        self.code = code


class SingletonMeta(ABCMeta):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class PreFetch:
    weight_dir = None

    @staticmethod
    @lru_cache(maxsize=None)
    def get_framework():
        if not Utils.is_in_container():
            return Framework.HOST

        if os.path.isdir('/vllm-workspace/vllm-ascend') or shutil.which("vllm"):
            return Framework.VLLM

        if shutil.which("sglang"):
            return Framework.SGLANG

        mindie_dir = '/usr/local/Ascend/mindie'
        if os.path.isdir(mindie_dir) or any('MINDIE' in env_name for env_name in os.environ):
            framework = Framework.MINDIE
            return framework

        return Framework.UNKNOWN

    @staticmethod
    def search_weight_dir_mindie(mies_config_path=''):
        if not mies_config_path:
            mies_install_path = os.environ.get('MIES_INSTALL_PATH')
            if mies_install_path is None or not os.path.isdir(mies_install_path):
                default_path = '/usr/local/Ascend/mindie/latest/mindie-service'
                LOGGER.info('MIES_INSTALL_PATH not set or invalid in env, using default path: {}'.format(
                    default_path))
                mies_install_path = default_path
            else:
                LOGGER.info('Detected MIES_INSTALL_PATH from env: {}'.format(mies_install_path))
            mies_config_path = os.path.join(os.path.join(mies_install_path, 'conf', 'config.json'))
            if not os.path.isfile(mies_config_path):
                Utils.log_error_and_exit('MIES configuration file not exist at: {}'.format(mies_config_path))
        LOGGER.info('Reading MIES configuration file: {}'.format(mies_config_path))
        try:
            with open_s(mies_config_path) as f:
                data = json.load(f)
        except Exception as e:
            Utils.log_error_and_exit('Failed to parse MIES configuration file: {}'.format(str(e)))

        try:
            return data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                "modelWeightPath"
            ]
        except KeyError as e:
            Utils.log_error_and_exit('Missing required configuration key in MIES config file: {}', str(e))

    @staticmethod
    def search_weight_dir_vllm(script_path=''):
        if not script_path:
            Utils.log_error_and_exit('Empty script path for VLLM weight path search.')

        try:
            with open(script_path) as f:
                content = f.read()
        except Exception as e:
            Utils.log_error_and_exit('Failed to read script file {}.'.format(script_path))

        weight_pattern = re.compile(
            r"""vllm\s+serve\s+(?:["']?)([^\s"']+)|--model(?:["']?\s*[=\s]?\s*(?:["']?)([^\s"']+))""")

        mo = weight_pattern.search(content)
        if mo:
            return mo.group(1) or mo.group(2)
        else:
            Utils.log_error_and_exit('Failed to find VLLM weight path in script file {}.'.format(script_path))

    @staticmethod
    def search_weight_dir_sglang(script_path=''):
        pass

    @classmethod
    def get_weight_dir(cls, path=''):
        if cls.weight_dir:
            return cls.weight_dir
        framework_to_search_method = {
            Framework.MINDIE: cls.search_weight_dir_mindie,
            Framework.VLLM: cls.search_weight_dir_vllm,
            Framework.SGLANG: cls.search_weight_dir_sglang
        }
        framework = cls.get_framework()
        if framework not in framework_to_search_method:
            Utils.log_error_and_exit('Unsupported framework {} for weight directory search.'.format(framework))
        weight_dir = framework_to_search_method[framework](path)
        if not os.path.exists(weight_dir):
            Utils.log_error_and_exit('Model weight path {} not exist.'.format(weight_dir))
        else:
            LOGGER.info('Got Model weight path: {}'.format(weight_dir))
        cls.weight_dir = weight_dir
        return cls.weight_dir
