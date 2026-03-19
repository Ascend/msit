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

import logging
import time

MAX_LOG_MSG_SIZE = 2048


class ANSIColoredFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: '\033[4;34m',  # 蓝色
        logging.INFO: '\033[4;1;37m',  # 灰色
        logging.WARNING: '\033[4;33;5m',  # 黄色 + 闪烁
        logging.ERROR: '\033[4;31;5m',  # 红色 + 闪烁
        logging.CRITICAL: '\033[4;35;5m',  # 紫色 + 闪烁
        'RESET': '\033[0m',  # 重置样式
    }

    def formatTime(self, record, datefmt=None):
        ct = time.localtime(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        return time.strftime('%Y-%m-%d %H:%M:%S', ct)

    def format(self, record):
        message = super().format(record)
        if record.levelno in self.COLOR_MAP:
            # 确保时间戳存在
            record.asctime = self.formatTime(record)
            if record.level == logging.DEBUG:
                # debug级别增加时间和堆栈
                return (f"[{record.asctime}] "
                        f"{self.COLOR_MAP[record.levelno]}[{record.levelname}]{self.COLOR_MAP['RESET']} "
                        f"[{record.name}] "
                        f"[PID:{record.process}] "
                        f"[{record.filename}:{record.funcName}:{record.lineno}] "
                        f"{message}")
            else:
                return (f"{self.COLOR_MAP[record.levelno]}[{record.levelname}]{self.COLOR_MAP['RESET']} "
                        f"[{record.name}] "
                        f"{message}")
        return message

    password_name_tuple = ('password', 'check_password', 'sql_pwd')


class LogHandler(logging.Handler):
    def __init__(self, **kwargs):
        super().__init__()
        self.component_name = kwargs.get('component_name')
        self.tag_name = kwargs.get('tag_name')

    def format(self, record):
        msg = super().format(record)
        msg = msg.replace("\n", "\\n")
        if len(msg) > MAX_LOG_MSG_SIZE:
            msg = msg[:MAX_LOG_MSG_SIZE] + "..."
        return msg


class LeveledLogger(logging.Logger):
    def handle(self, record):
        record.level = self.level
        super().handle(record)


class AntiCRLFLogRecord(logging.LogRecord):
    """
    记录日志时，转义CRLF
    """
    password_name_tuple = ('password', 'check_password', 'sql_pwd')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = logging.DEBUG

    def getMessage(self):
        """
        重写getMessage方法
        1. 过滤敏感字段打印
        2. 限制打印长度2048个字符
        3. 转义CRLF, '\n' --> '\\n', '\r' --> '\\r'
        """
        message = str(self.msg)
        if self.args:
            try:
                message = message % self.args
            except TypeError:
                message = ''
        for password_name in self.password_name_tuple:
            if password_name in message:
                return ''
        if len(message) > MAX_LOG_MSG_SIZE:
            message = message[:MAX_LOG_MSG_SIZE] + "..."
        message = message.replace('\n', '\\n').replace('\r', '\\r').replace('\f', '\\f').replace('\b', '\\b') \
            .replace('\u007F', '\\u007F').replace('\v', '\\v').replace('\t', '\\t')
        return message


def get_logger():
    logging.setLogRecordFactory(AntiCRLFLogRecord)
    logger = LeveledLogger("msprechecker", logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = ANSIColoredFormatter("%(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


LOGGER = get_logger()

LOG_LEVELS = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.CRITICAL,
}
