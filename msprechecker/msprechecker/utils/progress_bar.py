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
import sys
import time


class ProcessBarStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + "\r")  # 用 \r 代替默认的 \n
            self.flush()
        except Exception:
            self.handleError(record)


class SimpleProgressBar:
    def __init__(self, iterable, desc=None, total=None):
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total if total is not None else len(iterable)
        self.current = 0
        self.start_time = time.time()
        self.logger = self._init_logger()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.logger.info("\n")

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)
        local_logger.propagate = False

        if not local_logger.handlers:
            handler = ProcessBarStreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            local_logger.addHandler(handler)

        return local_logger

    def update(self, n=1):
        self.current += n
        self._print_progress()

    def _print_progress(self):
        progress = self.current / self.total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        percent = progress * 100

        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            remaining_time = (elapsed_time / progress) * (1 - progress)
        else:
            remaining_time = 0

        trailing_space = " " * 4  # invisible but better for progress bar
        self.logger.info(
            f"\r{self.desc} |{bar}| {percent:.1f}% [{self.current}/{self.total}] "
            f"ETA: {remaining_time:.1f}s{trailing_space}"
        )
