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
import sys

class SimpleProgressBar:
    def __init__(self, iterable, desc=None, total=None):
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total if total is not None else len(iterable)
        self.current = 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, n=1):
        self.current += n
        self._print_progress()

    def _print_progress(self):
        progress = self.current / self.total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100
        sys.stdout.write(f'\r{self.desc} |{bar}| {percent:.1f}% [{self.current}/{self.total}]')
        sys.stdout.flush()

    def close(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
