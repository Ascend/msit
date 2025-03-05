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

from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, SUGGESTION_TYPES

# 创建一个全局的注册表，注册为分析函数
REGISTRY = {}

ANSWERS = {ii: {} for ii in SUGGESTION_TYPES}


def register_checker(analyze_name=None):
    def decorator(func):
        name = analyze_name if analyze_name is not None else func.__name__
        REGISTRY[name] = func
        return func

    return decorator


def cached():
    # 缓存函数结果，反正所有输入都是一样的
    cache = {}

    def decorator(func):
        name = func.__name__

        def wrapper(*args, **kwargs):
            if name in cache:
                return cache[name]
            result = func(*args, **kwargs)
            cache[name] = result
            return result

        return wrapper

    return decorator


def answer(suggesion_type=None, suggesion_item=None, action=None, reason=""):
    ANSWERS[suggesion_type].setdefault(suggesion_item, []).append((action, reason))


def print_answer():
    print("\n<answer>")
    for suggesion_type in SUGGESTION_TYPES:
        for name, items in ANSWERS.get(suggesion_type, dict()).items():
            for action, reason in items:
                print("[{}] {}\n[action] {}\n[reason] {}\n".format(suggesion_type, name, action, reason))

    print("</answer>")