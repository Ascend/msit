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
import typing
from collections import namedtuple

import yaml

"""
Yaml 配置模板：
# 配置建议模板
version: 1.0
description: MindIE server config template

# 环境变量建议配置
environment_variables:
  # 简易配置
  - name: "HOST"
    value: "localhost"
    reason: "配置为本机地址"

  # 扩展配置
  - name: "PORT"
    suggestions:
      - value: "AIV"
        suggested:
          condition: {"Ascend-mindie": [">2.1"]}
          reason: "该版本使用该配置"
        not_suggested:
          condition: {"Ascend-mindie": ["2.0.T3", "2.0.T6"]}
          reason: "早期版本不建议指定"
      - value: "another_value"
        suggested:
          condition: {"Ascend-mindie": [">2.2"]}
          reason: "该版本使用该配置"

# config.json 建议配置
mindie_config_json:
  # 简易配置
  - path: "connection:host"
    value: "127.0.0.1"
    reason: "配置为本机地址"

  # 扩展配置
  - name: "features:experimental:enabled"
    suggestions:
      - value: false
        suggested:
          condition: {"mindie_version": [">2.1"]}
          reason: "实验性功能不应在生产环境启用"
        not_suggested:
          condition: {"mindie_version": ["<2.1"]}
          reason: "早期版本不建议指定"
"""

_DOMAIN = ["environment_variables", "mindie_config", "ranktable", "model_config", "user_config"]
DOMAIN = namedtuple("DOMAIN", _DOMAIN)(*_DOMAIN)
_CONFIG = ["name", "value", "reason", "suggestions", "condition", "suggested", "not_suggested"]
CONFIG = namedtuple("CONFIG", _CONFIG)(*_CONFIG)
NOT_EMPTY_VALUE = "非空值"

def get_default_suggestions():
    suggestion_file = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    with open(suggestion_file, "r") as ff:
        suggestion_content = yaml.safe_load(ff)
    return suggestion_content


def update_to_default_suggestions(domain, additional_checks_yaml):
    if not additional_checks_yaml or domain not in additional_checks_yaml:
        return
    sub_config = GLOBAL_DEFAULT_CONFIG.get(domain, [])
    suggestions_dict = {ii[CONFIG.name]: ii for ii in sub_config}
    for ii in additional_checks_yaml.pop(domain):  # pop out for only apply once
        cur_key = ii.get(CONFIG.name, None)
        if cur_key in suggestions_dict:
            suggestions_dict[cur_key].clear()
            suggestions_dict[cur_key].update(ii)
        else:
            sub_config.append(ii)
            suggestions_dict[cur_key] = ii


def is_condition_met(env_info, suggestion_condition):
    for condition_item, condition_value_list in suggestion_condition.items():
        cur = env_info.get(condition_item, None)
        if cur not in condition_value_list:
            return False
    return True


def convert_value_type(value, domain):
    # For environment_variables, all value is string
    return value if value is None or domain != DOMAIN.environment_variables else str(value)


def is_value_met_special_suggestions(current_value, condition, current_configs):
    return False  # [TODO], adding special checking rules


def is_value_met_suggestions(current_value, suggested_values, current_configs):
    if not suggested_values:
        return current_value is not None  # suggested_values is empty, check if current_value not None
    normal_value_suggestions, special_value_suggestions = [], []
    for ii in suggested_values:
        if isinstance(ii, str) and ii.startswith("="):
            special_value_suggestions.append(ii)
        else:
            normal_value_suggestions.append(ii)
    if isinstance(current_value, typing.Hashable) and current_value in normal_value_suggestions:
        return True
    for condition in special_value_suggestions:
        condition = condition[1:].strip()  # get rid of starting =
        if is_value_met_special_suggestions(current_value, condition, current_configs):
            return True
    return False


def suggestion_rule_checker(current_configs, suggestion_rule, env_info, domain, action_func=None):
    from ms_performance_prechecker.prechecker.register import show_check_result, CheckResult
    from ms_performance_prechecker.prechecker.utils import get_dict_value_by_pos, logger

    if not suggestion_rule:
        return (CheckResult.OK, None, None)
    suggestions = []

    check_item = suggestion_rule.get(CONFIG.name)
    if CONFIG.suggestions in suggestion_rule:
        suggestions = suggestion_rule[CONFIG.suggestions]
    else:
        cur = {CONFIG.value: suggestion_rule[CONFIG.value]} if CONFIG.value in suggestion_rule else {}
        cur.update({CONFIG.suggested: {CONFIG.reason: suggestion_rule.get(CONFIG.reason, "")}})
        suggestions.append(cur)
    logger.debug(f"suggestion_rule_checker: suggestions = {suggestions}")

    suggest_value_list = []  # (value, reason) 优先级从前到后，在前面的优先级高
    not_suggest_value_dict = {}  # value： reason

    for suggestion in suggestions:
        if CONFIG.value in suggestion:
            suggestion_value = suggestion.get(CONFIG.value, None)
            if not isinstance(suggestion_value, list):
                suggestion_value = [suggestion_value]
            value_list = [convert_value_type(ii, domain) for ii in suggestion_value]
        else:
            suggestion_value, value_list = [], []
        suggestion_reason = ""
        suggestion_condition = None
        not_suggestion_reason = ""
        not_suggestion_version_list = None
        if CONFIG.suggested in suggestion:
            cur_suggested = suggestion.get(CONFIG.suggested, {})
            suggestion_condition = cur_suggested.get(CONFIG.condition, suggestion_condition)
            suggestion_reason = cur_suggested.get(CONFIG.reason, suggestion_reason)

            if suggestion_condition is None or is_condition_met(env_info, suggestion_condition):
                suggest_value_list.append((value_list, suggestion_reason))
        if CONFIG.not_suggested in suggestion:
            cur_not_suggested = suggestion.get(CONFIG.not_suggested, {})
            not_suggestion_version_list = cur_not_suggested.get(CONFIG.condition, not_suggestion_version_list)
            not_suggestion_reason = cur_not_suggested.get(CONFIG.reason, not_suggestion_reason)
            if not_suggestion_version_list is None or is_condition_met(env_info, not_suggestion_version_list):
                not_suggest_value_dict.update({x: not_suggestion_reason for x in suggestion_value})

    current_value = get_dict_value_by_pos(current_configs, check_item)
    if isinstance(current_value, typing.Hashable) and current_value in not_suggest_value_dict:
        # 最后加一个建议，如果前面没有命中，就直接让用户unset 当前环境变量
        # 如果不建议配置为空，那么一定要有一个前置建议能命中，否则就是配置问题，代码中不做保证
        suggest_value_list.append(([None], not_suggest_value_dict[current_value]))

    for value_list, reason in suggest_value_list:
        suggested_values = [x for x in value_list if x not in not_suggest_value_dict]
        logger.debug(
            f"value_list={value_list}, suggested_values={suggested_values}, current_value={current_value}"
        )
        if not is_value_met_suggestions(current_value, suggested_values, current_configs):
            suggestion_value = suggested_values[0] if len(suggested_values) > 0 else NOT_EMPTY_VALUE
            show_check_result(
                domain,
                check_item,
                CheckResult.ERROR,
                action=action_func and action_func(check_item, suggestion_value),
                reason=reason,
            )
            return (CheckResult.ERROR, suggestion_value, current_value)

    show_check_result(domain, check_item, CheckResult.OK)
    return (CheckResult.OK, None, None)


GLOBAL_DEFAULT_CONFIG = get_default_suggestions()
