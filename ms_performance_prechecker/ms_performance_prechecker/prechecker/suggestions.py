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
import yaml
from collections import namedtuple

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

"""
配置指南：
ENV_suggestions 配置项说明：
  * ENV: 环境变量名
  * suggestions: 建议列表
      - VALUE: 环境变量值（如果不建议配置，可以配置为None，表示不建议配置该环境变量）
      - SUGGESTION: 建议配置该值
          + VERSION_LIST: 哪些版本建议配置，不配置表示所有版本适用
          + REASON: 建议值对应的原因
      - NOT_SUGGESTION: 不建议配置该值（优先级高）
          + VERSION_LIST: 哪些版本不建议配置，不配置表示所有版本适用
          + REASON: 不建议的原因

简化配置：
  ENV: 环境变量名
  SUGGESTION_VALUE: 建议值
  REASON: 建议原因


建议样例1： 环境变量 ENV_SUGGEST_DEMO, 一般情况下建议配置为 VALUE1
建议配置1：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "SUGGESTION_VALUE": "VALUE1",
    "REASON": "建议配置为VALUE1",
  }
建议样例2： 背景同样例1，但是在版本 1.0.0 版本不建议配置为 VALUE1（建议不配置）
建议配置2：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "suggestions": [
      {
        "VALUE": "VALUE1",
        "SUGGESTION": {
          "REASON": "建议配置为VALUE1",
        },
        "NOT_SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.0"]},
          "REASON": "不建议配置为VALUE1",
        }
      }
    ]
  }

建议样例3： 背景同样例2，但是在版本 1.0.2 版本建议配置为 VALUE2
建议配置3：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "suggestions": [
      {
        "VALUE": "VALUE1",
        "SUGGESTION": {
          "REASON": "建议配置为VALUE1",
        },
        "NOT_SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.0"]},
          "REASON": "不建议配置为VALUE1",
        }
      },
      {
        "VALUE": "VALUE2",
        "SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.2"]},
          "REASON": "建议配置为 VALUE2",
        },
    ]
  }

"""

_DOMAIN = ["environment_variables", "mindie_config", "ranktable", "model_config"]
DOMAIN = namedtuple("DOMAIN", _DOMAIN)(*_DOMAIN)
_CONFIG = ["name", "value", "reason", "suggestions", "condition", "suggested", "not_suggested"]
CONFIG = namedtuple("CONFIG", _CONFIG)(*_CONFIG)


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
        if cur_key in suggestions_dict:
            suggestions_dict[cur_key].clear()
            suggestions_dict[cur_key].update(cur)
        else:
            sub_config.append(cur)
            suggestions_dict[cur_key] = cur


def is_condition_met(env_info, suggestion_condition):
    for condition_item, condition_value_list in suggestion_condition.items():
        cur = env_info.get(condition_item, None)
        if cur not in condition_value_list:
            return False
    return True


def suggestion_rule_checker(current_configs, suggestion_rule, env_info, domain, action_func=None):
    from ms_performance_prechecker.prechecker.register import show_check_result, CheckResult
    from ms_performance_prechecker.prechecker.utils import get_dict_value_by_pos

    if not suggestion_rule:
        return (CheckResult.OK, None, None)
    suggestions = []

    check_item = suggestion_rule.get(CONFIG.name)
    if CONFIG.suggestions in suggestion_rule:
        suggestions = suggestion_rule[CONFIG.suggestions]
    if CONFIG.value in suggestion_rule:
        suggestions.append(
            {
                CONFIG.value: suggestion_rule.get(CONFIG.value, None),
                CONFIG.suggested: {CONFIG.reason: suggestion_rule.get(CONFIG.reason, "")},
            }
        )

    suggest_value_list = []  # (value, reason) 优先级从前到后，在前面的优先级高
    not_suggest_value_dict = {}  # value： reason

    for suggestion in suggestions:
        suggestion_value = suggestion.get(CONFIG.value, None)
        if not isinstance(suggestion_value, list):
            suggestion_value = [suggestion_value]
        value_list = [x if x is None else str(x) for x in suggestion_value]
        suggestion_reason = ""
        suggestion_condition = None
        not_suggestion_reason = ""
        not_suggestion_version_list = None
        if CONFIG.suggested in suggestion:
            cur_suggested = suggestion.get(CONFIG.suggested, {})
            suggestion_condition = cur_suggested.get(CONFIG.condition, suggestion_condition)
            suggestion_reason = cur_suggested.get(CONFIG.reason, suggestion_reason)

            if suggestion_condition is None or is_condition_met(env_info, suggestion_condition):
                suggest_value_list.append((value_list, suggestion_reason))  # [TODO] apply specific ruls
        if CONFIG.not_suggested in suggestion:
            cur_not_suggested = suggestion.get(CONFIG.not_suggested, {})
            not_suggestion_version_list = cur_not_suggested.get(CONFIG.condition, not_suggestion_version_list)
            not_suggestion_reason = cur_not_suggested.get(CONFIG.reason, not_suggestion_reason)
            if not_suggestion_version_list is None or is_condition_met(env_info, not_suggestion_version_list):
                not_suggest_value_dict.update({x: not_suggestion_reason for x in suggestion_value})

    current_value = get_dict_value_by_pos(current_configs, check_item)
    if current_value in not_suggest_value_dict:
        # 最后加一个建议，如果前面没有命中，就直接让用户unset 当前环境变量
        # 如果不建议配置为空，那么一定要有一个前置建议能命中，否则就是配置问题，代码中不做保证
        suggest_value_list.append(([None], not_suggest_value_dict[current_value]))

    for value_list, reason in suggest_value_list:
        not_in_unsuggest_values = [x for x in value_list if x not in not_suggest_value_dict]
        if len(not_in_unsuggest_values) > 0 and current_value not in not_in_unsuggest_values:
            suggestion_value = not_in_unsuggest_values[0]
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
