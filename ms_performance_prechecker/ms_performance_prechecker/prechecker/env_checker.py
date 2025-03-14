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
from ms_performance_prechecker.prechecker.register import register_checker, cached
from ms_performance_prechecker.prechecker.register import check_result, record, CONTENT_PARTS, CheckResult
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, logger, SUGGESTION_TYPES, get_version_info
from ms_performance_prechecker.prechecker.env_suggestion import ENV_SUGGESTIONS


def save_env_contents(fix_pair, save_path):
    save_path = os.path.realpath(save_path)

    with open(save_path, "w") as ff:
        ff.write("ENABLE=${1-1}\n")
        ff.write('echo "ENABLE=$ENABLE"\n\n')
        ff.write('if [ "$ENABLE" = "1" ]; then\n    ')
        ff.write("\n    ".join((x[0] for x in fix_pair)) + "\n")
        ff.write('else\n    ')
        ff.write("\n    ".join((x[1] for x in fix_pair)) + "\n")
        ff.write('fi\n')
    return save_path


def version_in_list(version_info, version_list):
    for version_item, version_value_list in version_list.items():
        now_version = version_info.get(version_item, None)
        if now_version not in version_value_list:
            return False

    return True

def env_rule_checker(env_rule, version_info):
    if not env_rule:
        return (CheckResult.OK, None, None)
    suggestions = []
    
    env_item = env_rule.get("ENV")
    if "SUGGESTIONS" in env_rule:
        suggestions = env_rule["SUGGESTIONS"]
    if "SUGGESTION_VALUE" in env_rule:
        suggestions.append(dict(VALUE=env_rule.get("SUGGESTION_VALUE", None), 
            SUGGESTION=dict(REASON=env_rule.get("REASON", ""))))
    
    suggest_value_list = [] # (value, reason) 优先级从前到后，在前面的优先级高
    not_suggest_value_dict = {} # value： reason

    for suggestion in suggestions:
        suggestion_value = suggestion.get("VALUE", None)
        if not isinstance(suggestion_value, list):
            suggestion_value = [suggestion_value]
        suggestion_reason = ""
        suggestion_version_list = None
        not_suggestion_reason = ""
        not_suggestion_version_list = None
        if "SUGGESTION" in suggestion:
            suggestion_version_list = suggestion.get("SUGGESTION").get("VERSION_LIST", suggestion_version_list)
            suggestion_reason = suggestion.get("SUGGESTION").get("REASON", suggestion_reason)
            
            if suggestion_version_list is None or version_in_list(version_info, suggestion_version_list):
                suggest_value_list.extend(((x, suggestion_reason) for x in suggestion_value))
        if "NOT_SUGGESTION" in suggestion:
            not_suggestion_version_list = suggestion.get("NOT_SUGGESTION").get("VERSION_LIST",
                not_suggestion_version_list)
            not_suggestion_reason = suggestion.get("NOT_SUGGESTION").get("REASON", not_suggestion_reason)
            if  not_suggestion_version_list is None or version_in_list(version_info, not_suggestion_version_list):
                not_suggest_value_dict.update({x: not_suggestion_reason for x in suggestion_value})

    env_value = os.getenv(env_item, None)
    if env_value in not_suggest_value_dict:
        # 最后加一个建议，如果前面没有命中，就直接让用户unset 当前环境变量
        # 如果不建议配置为空，那么一定要有一个前置建议能命中，否则就是配置问题，代码中不做保证
        suggest_value_list.append((None, not_suggest_value_dict[env_value]))

    for suggestion_value, reason in suggest_value_list:
        if suggestion_value not in not_suggest_value_dict:
            env_cmd = f"export {env_item}={suggestion_value}" if suggestion_value else f"unset {env_item}"
            undo_env_cmd = f"export {env_item}={env_value}" if env_value else f"unset {env_item}"
            check_result("env", env_item, CheckResult.ERROR,
                action=env_cmd,
                reason=reason,
            )
            return (CheckResult.ERROR, env_cmd, undo_env_cmd)

    check_result("env", env_item, CheckResult.OK)
    return (CheckResult.OK, None, None)
        

@register_checker()
def simple_env_checker(env_save_path, mindie_service_path, **kwargs):
    env = kwargs.get("env", {})
    fix_pair = []
    version_info = get_version_info(mindie_service_path)
    for item in ENV_SUGGESTIONS:
        result, env_cmd, undo_env_cmd = env_rule_checker(item, version_info)

        if result == CheckResult.ERROR:
            fix_pair.append((env_cmd, undo_env_cmd))

    if not env_save_path:
        check_result("env", "SAVE ENV FILE", CheckResult.UNFINISH,
            reason="save_env setting to None/Empty",
        )
        return
    
    if len(fix_pair) == 0:
        check_result("env", "SAVE ENV FILE", CheckResult.VIP,
            action=f"None env related needs to save",
        )
        return 

    save_path = save_env_contents(fix_pair, env_save_path)
    
    check_result("env", "", CheckResult.VIP,
        action=f"环境相关改动使能：source {save_path};",
    )
    check_result("env", "", CheckResult.VIP,
        action=f"使能后恢复：source {save_path} 0",
    )


