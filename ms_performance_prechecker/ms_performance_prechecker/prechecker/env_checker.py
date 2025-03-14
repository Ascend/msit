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
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, logger, SUGGESTION_TYPES
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


def version_in_black_list(version_info, black_list):
    for black_version in black_list:
        if version_info.startswith(black_version):
            return True
    return False


@register_checker()
def simple_env_checker(env_save_path, **kwargs):
    env = kwargs.get("env", {})
    fix_pair = []
    for item in ENV_SUGGESTIONS:
        env_item = item.get("ENV")
        env_value = os.getenv(env_item, "")
        env_suggest_value = item.get("SUGGESTION_VALUE", "")
        suggest_reason = item.get("REASON", "")
        version_black_list = item.get("VERSION_BLACK_LIST", [])
        allow_undefined = item.get("ALLOW_UNDEFINED", False)
        if allow_undefined and not env_value:
            check_result("env", env_item, CheckResult.OK)
            continue
        if str(env_value).lower() == str(env_suggest_value).lower():
            check_result("env", env_item, CheckResult.OK)
            continue

        logger.debug(f"{env_item}: {env_value} -> {env_suggest_value}")
        env_cmd = f"export {env_item}={env_suggest_value}" if env_suggest_value else f"unset {env_item}"

        
        check_result("env", env_item, CheckResult.ERROR,
            action=env_cmd,
            reason=suggest_reason,
        )

        pre_env = f"export {env_item}={env_value}" if env_value else f"unset {env_item}"

   
        fix_pair.append((env_cmd, pre_env))

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


