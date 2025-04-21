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
from ms_performance_prechecker.prechecker.register import register_checker, cached, PrecheckerBase
from ms_performance_prechecker.prechecker.register import show_check_result, record, CONTENT_PARTS, CheckResult
from ms_performance_prechecker.prechecker.utils import logger, get_version_info, get_npu_info, get_global_env_info
from ms_performance_prechecker.prechecker.suggestions import GLOBAL_DEFAULT_CONFIG, DOMAIN, CONFIG
from ms_performance_prechecker.prechecker.suggestions import update_to_default_suggestions, suggestion_rule_checker


def save_env_contents(fix_pair, save_path):
    save_path = os.path.realpath(save_path)

    indent = " " * 4
    with open(save_path, "w") as ff:
        ff.write("ENABLE=${1-1}\n")
        ff.write('echo "ENABLE=$ENABLE"\n\n')
        ff.write('if [ "$ENABLE" = "1" ]; then\n')
        ff.write(indent + f"\n{indent}".join((x[0] for x in fix_pair)) + "\n")
        ff.write("else\n")
        ff.write(indent + f"\n{indent}".join((x[1] for x in fix_pair)) + "\n")
        ff.write("fi\n")
    return save_path



class EnvChecker(PrecheckerBase):
    __checker_name__ = "Env"

    def collect_env(self, additional_checks=None, **kwargs):
        env_vars = os.environ

        # Using names only for collecting
        update_to_default_suggestions(DOMAIN.environment_variables, additional_checks)
        ret_envs = {}
        for suggestion in GLOBAL_DEFAULT_CONFIG.get(DOMAIN.environment_variables, []):
            env_name = suggestion.get(CONFIG.name)
            ret_envs.update({env_name: env_vars.get(env_name)})
        ret_envs.update(get_global_env_info())
        return ret_envs

    @staticmethod
    def action(env_key, env_value):
        return f"export {env_key}={env_value}" if env_value else f"unset {env_key}"

    def do_precheck(self, envs, additional_checks=None, env_save_path=None, mindie_service_path=None, **kwargs):
        if not envs:
            return
        # Should do nothing after called in collect_env
        update_to_default_suggestions(DOMAIN.environment_variables, additional_checks)

        fix_pair = []
        version_info = get_version_info(mindie_service_path)
        version_info["NPU_TYPE"] = get_npu_info()
        # if version_info["NPU_TYPE"] not in ["d802", "d803"]:
            # return
        for suggestion_rule in GLOBAL_DEFAULT_CONFIG.get(DOMAIN.environment_variables, []):
            result, suggestion_value, current_value = suggestion_rule_checker(
                envs, suggestion_rule, version_info, domain=DOMAIN.environment_variables, action_func=self.action
            )

            if result == CheckResult.ERROR:
                env_key = suggestion_rule.get(CONFIG.name)
                env_cmd = self.action(env_key, suggestion_value)
                undo_env_cmd = self.action(env_key, current_value)
                fix_pair.append((env_cmd, undo_env_cmd))

        if not env_save_path:
            show_check_result("env", "ENV FILE", CheckResult.UNFINISH, reason="save_env setting to None/Empty")
            return

        if len(fix_pair) == 0:
            show_check_result("env", "ENV FILE", CheckResult.VIP, action=f"None env related needs to save")
            return

        save_path = save_env_contents(fix_pair, env_save_path)
        show_check_result("env", "", CheckResult.VIP,action=f"使能环境变量配置：source {save_path}")
        show_check_result("env", "", CheckResult.VIP, action=f"恢复环境变量配置：source {save_path} 0")


env_checker = EnvChecker()
