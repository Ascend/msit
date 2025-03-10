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
from ms_performance_prechecker.prechecker.register import register_checker, cached, answer
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, SUGGESTION_TYPES


@register_checker()
def simple_env_checker(*_):
    suggestion_file = os.path.join(os.path.dirname(__file__), "env_suggestion.yml")
    import yaml

    with open(suggestion_file, "r") as f:
        suggestion_content = yaml.safe_load(f)

    for item in suggestion_content.get("envs"):
        env_item = item.get("ENV")
        env_value = os.getenv(env_item, "")
        env_suggest_value = item.get("SUGGESTION_VALUE") or ""
        suggest_reason = item.get("REASON", "")
        allow_undefined = item.get("ALLOW_UNDEFINED", False)
        if allow_undefined and not env_value:
            continue
        if str(env_value) != str(env_suggest_value):
            print(env_item, ":", str(env_value), "->", str(env_suggest_value))
            answer(
                suggesion_type=SUGGESTION_TYPES.env,
                suggesion_item=env_item,
                action=f"export {env_item}={env_suggest_value}" if env_suggest_value else f"unset {env_item}",
                reason=suggest_reason,
            )
