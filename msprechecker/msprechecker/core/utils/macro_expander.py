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

import re
from typing import Any, Dict


class MacroExpander:
    DOT = "."
    VAR_REGEX = re.compile(r"\$\{([\w_.]+?)\}")
    VERSION_REGEX = re.compile(r"Version\{([\w_.]+?)\}")
    FILE_PERM_REGEX = re.compile(r"FilePerm\{([\w_.]+?)\}")


    def __init__(self, config_source: Dict[str, Any], context_path: str) -> None:
        self.config_source = config_source
        self.context_path = context_path

    @staticmethod
    def _expand_version(match_object: re.Match):
        version_str = match_object.group(1)

        return f"Version({version_str})"

    @staticmethod
    def _expand_file_perm(match_object: re.Match):
        perm_bit = match_object.group(1)

        return f"FilePerm({perm_bit})"

    def expand(self, expr: Any):
        if isinstance(expr, dict):
            return {k: self.expand(v) for k, v in expr.items()}
        elif isinstance(expr, list):
            return [self.expand(item) for item in expr]
        elif isinstance(expr, str):
            s = self.VAR_REGEX.sub(self._expand_var, expr)
            s = self.VERSION_REGEX.sub(self._expand_version, s)
            s = self.FILE_PERM_REGEX.sub(self._expand_file_perm, s)
            return s
        else:
            return expr

    def _expand_var(self, match_object: re.Match):
        path = match_object.group(1)
        if not path or path.replace(self.DOT, "") == "":
            raise ValueError(
                f"${{}} expression must not be empty or only dots. Got {path!r}"
            )

        full_path = self._build_full_path(path)
        if full_path not in self.config_source:
            raise ValueError(
                f"Path not found: {full_path!r}"
            )

        val = self.config_source[full_path]

        return f"{type(val).__name__}({val})" if not isinstance(val, str) else val

    def _build_full_path(self, path: str) -> str:
        if path.startswith(self.DOT):
            base_parts = self.context_path.split(self.DOT) if self.context_path else []
            i = 0
            while i < len(path) and path[i] == self.DOT:
                if base_parts:
                    base_parts.pop()
                else:
                    raise ValueError(
                        f"Relative path {path!r} is not valid from {self.context_path!r}."
                    )
                i += 1
            remaining = path[i:]
            if remaining:
                base_parts.append(remaining)
            return self.DOT.join(base_parts)
        else:
            return path
