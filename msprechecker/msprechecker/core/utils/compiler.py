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
import operator
from typing import Any
from ..utils.version import Version
from ..utils.perm import FilePerm


class Compiler:
    OPS = {
        '+': (1, operator.add),
        '-': (1, operator.sub),
        '*': (2, operator.mul),
        '/': (2, operator.truediv),
        '//': (2, operator.floordiv),
        '%': (2, operator.mod),
        '**': (3, operator.pow)
    }

    _TOKEN_REGEX = re.compile(
        r'(?P<FUNC>float|int|Version|FilePerm)\((?P<FUNC_ARG>[^()]*?)\)'    # Function calls
        r'|(?P<OP>\*\*|//|[+\-*/%()])'                                      # Operators3
        r'|(?P<NUMBER>\d+([.]\d*)?)'                                        # Integer or decimal number
        r'|(?P<SKIP>[ \t]+)'                                                # Whitespace
    )

    _FUNC_MAP = {
        'float': float,
        'int': int,
        'Version': Version,
        'FilePerm': FilePerm
    }

    @staticmethod
    def _split_tokens(expr: str):
        tokens = []
        has_number = False
        has_int_or_float = False
        has_version = False
        has_op = False
        for mo in Compiler._TOKEN_REGEX.finditer(expr):
            kind = mo.lastgroup
            value = mo.group()

            if kind == 'SKIP':
                continue
            elif kind == 'FUNC_ARG':
                func_name = mo.group('FUNC')
                func_arg = mo.group('FUNC_ARG')
                if func_name == 'Version':
                    has_version = True
                else:
                    has_int_or_float = True
                tokens.append((func_name.upper(), Compiler._FUNC_MAP[func_name](func_arg)))
            elif kind == 'NUMBER':
                has_number = True
                tokens.append(('NUMBER', value))
            elif kind == 'OP':
                has_op = True
                tokens.append(('OP', value))

        # Constraint checks
        version_conflict = has_version and (has_int_or_float or has_op or has_number)
        if version_conflict:
            raise SyntaxError("Expression cannot contain VERSION with INT/FLOAT/NUMBER/OP tokens.")

        return tokens

    @classmethod
    def compile(cls, expr: str):
        if isinstance(expr, list):
            return [cls.compile(ex) for ex in expr]
        elif isinstance(expr, dict):
            return {k: cls.compile(ex) for k, ex in expr.items()}
        elif isinstance(expr, str):
            return cls._compile(expr)
        else:
            return expr

    @classmethod
    def _handle_tokens(cls, tokens, output, stack):
        i = 0
        while i < len(tokens):
            typ, val = tokens[i]
            if typ in ('INT', 'FLOAT'):
                output.append(val)
            elif typ == "NUMBER":
                output.append(float(val) if '.' in val else int(val))
            elif typ == 'OP':
                if val == '(':
                    stack.append(val)
                elif val == ')':
                    while stack and stack[-1] != '(':
                        output.append(stack.pop())
                    if not stack:
                        raise ValueError("Mismatched parentheses in expression")
                    stack.pop()
                else:
                    op = val
                    # Handle multi-char ops
                    if op in ('*', '/') and i + 1 < len(tokens) and tokens[i + 1][1] in ('*', '/'):
                        op += tokens[i + 1][1]
                        i += 1
                    if op not in cls.OPS:
                        raise ValueError(f"Unsupported operator: {op}")
                    prec = cls.OPS[op][0]
                    while stack and stack[-1] in cls.OPS and cls.OPS[stack[-1]][0] >= prec:
                        output.append(stack.pop())
                    stack.append(op)
            i += 1

    @classmethod
    def _convert_tokens_to_rpn(cls, tokens):
        # Handle single token (NUMBER, INT, FLOAT, VERSION)
        if len(tokens) == 1:
            _, val = tokens[0]
            return [val]

        # Handle Version (should be single token)
        if any(t[0] == 'VERSION' for t in tokens):
            if len(tokens) != 1:
                raise SyntaxError("Version expressions must be a single token.")
            return [tokens[0][1]]

        # Handle expressions with INT/FLOAT and OPs
        output = []
        stack = []
        cls._handle_tokens(tokens, output, stack)

        while stack:
            if stack[-1] in ('(', ')'):
                raise ValueError("Mismatched parentheses in expression")
            output.append(stack.pop())

        return output

    @classmethod
    def _evaluate_rpn(cls, rpn):
        if len(rpn) == 1:
            return rpn[0]
        stack = []
        for token in rpn:
            if isinstance(token, (int, float, Version, FilePerm)):
                stack.append(token)
            elif token in cls.OPS:
                b = stack.pop()
                a = stack.pop()
                stack.append(cls.OPS[token][1](a, b))
            else:
                raise ValueError(f"Unknown token in RPN: {token}")
        if len(stack) != 1:
            raise ValueError("Invalid RPN evaluation")
        return stack[0]

    @classmethod
    def _compile(cls, expr: Any):
        tokens = cls._split_tokens(expr)
        if not tokens or all(t[0] == 'NUMBER' for t in tokens):
            return expr

        rpn = cls._convert_tokens_to_rpn(tokens)
        try:
            result = cls._evaluate_rpn(rpn)
        except Exception:
            return expr

        return result
