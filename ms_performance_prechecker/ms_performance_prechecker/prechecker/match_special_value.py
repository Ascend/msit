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
import json
import re
from ms_performance_prechecker.prechecker.utils import get_dict_value_by_pos, deep_compare_dict, str_to_digit, logger

CALCULATING_OPS = ["+", "-", "*", "/", "//"]
COMPARING_OPS = ['>=', '<=', '!=', '=', '>', '<']

def parse_calculation_expression(input_value, expr, config):
    """
    处理含复合变量名的计算表达式（如 "prefix:dp + ns:x/y"）：
    1. 提取变量名（支持包含冒号的复合键）
    3. 安全计算并比较
    """
    # 步骤1：提取变量名（匹配包含冒号的键名）
    variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_:]*\b', expr)
    #breakpoint()
    # 步骤2：替换变量值为实际值
    for var in variables:
        var_value = get_dict_value_by_pos(config, var)
        if var_value is None:
            return False
        expr = expr.replace(var, str(var_value))
    
    # 步骤3：计算并比较
    logger.debug(f"parse_calculation_expression expr={expr}, input_value={input_value}")
    try:
        calculated_value = eval(expr)
        return input_value == calculated_value
    except:
        return False


def parse_comparison_expression(input_value, condition, config):
    operator, condition_key_or_value = _parse_condition(condition)
    logger.debug(f"parse_comparison_expression operator={operator}, condition_key_or_value={condition_key_or_value}")

    # 判断条件值是数字还是键路径
    if isinstance(condition_key_or_value, (int, float)):
        condition_val = condition_key_or_value
    else:
        # 从配置中获取值
        condition_val = get_dict_value_by_pos(config, condition_key_or_value)

    return _apply_operator(operator, input_value, condition_val)


def _parse_condition(condition_str):
    """解析操作符"""
    # 强制检查操作符存在性
    for op in COMPARING_OPS:
        if condition_str.startswith(op):
            remaining = condition_str[len(op):].lstrip(':').strip()
            # 尝试解析剩余部分为数字
            num_val = str_to_digit(remaining)
            logger.debug(f"_parse_condition remaining={remaining}, num_val={num_val}")
            if num_val is not None:
                return op, num_val  # 返回操作符和数字
            else:
                return op, remaining  # 返回操作符和键路径

    # 如果未找到操作符，直接报错（原设计不允许无操作符）
    raise ValueError(f"Invalid condition format: {condition_str}")


def _apply_operator(operator, input_val, condition_val):
    """处理操作符逻辑"""
    # 等值判断（深度结构比较）
    logger.debug(f"_apply_operator operator={operator}, input_val={input_val}, condition_val={condition_val}")
    if operator in ('=', None):
        return not deep_compare_dict(
            dicts=[input_val, condition_val],
            names=["input", "condition"],
            parent_key="",
            skip_keys=None
        )

    # 不等判断
    if operator == '!=':
        return deep_compare_dict(
            dicts=[input_val, condition_val],
            names=["input", "condition"],
            parent_key="",
            skip_keys=None
        )

    # 数值比较操作符处理
    if operator in ('>', '<', '>=', '<='):
        # 如果条件值是字符串形式的数字，转换为数值
        if isinstance(condition_val, str):
            condition_val = str_to_digit(condition_val)

        # 类型校验
        if not _is_numeric(input_val, condition_val):
            return False

        # 执行比较
        return {
            '>': input_val > condition_val,
            '<': input_val < condition_val,
            '>=': input_val >= condition_val,
            '<=': input_val <= condition_val
        }[operator]

    raise ValueError(f"无效操作符: {operator}")


def _is_numeric(*values):
    """数值类型校验"""
    return all(isinstance(v, (int, float)) for v in values)


def to_json_object(value):
    try:
        json_object = json.loads(value.replace("'", '"'))  # 兼容单引号
    except:
        json_object = None
    return json_object


def compare_dicts(dict1: dict, dict2: dict, skip_keys: list = None) -> bool:
    """
    深度比较两个嵌套字典是否完全一致

    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :param skip_keys: 需要跳过的键路径列表 (例如 ["root.key1", "root.key2[0]"])
    :return: True表示完全一致，False表示存在差异
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both inputs must be dictionaries")

    # 调用deep_compare_dict函数 -> 返回True表示有差异
    has_diff = deep_compare_dict(
        dicts=[dict1, dict2],
        names=["dict1", "dict2"],
        parent_key="root",
        skip_keys=skip_keys
    )

    #  返回比较结果的取反值，此返回值True代表无差异
    return not has_diff


def parse_nested_dict_condition(input_value, condition, config):
    """
    处理嵌套字典字符串（如 '{"xxx":{"x1":2, "y1":3}}'）：
    1. 解析字符串为字典
    2. 从 config 中获取外层键对应的内层键
    3. 比较 input_value 和内层键对应的值
    """
    condition_dict = to_json_object(condition)
    if not condition_dict:
        return False
    
    outer_key = next(iter(condition_dict))  # 获取最外层的 key（如 "xxx"）
    inner_dict = condition_dict.get(outer_key)   # 获取嵌套字典（如 {"x1":2, "y1":3}）
        
    # 从 config 中读取实际使用的 key（如 config["xxx"]="y1"）
    inner_key = get_dict_value_by_pos(config, outer_key)

    ## 必然匹配吗？
    if inner_key not in inner_dict:
        return False
        
    target_value = inner_dict.get(inner_key)  # 获取目标值（如 3）
    return input_value == target_value


def is_value_met_special_suggestions(input_value, condition, config):
    if isinstance(input_value, dict):
        condition_dict = to_json_object(condition)
        return compare_dicts(input_value, condition_dict or {})

    logger.debug(f"is_value_met_special_suggestions condition={condition}")
    if isinstance(condition, str):
        # 处理多条件（用分号分隔）
        if ';' in condition:
            sub_conditions = [ii.strip() for ii in condition.split(';') if ii.strip()]
            return all(is_value_met_special_suggestions(input_value, sub_cond, config) for sub_cond in sub_conditions)
        
        # 情况1：字符串形式的列表（如 "[2, 4, 8]"）
        if condition.startswith('[') and condition.endswith(']'):
            allowed_values = to_json_object(condition)
            logger.debug(f"is_value_met_special_suggestions input_value={input_value} allowed_values={allowed_values}")
            return input_value in allowed_values if allowed_values else False
        
        # 情况4：嵌套字典字符串（如 '{"xxx":{"x1":2, "y1":3}}'）
        if condition.startswith('{') and condition.endswith('}'):
            return parse_nested_dict_condition(input_value, condition, config)
        
        # 情况3：计算表达式（如 "16 // dp"）
        if any(op in condition for op in CALCULATING_OPS):
            return parse_calculation_expression(input_value, condition, config)
        
        # 情况2：比较运算（如 ">=2", "!=2"）
        if any(condition.startswith(op) for op in COMPARING_OPS):
            return parse_comparison_expression(input_value, condition, config)
            
    
    # 默认返回 False（不支持的 condition 类型）
    return False