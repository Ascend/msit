# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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


import pandas as pd
from components.utils.check.rule import Rule


# 读取
def get_csv_to_df(file_path) -> any:
    '''返回pd类型数据'''
    if not Rule.input_file().check(file_path):
        logger.error("read csv file failed, please check %r", file_path)
        raise OSError
    return pd.read_csv(file_path)


# 返回在按指定列的最大值对 DataFrame 进行排序后从顶部开始，返回指定数量的行
def get_nlargest(df, n, columns, keep='last') -> any:
    return df.nlargest(n, columns, keep)


# 计算top-3的和:
def get_top3_sum(df, columns, keep='last') -> any:
    top3 = get_nlargest(df, 3, columns, keep)
    return top3[columns].sum()


# 获取前n行数据
def get_nhead(df, n) -> any:
    return df.head(n)


# 属性获取或设置指定位置的值。指定要返回的单元格的行（索引）和列（标签）
def get_value_from_str(df, row_str, col_str) -> any:
    return df.loc[row_str, col_str]


# 属性获取或设置指定位置的值。指定要返回的单元格的行（索引）和列（标签）
def get_value_from_index(df, row_num, col_num) -> any:
    return df.iat[row_num, col_num]


def get_label_and_content(df) -> any:
    label_list = []
    content_list = []
    for label, content in df.iteritems():
        label_list.append(label)
        content_list.append(content)
    return label_list, content_list


def get_index_and_row(df) -> any:
    index_list = []
    row_list = []
    for index, row in df.iterrows():
        index_list.append(index)
        row_list.append(row)
    return index_list, row_list


# 方法检查 DataFrame 是否包含指定的值。
def check_value(df, value) -> any:
    return df.isin(value)


# 获取条件下更大的数据
def get_bigger_value(df, label, value) -> any:
    return df.where(df[label] > value)


# 获取条件下更小的数据
def get_smaller_value(df, label, value) -> any:
    return df.where(df[label] < value)


# 返回满足col1列值条件的col2列最后一个元素值
def get_item_value(df, col1, col1_value, col2) -> any:
    return df.at[df.index[df[col1] == col1_value].values[-1], col2]


# 修改满足col1列值条件的col2列最后一个元素值
def set_item_value(df, col1, col1_value, col2, col2_value) -> None:
    df.at[df.index[df[col1] == col1_value].values[-1], col2] = col2_value


# 返回满足col1列值条件的col2列元素值列表
def get_col_value(df, col1, col1_value, col2) -> any:
    return df.loc[df[col1] == col1_value, col2].values


# 修改满足col1列值条件的col2列元素值
def set_col_value(df, col1, col1_value, col2, col2_value) -> None:
    df.loc[df[col1] == col1_value, col2] = col2_value


# 在末尾增加一行或多行记录
def add_row(df, rows) -> any:
    return df.append(pd.DataFrame(rows), ignore_index=True)


# 在满足col1列值条件的末尾插入行
def insert_row_1(df, col, col_value, rows) -> any:
    i = df.index[(df[col] == col_value)].values[-1]
    df_1 = df.loc[0:i, :].append(pd.DataFrame(rows), ignore_index=True)
    return df_1.append(df.loc[i + 1 :, :], ignore_index=True)


# 在满足col1、col2两列值条件的末尾插入行
def insert_row_2(df, col1, col1_value, col2, col2_value, rows) -> any:
    i = df.index[(df[col1] == col1_value) & (df[col2] == col2_value)].values[-1]
    df_1 = df.loc[0:i, :].append(pd.DataFrame(rows), ignore_index=True)
    return df_1.append(df.loc[i + 1 :, :], ignore_index=True)


# 删除指定某列值的行
def drop_row_1(df, col, col_value) -> None:
    df.drop(df.index[(df[col] == col_value)], inplace=True)


# 删除指定某两列值的行
def drop_row_2(df, col1, col1_value, col2, col2_value) -> None:
    df.drop(df.index[(df[col1] == col1_value) & (df[col2] == col2_value)], inplace=True)
