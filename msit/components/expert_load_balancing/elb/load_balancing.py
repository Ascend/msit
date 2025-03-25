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
import sys
import glob
import re

import json
import pandas as pd
import torch

from c2lb import lb_and_intra_layer_affinity_redundancy_deploy
from speculative_moe import speculative_moe_algo_multi_process
from components.utils.log import logger


PREFILL = "prefill"
DECODE = "decode"


def load_expert_popularity_csv(csv_file_path):
    """
    从指定目录加载 decode_info.csv 和 prefill_info.csv 文件，支持以下场景：
    - 仅存在其中一个文件。
    - 同时存在两个文件。
    
    参数:
    - csv_file_path (str): 包含 CSV 文件的目录路径。
    
    返回:
    - (decode_df, prefill_df): 返回元组。
        - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
        - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    def _load_single_file(file_name):
        file_path = os.path.join(csv_file_path, file_name)
        if not os.path.isfile(file_path):
            logger.warning(f"Warning: File not found: {file_path}")
            return None
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {file_name}. Shape: {df.shape}")
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"Error: File is empty: {file_path}")
        except pd.errors.ParserError:
            logger.error(f"Error: Failed to parse CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
        return None

    # 分别加载 decode 和 prefill 文件
    decode_df = _load_single_file("decode_info.csv")
    prefill_df = _load_single_file("prefill_info.csv")
    
    return decode_df, prefill_df


def get_csv_path(csv_file_path, file_name):
    """
    从指定目录得到decode_info.csv和prefill_info.csv文件路径。
    
    参数:
    - csv_file_path: 包含CSV文件的目录路径。
    
    返回:
    - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
    - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    file_path = os.path.join(csv_file_path, file_name)
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist.")

    return file_path


def merge_csv_columns(csv_path, pattern_prefix):
    """
    合并指定目录下以特定前缀开头的所有csv文件的列。

    参数:
    - csv_path: 文件所在的目录路径。
    - pattern_prefix: 文件名前缀。
    
    返回:
    - 生成的DataFrame对象。
    """
    file_pattern = os.path.join(csv_path, f"{pattern_prefix}_*.csv")
    
    def sort_filenames(filenames):
        return sorted(filenames, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    files = glob.glob(file_pattern)
    filtered_files = [f for f in files if re.match(rf".*{pattern_prefix}_(\d+)\.csv", f)]
    filtered_files = sort_filenames(filtered_files)

    logger.info(f"{pattern_prefix} file number is {len(filtered_files)}")
    if not filtered_files:
        logger.error(f"No files found matching the pattern {file_pattern}")
        return None
    
    # 初始化一个空的列表，用于存储列
    all_columns = []
    column_count = 0
    
    # 遍历每个文件，将其每列添加到all_columns列表中
    for _, file in enumerate(filtered_files):
        logger.info(f"Processing file: {file}")
        df = pd.read_csv(file, header=None)
        if df.empty:
            logger.warning(f"Warning: File {file} is empty.")
            continue
        
        for col in range(df.shape[1]):
            all_columns.append(df.iloc[:, col].rename(f'expert_{column_count}'))
            column_count += 1
    
    if not all_columns:
        logger.warning("No columns to concatenate.")
        return None
    final_df = pd.concat(all_columns, axis=1)
    return final_df


def save_matrix_to_json(output_path, file_name, deployment):
    if deployment.ndim != 3:
        raise ValueError(f"部署矩阵必须是三维数组，但当前维度为 {deployment.ndim}D\n")
    num_layers = deployment.shape[0]
    num_cards = deployment.shape[1]

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            device = {"device_id": j, "device_expert": deployment[i, j].tolist()}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}/{file_name}.json"

    # 保存为 JSON 文件
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"保存json文件 {deployment} 时出错: {e}")


def save_dataframes(prefill_final_df, decode_final_df, output_dir):
    """Save prefill and decode dataframes to CSV files if they are not empty."""
    if prefill_final_df is not None and not prefill_final_df.empty:
        prefill_final_df.to_csv(os.path.join(output_dir, 'prefill_info.csv'), index=False)
    if decode_final_df is not None and not decode_final_df.empty:
        decode_final_df.to_csv(os.path.join(output_dir, 'decode_info.csv'), index=False)


def get_csv_dimensions(input_csv_path):
    """
    读取CSV文件并返回其行数（第0维）和列数（第1维）。

    参数:
        input_csv_path (str): CSV文件路径。

    返回:
        tuple: (行数, 列数) 的元组，失败时返回 (None, None)。
    """
    dimensions = (None, None)
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            logger.warning(f"警告: 文件包含空数据 {input_csv_path}")
            return dimensions
        row_count = df.shape[0]  # 第0维（行数）58
        col_count = df.shape[1]  # 第1维（列数）256
        dimensions = (row_count, col_count)
        
        logger.info(f"成功读取: {input_csv_path}")
        logger.info(f"输入文件: 行数={row_count}, 列数={col_count}")
        
    except pd.errors.EmptyDataError:
        logger.error(f"错误: 文件无有效数据 {input_csv_path}")
    except pd.errors.ParserError:
        logger.error(f"错误: CSV格式解析失败 {input_csv_path}")
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
    
    return dimensions


def process_c2lb(args, output_dir):
    """Process C2LB algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    decode_file_name = "decode_global_deployment"
    prefill_file_name = "prefill_global_deployment" 
    
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            num_original_expert = decode_df.shape[1]  # 第1维（列数）
            decode_np = decode_df.to_numpy()
            lb_and_intra_layer_affinity_redundancy_deploy(
                decode_np, 
                args.num_redundancy_expert, 
                output_dir, 
                decode_file_name,
                args.num_npus, 
                num_original_expert, 
            )
            logger.info(f"C2LB processed decode data -> {decode_file_name}")
        except Exception as e:
            logger.error(f"Failed to process decode data: {str(e)}")

    if prefill_df is not None:
        try:
            num_original_expert = prefill_df.shape[1]  # 第1维（列数）
            prefill_np = prefill_df.to_numpy()
            lb_and_intra_layer_affinity_redundancy_deploy(
                prefill_np,  
                args.num_redundancy_expert, 
                output_dir, 
                prefill_file_name,
                args.num_npus, 
                num_original_expert,
            )
            logger.info(f"C2LB processed prefill data -> {prefill_file_name}")
        except Exception as e:
            logger.error(f"Failed to process prefill data: {str(e)}")
    
    # 如果两个文件都不存在
    if decode_df is None and prefill_df is None:
        logger.warning("Warning: No valid decode/prefill data found in", output_dir)
    

def process_speculative_moe(args, file_names, output_dir):
    """
    处理 Speculative MOE 算法，根据传入的 file_names 动态处理以下场景：
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。

    参数:
        args (Namespace): 包含算法参数的对象。
        file_names (list): 需要处理的 CSV 文件名列表（例如 ["decode_info.csv", "prefill_info.csv"]）。
        output_dir (str): 输出结果的目录路径。
    """
    # 定义输入文件与输出文件的映射关系（修正 preill 拼写错误）
    input_output_map = {
        "decode_info.csv": "decode_global_deployment.json",
        "prefill_info.csv": "prefill_global_deployment.json"
    }   

    for input_file in file_names:
        # 检查是否为支持的文件类型
        if input_file not in input_output_map:
            logger.warning(f"忽略不支持的文件 {input_file}")
            continue

        # 生成输入和输出路径
        input_csv_path = get_csv_path(output_dir, input_file)
        dimensions = get_csv_dimensions(input_csv_path)
        output_file_name = input_output_map[input_file]
        output_path = os.path.join(output_dir, output_file_name)
        num_layer = dimensions[0]
        num_original_expert = dimensions[1]
        # 处理当前文件
        try:
            speculative_moe_algo_multi_process(
                args.num_npus,
                args.num_nodes,
                num_layer,
                num_original_expert,
                args.num_redundancy_expert,
                input_csv_path,
                output_path
            )
            logger.info(f"Speculative MOE 处理完成: {input_file} -> {output_file_name}")
        except FileNotFoundError:
            logger.error(f"错误: 输入文件不存在 {input_csv_path}")
        except Exception as e:
            logger.error(f"处理 {input_file} 时发生异常: {str(e)}")


def check_file_type(folder_path):
    """
    Check the folder for files containing 'prefill' or 'decode' in their names.
    Returns a tuple of:
    - detected file types ('prefill', 'decode', or both).
    - number of 'prefill' files.
    - number of 'decode' files.
    """
    file_types = set()
    prefill_count = 0
    decode_count = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".csv"):
            if PREFILL in file_name.lower():
                file_types.add(PREFILL)
                prefill_count += 1
            if DECODE in file_name.lower():
                file_types.add(DECODE)
                decode_count += 1
    
    return file_types, prefill_count, decode_count


def load_balancing(args):
    """
    Main function to perform load balancing based on the specified algorithm.
    """
    # Check the file types in the input path
    file_types, prefill_count, decode_count = check_file_type(args.expert_popularity_csv_load_path)

    # Print the file types and their count
    logger.info(f"Detected file types: {', '.join(file_types)}")
    logger.info(f"Total 'prefill' files: {prefill_count}")
    logger.info(f"Total 'decode' files: {decode_count}")

    # Merge CSV columns for prefill and decode
    prefill_final_df = None
    decode_final_df = None
    if PREFILL in file_types:
        prefill_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, PREFILL)
    if DECODE in file_types:
        decode_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, DECODE)
    
     # Save merged dataframes to CSV files if they are not empty
    save_dataframes(prefill_final_df, decode_final_df, args.output_dir)

    # Define file names for further processing
    file_names = []
    if PREFILL in file_types:
        file_names.append("prefill_info.csv")
    if DECODE in file_types:
        file_names.append("decode_info.csv")

    # c2lb算法
    if args.algorithm == '0':
        process_c2lb(args, output_dir=args.output_dir)
    # process_speculative_moe算法
    else:
        process_speculative_moe(args, file_names=file_names, output_dir=args.output_dir)