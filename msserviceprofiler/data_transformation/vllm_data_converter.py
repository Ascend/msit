# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import json
import argparse
import os
import re
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import numpy as np
import pandas as pd

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('data_transformation.log')  # 文件输出
    ]
)
logger = logging.getLogger(__name__)

# 定义需要跳过的方法列表
SKIP_METHODS = [
    "vllm.entrypoints.openai.serving_completion.OpenAIServingCompletion.create_completion",
    "vllm.entrypoints.openai.serving_chat.OpenAIServingChat.create_chat_completion",
]

# 方法名映射规则：将原始方法名映射到域和简化名称
METHOD_MAPPING = {
    "vllm.entrypoints.openai.serving_chat.OpenAIServingChat.create_chat_completion": {
        "domain": "Request",
        "name": "createChatCompletion"
    },
    "vllm.entrypoints.openai.serving_completion.OpenAIServingCompletion.create_completion": {
        "domain": "Request",
        "name": "createCompletion"
    },
    "vllm.v1.engine.async_llm.AsyncLLM.generate": {
        "domain": "Request",
        "name": "asyncGenerate"
    },
    "vllm.v1.engine.async_llm.AsyncLLM.add_request": {
        "domain": "Request",
        "name": "httpReq"
    },
    "vllm.entrypoints.openai.serving_completion.OpenAIServingCompletion.request_output_to_completion_response": {
        "domain": "Request",
        "name": "httpRes"
    },
    "vllm.forward_context.set_forward_context": {
        "domain": "ModelExecute",
        "name": "forward"
    },
    "vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots": {
        "domain": "KVCache",
        "name": "Allocate"
    },
    "vllm.v1.core.kv_cache_manager.KVCacheManager.free": {
        "domain": "KVCache",
        "name": "Free"
    },
    "vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks": {
        "domain": "KVCache",
        "name": "GetCacheHitRate"
    },
    "vllm.v1.core.sched.scheduler.Scheduler.add_request": {
        "domain": "BatchSchedule",
        "name": "Enqueue"
    },
    "vllm.v1.core.sched.scheduler.Scheduler.schedule": {
        "domain": "BatchSchedule",
        "name": "BatchSchedule"
    },
    "vllm.v1.engine.core_client.AsyncMPClient._send_input_message": {
        "domain": "SendInputMessage",
        "name": "sendInputMessage"
    },
    "vllm.v1.engine.core.EngineCore.execute_model": {
        "domain": "ModelExecute",
        "name": "modelExec"
    },
    "vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model": {
        "domain": "NPUExecute",
        "name": "npuModelExec"
    },
    "vllm_ascend.worker.model_runner_v1.NPUModelRunner._process_reqs": {
        "domain": "NPUExecute",
        "name": "npuReqsProc"
    },
    "vllm_ascend.worker.model_runner_v1.NPUModelRunner._make_attention_mask": {
        "domain": "NPUExecute",
        "name": "npuAttnMaskBuilder"
    },
    "vllm.v1.engine.core_client.AsyncMPClient.get_output_async": {
        "domain": "Request",
        "name": "getOutputAsync"
    },
    "vllm.v1.engine.output_processor.RequestOutputCollector.get": {
        "domain": "Output",
        "name": "outputSync"
    },
    "vllm.v1.metrics.stats.IterationStats.update_from_output": {
        "domain": "Output",
        "name": "recordIterationMetrics"
    },
    "vllm.v1.engine.output_processor.OutputProcessor.process_outputs": {
        "domain": "Request",
        "name": "processOutputs"
    },
    "vllm.v1.core.sched.scheduler.Scheduler._free_request": {
        "domain": "BatchSchedule",
        "name": "FINISHED+"
    }
}

# 消息字段重命名规则
MESSAGE_KEY_MAPPING = {
    "request_ids": "rid",
    "num_free_blocks": "deviceBlock=",
    "request_id": "rid",
    "total_num_scheduled_tokens": "batch_size",
    "free_request_id": "rid",
    "prompt_len": "recvTokenSize="
}

# 需要从消息中删除的字段列表
FIELDS_TO_DELETE = [
    "num_scheduled_tokens",
    "reqs_scheduled_tokens"
]

# 需要提取的字段及其重命名映射
COLUMN_MAPPING = {
    "method_identifier": "method",  # 保留原始方法名用于映射
    "start_ms": "timestamp",
    "end_ms": "endTimestamp",
    "thread_id": "tid",
    "trace_data_json": "message"
}

# 用于跟踪请求ID迭代次数的计数器
batch_rid_counter = defaultdict(int)
model_exec_rid = defaultdict(int)


def extract_role_and_pid_from_filename(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """从文件名中提取角色和进程ID"""
    filename = os.path.basename(file_path)
    # 使用正则表达式匹配文件名格式：角色_PID.json
    match = re.search(r'^([a-zA-Z0-9]+)_(\d+)\.json$', filename)
    if match:
        try:
            role = match.group(1).lower()  # 角色名称转为小写
            pid = int(match.group(2))  # 进程ID转为整数
            return role, pid
        except ValueError:
            # 如果PID不是有效的整数，返回None
            pass
    return None, None


def find_json_files(directory: str) -> List[str]:
    """在指定目录中递归查找所有JSON文件"""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def apply_method_mapping(original_method: str) -> Tuple[str, str]:
    """应用方法名映射规则"""
    # 查找匹配的映射规则
    for pattern, mapping in METHOD_MAPPING.items():
        # 使用精确匹配而不是部分匹配
        if pattern == original_method:
            return mapping["domain"], mapping["name"]

    # 如果没有匹配规则，返回默认值
    return "UnknownDomain", original_method


def should_skip_method(method_identifier: str, parse_all: bool) -> bool:
    """
    判断是否应该跳过某个方法

    参数:
        method_identifier: 方法标识符
        parse_all: 是否解析所有方法

    返回:
        True表示跳过，False表示保留
    """
    if parse_all:
        return False  # 解析所有方法，不跳过任何方法
    return method_identifier in SKIP_METHODS


def transform_message(message_str: str, domain: str, name: str, parse_all: bool) -> str:
    """转换消息字段内容并添加域信息"""
    if not message_str or message_str == "{}":
        # 如果原始消息为空，创建一个包含域信息的新消息
        return json.dumps({"domain": domain, "name": name})

    try:
        # 解析JSON字符串
        message_data = json.loads(message_str)

        # 删除指定的字段
        for field_to_delete in FIELDS_TO_DELETE:
            if field_to_delete in message_data:
                del message_data[field_to_delete]
                logger.debug("Deleted field %s from message", field_to_delete)

        # 特殊处理：当不使用parse-all时，移除npuModelExec和npuReqsProc的request_ids字段
        should_del_request_ids = (not parse_all and
                                  (name == "npuModelExec" or name == "npuReqsProc") and
                                  "request_ids" in message_data)
        if should_del_request_ids:
            del message_data["request_ids"]

        # 特殊处理：processOutputs方法需要拆分为多个记录
        if name == "processOutputs" and "req_trace_info" in message_data:
            # 返回原始消息，拆分逻辑将在process_single_file中处理
            message_data["domain"] = domain
            message_data["name"] = name
            return json.dumps(message_data)

        # 添加域信息
        message_data["domain"] = domain
        # 添加名称信息
        message_data["name"] = name

        # 特殊处理：FINISHED+记录
        if name == "FINISHED+":
            message_data["name"] = "ReqState"  # 将名称改为ReqState
            message_data["FINISHED+"] = 1  # 添加FINISHED+字段

        # 应用键名映射
        transformed_data = {}
        for key, value in message_data.items():
            # 应用映射规则
            new_key = MESSAGE_KEY_MAPPING.get(key, key)
            transformed_data[new_key] = value

        def process_rid(rid_list, counter):
            """处理请求ID列表，为每个ID添加迭代计数"""
            trans_rid = []
            for rid in rid_list:
                iter_size = counter[rid]
                trans_rid.append({
                    "rid": rid,
                    "iter": iter_size
                })
                counter[rid] += 1
            return trans_rid

        # 特殊处理：BatchSchedule方法的请求ID
        if name == "BatchSchedule":
            if "rid" in transformed_data:
                rid_list = transformed_data["rid"]
                if isinstance(rid_list, list):
                    transformed_data["rid"] = process_rid(rid_list, batch_rid_counter)

        # 特殊处理：modelExec方法的请求ID
        if name == "modelExec":
            if "rid" in transformed_data:
                rid_list = transformed_data["rid"]
                if isinstance(rid_list, list):
                    transformed_data["rid"] = process_rid(rid_list, model_exec_rid)

        # 序列化回JSON字符串
        return json.dumps(transformed_data)
    except json.JSONDecodeError:
        # 如果解析失败，创建一个包含域信息的新消息
        return json.dumps({"domain": domain, "name": name, "message": message_str})


def process_npu_model_exec_groups(records: List[Dict], pid: int) -> List[Dict]:
    """
    基于npuModelExec进行分组处理

    每个组包含一个npuModelExec、一个npuReqsProc和一个forward记录
    """
    new_records = []
    npu_model_exec_records = []
    npu_reqs_proc_records = []
    forward_records = []

    # 第一步：分类记录
    for record in records:
        try:
            message_data = json.loads(record.get('message', '{}'))
            name = message_data.get('name', '')

            if name == 'npuModelExec':
                npu_model_exec_records.append(record)
            elif name == 'npuReqsProc':
                npu_reqs_proc_records.append(record)
            elif name == 'forward':
                forward_records.append(record)
            else:
                new_records.append(record)  # 其他记录直接保留
        except Exception as e:
            new_records.append(record)  # 解析失败的记录也直接保留

    # 第二步：按时间窗口分组（基于npuModelExec时间范围）
    for npu_model_exec in npu_model_exec_records:
        model_exec_start = npu_model_exec['timestamp']
        model_exec_end = npu_model_exec['endTimestamp']

        # 在npuModelExec时间范围内查找匹配的npuReqsProc和forward记录
        matching_npu_reqs_proc = None
        matching_forward = None

        # 查找npuReqsProc
        for npu_reqs_proc in npu_reqs_proc_records:
            if (npu_reqs_proc['timestamp'] >= model_exec_start and
                    npu_reqs_proc['endTimestamp'] <= model_exec_end):
                matching_npu_reqs_proc = npu_reqs_proc
                break

        # 查找forward
        for forward in forward_records:
            if (forward['timestamp'] >= model_exec_start and
                    forward['endTimestamp'] <= model_exec_end):
                matching_forward = forward
                break

        # 如果找到匹配的三个记录，处理它们
        if matching_npu_reqs_proc and matching_forward:
            # ① 添加预处理记录
            preprocess_record = create_preprocess_record(matching_npu_reqs_proc, matching_forward)
            new_records.append(preprocess_record)

            # ② 修改forward记录的endTimestamp
            modified_forward = matching_forward.copy()
            modified_forward['endTimestamp'] = matching_npu_reqs_proc['endTimestamp']
            new_records.append(modified_forward)

            # ③ 添加后处理记录
            postprocess_record = create_postprocess_record(matching_npu_reqs_proc, npu_model_exec)
            new_records.append(postprocess_record)

            # 添加原始的npuModelExec和npuReqsProc记录
            new_records.append(npu_model_exec)
            new_records.append(matching_npu_reqs_proc)

        else:
            # 如果没有找到匹配的记录，直接添加所有记录
            new_records.append(npu_model_exec)
            if matching_npu_reqs_proc:
                new_records.append(matching_npu_reqs_proc)
            if matching_forward:
                new_records.append(matching_forward)

    return new_records


def create_preprocess_record(npu_reqs_proc_record, forward_record):
    """创建预处理记录"""
    preprocess_record = npu_reqs_proc_record.copy()
    preprocess_record['message'] = json.dumps({
        "domain": "ModelExecute",
        "name": "preprocess"
    })
    preprocess_record['timestamp'] = npu_reqs_proc_record['timestamp']
    preprocess_record['endTimestamp'] = forward_record['timestamp'] - 1000  # forward开始时间
    return preprocess_record


def create_postprocess_record(npu_reqs_proc_record, npu_model_exec_record):
    """创建后处理记录"""
    postprocess_record = npu_model_exec_record.copy()
    postprocess_record['message'] = json.dumps({
        "domain": "ModelExecute",
        "name": "postprocess"
    })
    postprocess_record['timestamp'] = npu_reqs_proc_record['endTimestamp'] + 1000  # npuReqsProc结束时间
    postprocess_record['endTimestamp'] = npu_model_exec_record['endTimestamp']  # npuModelExec结束时间
    return postprocess_record


def process_single_file(file_path: str, role: str, pid: int, parse_all: bool = False) -> List[Dict]:
    """处理单个JSON文件并提取记录"""
    records = []
    skipped_count = 0
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw_data = json.loads(line)

                    # 检查是否应该跳过该方法
                    method_identifier = raw_data.get('method_identifier', '')
                    if should_skip_method(method_identifier, parse_all):
                        skipped_count += 1
                        continue

                    record = {}

                    # 添加角色和进程ID
                    record['role'] = role
                    record['pid'] = pid

                    # 提取并重命名字段
                    for orig_key, new_key in COLUMN_MAPPING.items():
                        if orig_key in raw_data:
                            # 特殊处理时间字段
                            if orig_key in ['start_ms', 'end_ms']:
                                # 乘以1000并保持为浮点数
                                record[new_key] = raw_data[orig_key] * 1000.0
                            else:
                                record[new_key] = raw_data[orig_key]

                    # 确保方法字段存在
                    if 'method' in record:
                        # 应用方法名映射
                        domain, new_method_name = apply_method_mapping(record['method'])

                        # 特殊处理：outputSync方法
                        if new_method_name == "outputSync":
                            # 将timestamp和endTimestamp都设置为end_ms值
                            record['timestamp'] = record.get('endTimestamp')

                        # Debug日志：打印方法名映射关系
                        logger.debug("Method mapping: %s -> %s -> %s", record['method'], new_method_name, domain)
                        logger.debug("File: %s, Role: %s, PID: %s", file_path, role, pid)

                        # 转换消息字段并添加域信息
                        if 'message' in record:
                            record['message'] = transform_message(
                                record['message'],
                                domain,
                                new_method_name,
                                parse_all
                            )
                        else:
                            # 如果消息字段不存在，创建一个包含域信息的新消息
                            record['message'] = json.dumps({"domain": domain, "name": new_method_name})
                    else:
                        # 如果原始方法名不存在，设置默认方法名
                        record['method'] = "UnknownMethod"
                        # 创建包含默认域信息的消息
                        record['message'] = json.dumps({"domain": "UnknownDomain", "name": "UnknownName"})

                    # 特殊处理：拆分processOutputs记录
                    if (new_method_name == "processOutputs" and
                            'message' in record and
                            record['message']):
                        try:
                            message_data = json.loads(record['message'])
                            if 'req_trace_info' in message_data:
                                # 为每个请求ID创建单独的记录
                                for req_id, req_info in message_data['req_trace_info'].items():
                                    new_record = record.copy()
                                    new_record['message'] = json.dumps({
                                        "rid": req_id,
                                        "recvTokenSize=": req_info.get("prompt_len", 0),
                                        "domain": "Request",
                                        "name": "httpRes"
                                    })
                                    records.append(new_record)
                                continue  # 跳过原始记录
                        except json.JSONDecodeError:
                            pass  # 如果解析失败，保留原始记录

                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.error("JSON parsing error: %s, File: %s, Line: %d", e, file_path, line_num)
                except Exception as e:
                    logger.error("Record processing error: %s, File: %s, Line: %d", e, file_path, line_num)

        if skipped_count > 0:
            logger.debug("File %s skipped %d records", file_path, skipped_count)

    except IOError as e:
        logger.error("Cannot read file %s: %s", file_path, e)

    # 处理npuModelExec分组
    if records:
        records = process_npu_model_exec_groups(records, pid)

    return records


def process_files(
        file_paths: List[str],
        db_path: str = None,
        parse_all: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """处理多个文件并返回包含结果的DataFrame"""
    if not file_paths:
        return pd.DataFrame(), pd.DataFrame()

    main_records = []
    meta_records = []
    files_without_role_pid = []

    for file_path in file_paths:
        role, pid = extract_role_and_pid_from_filename(file_path)

        # 创建元数据记录
        meta_records.append({
            'file_path': file_path,
            'role': role,
            'pid': pid
        })

        if role is None or pid is None:
            files_without_role_pid.append(file_path)
            continue

        # 处理文件并收集记录
        file_records = process_single_file(file_path, role, pid, parse_all)
        if file_records:
            main_records.extend(file_records)

    # 创建DataFrame
    main_df = pd.DataFrame(main_records) if main_records else pd.DataFrame()
    meta_df = pd.DataFrame(meta_records) if meta_records else pd.DataFrame()

    # 添加时间戳排序和markId列
    if not main_df.empty:
        # 确定用于排序的时间戳列
        time_col = None
        if 'timestamp' in main_df.columns:
            time_col = 'timestamp'
        elif 'endTimestamp' in main_df.columns:
            time_col = 'endTimestamp'

        # 按时间戳排序
        if time_col:
            main_df = main_df.sort_values(by=time_col)
            logger.debug("Sorted by %s", time_col)
        else:
            logger.warning("No timestamp column found, using original order")

        # 添加连续递增的markId
        main_df = main_df.reset_index(drop=True)
        main_df['markId'] = main_df.index
        logger.debug("Added markId column (0 to %d)", len(main_df) - 1)

    # 将结果保存到SQLite数据库
    if db_path and not main_df.empty:
        try:
            # 确保目录存在
            db_dir = os.path.dirname(db_path) or '.'
            os.makedirs(db_dir, exist_ok=True)
            os.chmod(db_dir, 0o750)

            # 连接到SQLite数据库
            conn = sqlite3.connect(db_path)

            # 将主数据保存到Mstx表
            main_df.to_sql('Mstx', conn, if_exists='replace', index=False)

            # 将元数据保存到Meta表
            meta_df['name'] = "hostname"
            meta_df['value'] = "vllm"
            meta_df.to_sql('Meta', conn, if_exists='replace', index=False)

            # 添加额外记录
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Meta (name, value) VALUES (?, ?)", ('hostname', 'vllm'))
            conn.commit()
            logger.debug("Added extra record: name='hostname', value='vllm'")

            conn.close()
            logger.info("Database saved to: %s", db_path)
            logger.debug("  - Mstx table records: %d", len(main_df))
            logger.debug("  - Meta table records: %d", len(meta_df) + 1)

            os.chmod(db_path, 0o750)
        except Exception as e:
            logger.error("Error saving database: %s", e)

    # 报告未处理的文件
    if files_without_role_pid:
        logger.warning("Files without role/PID extraction, not processed: %d", len(files_without_role_pid))
        for file in files_without_role_pid:
            logger.debug("  - %s", file)

    return main_df, meta_df


def generate_db_filename(output_dir: str = '.') -> str:
    """在指定的输出目录中生成数据库文件名"""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    db_name = "ms_service_%s.db" % timestamp
    return os.path.join(output_dir, db_name)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description=(
            'Process JSON files efficiently, extract specific fields and apply method name mapping, '
            'results saved to SQLite database'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        help='Input directory path, will recursively find all .json files'
    )

    parser.add_argument(
        '-o', '--output',
        help=(
            'Output directory path for storing database files (optional), if not specified, '
            'prof directory will be created in current directory'
        ),
        default=None
    )

    parser.add_argument(
        '--parse-all',
        action='store_true',
        help='Parse all methods, including those skipped by default'
    )

    parser.add_argument(
        '--level', '-l',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',  # 默认为info级别
        help='Set logging level (debug, info, warning, error, critical)'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 根据详细程度设置日志级别
    log_level = getattr(logging, args.level.upper())
    logger.setLevel(log_level)

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        logger.error("Input directory does not exist: %s", args.input_dir)
        return

    # 确定输出目录
    if args.output is None:
        # 如果未指定输出目录，使用当前目录下的prof目录
        args.output = os.path.join(os.getcwd(), 'prof')
        logger.info("No output directory specified, using: %s", args.output)

    # 检查输出目录
    if not os.path.isdir(args.output):
        logger.info("Creating output directory: %s", args.output)
        os.makedirs(args.output, exist_ok=True)

    # 查找JSON文件
    file_paths = find_json_files(args.input_dir)
    if not file_paths:
        logger.error("No .json files found in directory: %s", args.input_dir)
        return

    logger.info("Found %d JSON files", len(file_paths))
    if args.parse_all:
        logger.info("Mode: Parse all methods (including those skipped by default)")
    else:
        logger.debug("Mode: Skip %d configured methods", len(SKIP_METHODS))

    # 在输出目录中生成数据库文件路径
    db_path = generate_db_filename(args.output)
    logger.info("Using database path: %s", db_path)

    # 处理文件
    main_df, meta_df = process_files(file_paths, db_path, args.parse_all)

    # 打印摘要 - 大部分为debug级别，只保留关键信息在info级别
    logger.info("Processing completed.")

    if not main_df.empty:
        logger.debug("- Main data records: %d", len(main_df))
        logger.debug("- Main data columns: %d", len(main_df.columns))

        # 显示时间戳范围
        if 'timestamp' in main_df.columns:
            min_start = main_df['timestamp'].min()
            max_start = main_df['timestamp'].max()
            logger.debug("- Start time range: %.2f to %.2f", min_start, max_start)

        if 'endTimestamp' in main_df.columns:
            min_end = main_df['endTimestamp'].min()
            max_end = main_df['endTimestamp'].max()
            logger.debug("- End time range: %.2f to %.2f", min_end, max_end)

        # 显示方法分布
        if 'method' in main_df.columns:
            method_counts = main_df['method'].value_counts()
            logger.debug("Method distribution:")
            for method, count in method_counts.items():
                logger.debug("  %s: %d", method, count)

        # 显示角色分布
        if 'role' in main_df.columns:
            role_counts = main_df['role'].value_counts()
            logger.debug("Role distribution:")
            for role, count in role_counts.items():
                logger.debug("  %s: %d", role, count)

        # 显示进程ID分布
        if 'pid' in main_df.columns:
            pid_counts = main_df['pid'].value_counts()
            logger.debug("PID distribution:")
            for pid, count in pid_counts.items():
                logger.debug("  %d: %d", pid, count)

        # 显示示例记录
        logger.debug("Sample records (first 3):")
        if 'message' in main_df.columns and 'markId' in main_df.columns:
            sample_df = main_df.head(3).copy()
            for _, row in sample_df.iterrows():
                logger.debug("  markId=%d, role=%s, pid=%d, method=%s, "
                             "timestamp=%.2f, endTimestamp=%.2f, message=%s...",
                             row['markId'], row['role'], row['pid'], row['method'],
                             row['timestamp'], row['endTimestamp'], row['message'][:100])
    else:
        logger.warning("- No valid main data loaded")

    if not meta_df.empty:
        logger.debug("- Metadata records: %d", len(meta_df))
        missing_info = meta_df[(meta_df['role'].isna()) | (meta_df['pid'].isna())]
        if not missing_info.empty:
            logger.warning("- Files without role/PID extraction: %d", len(missing_info))
    else:
        logger.warning("- No metadata generated")


if __name__ == '__main__':
    main()