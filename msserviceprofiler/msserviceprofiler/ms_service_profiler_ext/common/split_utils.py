# -*- coding: utf-8 -*-
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

import pandas as pd
from ms_service_profiler.utils.log import logger
from .constants import MAX_BATCH_NUMBER


EVENT_PAIRS = [
    ('batchFrameworkProcessing', 'preprocessBatch'),
    ('preprocessBatch', 'serializeExcueteMessage'),
    ('serializeExcueteMessage', 'deserializeExecuteRequestsForInfer'),
    ('grpcWriteToSlave', 'deserializeExecuteRequestsForInfer'),
    ('deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend'),
    ('convertTensorBatchToBackend', 'getInputMetadata'),
    ('getInputMetadata', 'preprocess'),
    ('getInputMetadata', 'prepareInputs'),
    ('forward', 'sample'),
    ('operatorExecute', 'sample'),
    ('postprocess', 'generateOutput'),
    ('generateOutput', 'processPythonExecResult'),
    ('deserializeExecuteResponse', 'saveoutAndContinueBatching'),
]

SYNC_EVENT_PAIRS = [
    ('processBroadcastMessage', 'deserializeExecuteRequestsForInfer'),
    ('getInputMetadata', 'preprocess'),
    ('getInputMetadata', 'prepareInputs'),
    ('preprocess', 'forward'),
    ('forward', 'sample'),
    ('prepareInputs', 'operatorExecute'),
    ('operatorExecute', 'sample'),
    ('getInputMetadata', 'sample'),
    ('sample', 'postprocess'),
    ('postprocess', 'processPythonExecResult'),
]

FILTER_LIST = ['deserializeExecuteRequestsForInfer', 'deserializeExecuteResponse', 'convertTensorBatchToBackend', 
               'processPythonExecResult', 'forward', 'sample', 'setInferBuffer', 'grpcWriteToSlave']

NAME_LIST_MINDIE = ['httpReq', 'encode', 'preprocessBatch', 'batchFrameworkProcessing', 'serializeExcueteMessage',
             'deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend', 'getInputMetadata',
             'preprocess', 'forward', 'sample', 'postprocess', 'generateOutput', 'prepareInputs', 'operatorExecute',
             'processPythonExecResult', 'deserializeExecuteResponse', 'saveoutAndContinueBatching', 'continueBatching',
             'setInferBuffer', 'grpcWriteToSlave', 'receiveInfer', 'handleTaskExecution', 'decodeEnd', 'httpRes']

NAME_LIST_VLLM = ['httpReq', 'encode', 'batchFrameworkProcessing', 'preprocess', 'forward', 'httpRes']

FULL_BATCH = ['serializeExcueteMessage', 'deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend', 
              'getInputMetadata', 'preprocess', 'forward', 'sample', 'postprocess']

CSV_COLUMNS = ['name', 'during_time', 'pid', 'tid', 'start_time', 'end_time', 'rid', 'start_datetime', 
               'end_datetime', 'batch_type', 'batch_size', 'rid_list', 'token_id_list']

HTTP_LIST = ['encode', 'httpReq', 'decodeEnd', 'httpRes']

RENAMED_COLUMNS = {
        'start_time': 'start_time(microsecond)',
        'end_time': 'end_time(microsecond)',
        'during_time': 'during_time(microsecond)'
    }


def get_name_list(service_type):
    if service_type == 'mindie':
        name_list = NAME_LIST_MINDIE
    else:
        name_list = NAME_LIST_VLLM

    return name_list



def preprocess_framework_df(framework_df):
    try:
        framework_df = framework_df[framework_df['name'].isin(NAME_LIST_MINDIE)]
        framework_df = framework_df[CSV_COLUMNS]
        framework_df = framework_df.rename(columns=RENAMED_COLUMNS)
    except KeyError as e:
        logger.warning(f"Field '{e.args[0]}' not found in msproftx.db.")
        return None

    return framework_df


def is_valid_prefill(batch_group, rid, framework_df):
    batch_row = batch_group.iloc[0]
    cur_rid = batch_row['rid_list'][0]
    if rid != '-1':
        cur_rid = rid

    target_encode = framework_df[(framework_df['rid'] == str(cur_rid)) & 
                                 (framework_df['name'] == 'httpReq')]
    return not target_encode.empty


def get_pre_half_batch(full_batch, index_list, framework_df, pid, tid):
    df_list = []
    for name, i in zip(full_batch, index_list):
        try:
            if name == 'serializeExcueteMessage':
                index = i
            elif name == 'deserializeExecuteRequestsForInfer' or name == 'convertTensorBatchToBackend':
                index = framework_df[(framework_df.index >= i) &
                            (framework_df['name'] == name) &
                            (framework_df['pid'] == pid)].index[0]
            else:
                index = framework_df[(framework_df.index >= i) &
                            (framework_df['name'] == name) &
                            (framework_df['pid'] == pid) &
                            (framework_df['tid'] == tid)].index[0]
        except IndexError:
            logger.warning(f"no named {name} line, skip this batch")
            return pd.DataFrame()
        df_list.append(index)
    return pd.concat([framework_df.loc[df_list]])
    

def get_last_half_batch(framework_df, generate_index, pid, tid):
    result = pd.DataFrame()
    index = []
    try:
        py_result_index = framework_df[(framework_df.index >= generate_index) &
                            (framework_df['name'] == 'processPythonExecResult') &
                            (framework_df['pid'] == pid) & (framework_df['tid'] == tid)].index[0]
        deserialize_index = framework_df[(framework_df.index > py_result_index) &
                            (framework_df['name'] == 'deserializeExecuteResponse')].index[0]
    except IndexError:
        logger.warning(f"no match line, skip this batch")
        return result
    index.extend([py_result_index, deserialize_index])
    result = pd.concat([result, framework_df.loc[index]])
    return result


def get_full_batch_vllm(group, framework_df):
    result = pd.DataFrame()
    try:
        batch_start_index = group[group['name'] == 'batchFrameworkProcessing'].index[0]
        all_time_index = group[group['name'] == 'AllTime'].index[0]
        concat_list = [batch_start_index, all_time_index]
        preprocess_row = framework_df[(framework_df.index > batch_start_index) &
                                    (framework_df['name'] == 'preprocess')].iloc[0]
        preprocess_pid, preprocess_tid = preprocess_row['pid'], preprocess_row['tid']
        preprocess_index = framework_df[(framework_df.index > batch_start_index) &
                                      (framework_df['name'] == 'preprocess')].index[0]
        forward_index = framework_df[(framework_df.index > batch_start_index) &
                                      (framework_df['name'] == 'forward') &
                                      (framework_df['pid'] == preprocess_pid) &
                                      (framework_df['tid'] == preprocess_tid)].index[0]
    except IndexError:
        logger.warning("no match line, skip this batch")
        return result
    concat_list.extend([preprocess_index, forward_index])
    result = pd.concat([result, framework_df.loc[concat_list]])
    result = result.sort_values(by='start_time(microsecond)').reset_index(drop=True)
    return result


def get_full_batch(group, framework_df):
    service_type = get_service_type(framework_df)
    if service_type == 'vllm':
        return get_full_batch_vllm(group, framework_df)

    name_list = get_name_list(service_type)
    start_index = name_list.index('batchFrameworkProcessing')
    end_index = name_list.index('saveoutAndContinueBatching')
    batch_start_index = group[group['name'] == 'batchFrameworkProcessing'].index[0]
    concat_list = [batch_start_index]
    index_list = []
    index = batch_start_index
    # 找到generator_row
    for name in FULL_BATCH:
        query = f"index > @index and name =='{name}'"
        if framework_df.query(query).empty:
            logger.warning(f"no named {name} line, skip this batch")
            return pd.DataFrame()
        index = framework_df[(framework_df.index > index) &
                             (framework_df['name'] == name)].index[0]
        index_list.append(index)
    if framework_df.query('name == "generateOutput" and index > @index').empty:
        logger.warning(f"no generateOutput line, skip this batch")
        return pd.DataFrame()
    generate_row = framework_df[(framework_df.index > index) &
                                (framework_df['name'] == 'generateOutput')].iloc[0]
    generate_pid, generate_tid = generate_row['pid'], generate_row['tid']
    generate_index = framework_df[(framework_df.index > index) &
                                  (framework_df['name'] == 'generateOutput')].index[0]
    concat_list.append(generate_index)

    # 找到前半部分的字段
    result_pre = get_pre_half_batch(FULL_BATCH, index_list, framework_df, generate_pid, generate_tid)
    # 拼接后半部分字段
    result_last = get_last_half_batch(framework_df, generate_index, generate_pid, generate_tid)
    if result_last.empty or result_pre.empty:
        return pd.DataFrame()
    group = group.drop(group[group['name'].isin(name_list[start_index: end_index])].index)
    result = pd.concat([framework_df.loc[concat_list], result_pre, result_last, group])
    result = result.sort_values(by='start_time(microsecond)').reset_index(drop=True)
    return result


def get_groups(framework_df, batch_size, rid, name):
    result_df = []
    
    groups = framework_df.groupby((framework_df['name'] == 'batchFrameworkProcessing').cumsum())
    for _, group in groups:
        if rid != '-1':
            batch_group = group[(group['name'] == 'batchFrameworkProcessing') &
                                (group['batch_type'] == name)]
            batch_group.loc[:, 'rid_list'] = batch_group['rid_list'].apply(lambda x: [str(i) for i in x])
            batch_group = batch_group[batch_group['rid_list'].apply(lambda x: rid in x)]
        elif batch_size > 0:
            batch_group = group[(group['name'] == 'batchFrameworkProcessing') &
                                (group['batch_type'] == name) &
                                (group['batch_size'] == str(batch_size))]
        if batch_group.empty:
            continue

        if name == 'Prefill' and not is_valid_prefill(batch_group, rid, framework_df):
            continue
        result = get_full_batch(group, framework_df)

        if not result.empty:
            result_df.append(result) 

    return result_df


def get_event_pair_df(framework_df, name):
    new_rows = []
    # 遍历每个事件对
    for current_name, next_name in EVENT_PAIRS:
        # 收集当前事件的所有实例
        current_events = framework_df[framework_df['name'] == current_name].copy()
        len_current = len(current_events)
        if len_current == 0:
            logger.debug(f'{name}: No data named {current_name}')
            continue
        next_events = framework_df[framework_df['name'] == next_name].copy()
        len_next = len(next_events)
        if len_next == 0:
            logger.debug(f'{name}: No data named {next_name}')
            continue
        # 提取当前事件的pid，tid
        for i in range(len_current):
            cur_pid = current_events.iloc[i]['pid']
            cur_tid = current_events.iloc[i]['tid']
            if (current_name, next_name) in SYNC_EVENT_PAIRS:
                next_events_filter = next_events[(next_events['pid'] == cur_pid) & (next_events['tid'] == cur_tid)]
                if len(next_events_filter) == 0:
                    continue
                current_end = current_events.iloc[i]['end_time(microsecond)']
                next_start = next_events_filter.iloc[0]['start_time(microsecond)']
                if next_start < current_end:
                    continue
            else:
                next_events_filter = next_events[next_events['pid'] == cur_pid]
                if len(next_events_filter) == 0:
                    continue
                current_end = current_events.iloc[i]['end_time(microsecond)']
                next_start = next_events_filter.iloc[0]['start_time(microsecond)']
                if next_start < current_end:
                    continue
            # 创建新行
            new_row = {
                'name': f"Between-{current_name}-{next_name}",
                'during_time(microsecond)': next_start - current_end,
                'start_time(microsecond)': current_end,
                'end_time(microsecond)': next_start,
                'pid': current_events.iloc[i]['pid'],
                'tid': current_events.iloc[i]['tid'],
            }
            new_rows.append(new_row)

    # 将新行转换为 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 合并原数据和新行，并按时间排序
    result_df = pd.concat([framework_df, new_df], ignore_index=True)
    result_df = result_df.sort_values(by='start_time(microsecond)').reset_index(drop=True)
    return result_df


def get_post_event_df(post_event_pairs, framework_df, name):
    new_rows = []
    # 遍历每个事件对
    for current_name, next_name in post_event_pairs:
        # 收集当前事件的所有实例
        current_events = framework_df[framework_df['name'] == current_name].copy()
        len_current = len(current_events)
        if len_current == 0:
            logger.debug(f'{name}: No data named {current_name}')
            continue
        next_events = framework_df[framework_df['name'] == next_name].copy()
        len_next = len(next_events)
        if len_next == 0:
            logger.debug(f'{name}: No data named {next_name}')
            continue
        current_end = current_events.iloc[0]['end_time(microsecond)']
        next_start = next_events.iloc[0]['start_time(microsecond)']
        # 创建新行
        new_row = {
            'name': f"Between-{current_name}-{next_name}",
            'during_time(microsecond)': next_start - current_end,
            'start_time(microsecond)': current_end,
            'end_time(microsecond)': next_start,
            'pid': current_events.iloc[0]['pid'],
            'tid': current_events.iloc[0]['tid'],
        }
        new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)

    return new_df


def postprocess_framework_df(framework_df, post_event_pairs, name):
    if framework_df.empty:
        logger.warning(f'{name}: df is empty')
        return framework_df

    service_type = get_service_type(framework_df)
    
    # 更新AllTime行
    framework_df.loc[framework_df['name'] == 'AllTime', ['start_time(microsecond)', 'end_time(microsecond)']] = \
        framework_df.loc[framework_df['name'] == 'AllTime', ['end_time(microsecond)', 'start_time(microsecond)']].values

    if service_type == 'mindie':
        if name == 'Prefill':
            if 'continueBatching' not in framework_df['name'].values:
                post_event_pairs.append(('deserializeExecuteResponse', 'httpRes'))
                if 'httpRes' not in framework_df['name'].values:
                    post_event_pairs.append(('deserializeExecuteResponse', 'AllTime'))
        new_rows = []
        if 'preprocessBatch' not in framework_df['name'].values:
            post_event_pairs.append(('batchFrameworkProcessing', 'serializeExcueteMessage'))
        if 'continueBatching' not in framework_df['name'].values and name == 'Decode':
            post_event_pairs.append(('deserializeExecuteResponse', 'AllTime'))
    elif service_type == 'vllm':
        start_index = NAME_LIST_VLLM.index('batchFrameworkProcessing')
        end_index = NAME_LIST_VLLM.index('httpRes')
        key_names_vllm = NAME_LIST_VLLM[start_index:end_index]
        key_event_pairs = [(key_names_vllm[i], key_names_vllm[i + 1]) for i in range(len(key_names_vllm) - 1)]
        for pair in key_event_pairs:
            post_event_pairs.append(pair)
        post_event_pairs.append(('forward', 'AllTime'))


    
    post_df = get_post_event_df(post_event_pairs, framework_df, name)
    framework_df = pd.concat([framework_df, post_df], ignore_index=True)
    framework_df = framework_df.sort_values(by=['start_time(microsecond)']).reset_index(drop=True)
    framework_df = framework_df[framework_df['name'] != 'handleTaskExecution']  # 删除 'handleTaskExecution' 行
    if name == 'Decode':
        framework_df = framework_df.drop(framework_df[framework_df['name'].isin(HTTP_LIST)].index)
    framework_df['during_time(microsecond)'] = framework_df['during_time(microsecond)'] / 1000
    
    if 'prepareInputs' in framework_df['name'].values and 'operatorExecute' in framework_df['name'].values:
        delete_list = ['preprocess', 'forward', 'Between-getInputMetadata-preprocess', 
                       'Between-preprocess-forward', 'Between-forward-sample']
    else:
        delete_list = ['prepareInputs', 'operatorExecute', 'Between-getInputMetadata-prepareInputs', 
                       'Between-prepareInputs-operatorExecute', 'Between-operatorExecute-sample']

    framework_df = framework_df[~framework_df['name'].isin(delete_list)]
    all_time_rows = framework_df[framework_df['name'] == 'AllTime']
    framework_df = framework_df[framework_df['name'] != 'AllTime']
    framework_df = pd.concat([framework_df, all_time_rows], ignore_index=True)
    framework_df.drop_duplicates(subset='name', inplace=True)
    return framework_df


def get_filter_rule_df(framework_df):
    serialize_execute_index = framework_df[framework_df['name'] == 'serializeExcueteMessage'].index
    if len(serialize_execute_index) != 0:
        serialize_execute_index = serialize_execute_index[0]
        rows_to_drop = framework_df[(framework_df.index < serialize_execute_index) &
                                    (framework_df['name'].isin(FILTER_LIST))].index
        framework_df = framework_df.drop(rows_to_drop)
    
    # 删除 'name' 为 'encode'、'httpReq'、'decode'、'httpRes' 的行
    framework_df = framework_df.drop(framework_df[framework_df['name'].isin(HTTP_LIST)].index)
    return framework_df


def get_batch_framework(framework_df, name):
    if name == 'Prefill':
        framework_df = get_filter_rule_df(framework_df)
    framework_df = get_event_pair_df(framework_df, name)

    grouped_df = framework_df.groupby('name')
    sorted_groups = grouped_df.apply(lambda x: x.sort_values(by='during_time(microsecond)', ascending=True))
    if 'generateOutput' in grouped_df.groups:
        generate_output_pid = grouped_df.get_group('generateOutput')['pid'].iloc[0]
        generate_output_tid = grouped_df.get_group('generateOutput')['tid'].iloc[0]
    elif 'forward' in grouped_df.groups:
        sample_group = grouped_df.get_group('forward')
        last_sample_row = sample_group.tail(1)
        generate_output_pid = last_sample_row['pid'].values[0]
        generate_output_tid = last_sample_row['tid'].values[0]
    else:
        return pd.DataFrame()

    filter_name = ['deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend',
                   'Between-deserializeExecuteRequestsForInfer-convertTensorBatchToBackend',
                   'Between-convertTensorBatchToBackend-getInputMetadata']
    result_df = sorted_groups[(sorted_groups['pid'] == generate_output_pid) &
                              (sorted_groups['tid'] == generate_output_tid)]
    result_df1 = sorted_groups[(sorted_groups['pid'] == generate_output_pid) &
                               (sorted_groups['name'].isin(filter_name))]
    result_df = pd.concat([result_df, result_df1])
    other_group = sorted_groups[sorted_groups['pid'] != generate_output_pid]
    other_group = other_group[~other_group['name'].isin(result_df['name'])]
    other_df = other_group.drop_duplicates(subset='name', keep='first')
    result_df = pd.concat([result_df, other_df])
    return result_df


def get_filter_df(framework_df, name):
    """
    从第一条httpReq开始
    """
    if name == 'Prefill':
        filter_name = 'httpReq'
    else:
        filter_name = 'batchFrameworkProcessing'
    first_batch_schedule_index = 0
    try:
        first_batch_schedule_index = framework_df[framework_df['name'] == filter_name].index[0]
    except IndexError:
        logger.warning(f"{name}: No data named {filter_name}")
    filter_df = framework_df.loc[first_batch_schedule_index:]
    return filter_df


def get_statistics_data(framework_df, filter_name, name):
    if framework_df.empty:
        logger.warning(f"{name}: The dataframe is empty, no csv file create")
        return framework_df
    batch = framework_df[framework_df['name'] == filter_name]
    if len(batch) == 1:
        start_index = framework_df[framework_df['name'] == filter_name].index[-1]
        end_index = None
    else:
        start_index = framework_df[framework_df['name'] == filter_name].index[-2]
        end_index = framework_df[framework_df['name'] == filter_name].index[-1] - 1
    framework_df['max'] = framework_df.groupby('name')['during_time(microsecond)'].transform('max')
    framework_df['min'] = framework_df.groupby('name')['during_time(microsecond)'].transform('min')
    framework_df['mean'] = framework_df.groupby('name')['during_time(microsecond)'].transform('mean')
    framework_df['std'] = framework_df.groupby('name')['during_time(microsecond)'].transform('std')
    # 标准差为0时显示0
    framework_df['std'] = framework_df['std'].fillna(0)
    framework_df.insert(2, 'max', framework_df.pop('max'))
    framework_df.insert(3, 'min', framework_df.pop('min'))
    framework_df.insert(4, 'mean', framework_df.pop('mean'))
    framework_df.insert(5, 'std', framework_df.pop('std'))
    framework_df = framework_df.rename(columns={'during_time(microsecond)': 'during_time(millisecond)'})
    if name == 'Decode':
        framework_df = framework_df.iloc[:, :10]
    else:
        framework_df = framework_df.iloc[:, :11]
    return framework_df[start_index: end_index]


def get_batch_all_time(framework_df, name):
    new_rows = []
    batch_rows = framework_df[framework_df['name'] == 'batchFrameworkProcessing']
    pids = batch_rows['pid'].unique()
    for pid in pids:
        current_events = framework_df[(framework_df['name'] == 'batchFrameworkProcessing') & 
                                    (framework_df['pid'] == pid)].copy()

        len_current = len(current_events)

        if len_current < 2:
            logger.warning(f"{name}: The length of batchFrameworkProcessing is less two")
            return framework_df
        
        # 补充插入AllTime行
        for i in range(len_current - 1):
            current_row = current_events.iloc[i]
            next_row = current_events.iloc[i + 1]
            during_time = next_row['start_time(microsecond)'] - current_row['start_time(microsecond)']

            all_time_row = {
                    'name': 'AllTime',
                    'start_time(microsecond)': current_row['start_time(microsecond)'] + 2,
                    'end_time(microsecond)': next_row['start_time(microsecond)'],
                    'during_time(microsecond)': during_time,
                    'start_datetime': current_row['start_datetime'],
                    'end_datetime': next_row['start_datetime'],
                }

            # 将 'AllTime' 行添加到新行列表
            new_rows.append(all_time_row)

    new_df = pd.DataFrame(new_rows)
    result_df = pd.concat([framework_df, new_df], ignore_index=True)
    result_df = result_df.sort_values(by=['start_time(microsecond)', 'name']).reset_index(drop=True)

    return result_df


def get_batch_concat_df(filter_df, framework_df, cacl_num, rid, name):
    concat_df = pd.DataFrame()
    empty_row = pd.DataFrame(index=[0])
    for i in range(cacl_num):
        cur_rid = filter_df[i].iloc[0]['rid_list'][0]
        if rid != '-1':
            cur_rid = rid
        cur_df = get_batch_framework(filter_df[i], name)
        if cur_df.equals(pd.DataFrame()):
            continue
        if name == 'Prefill':
            http_df = framework_df[(framework_df['rid'] == str(cur_rid)) & (framework_df['name'].isin(HTTP_LIST))]
            cur_df = pd.concat([cur_df, http_df], ignore_index=True)
            post_event_pairs = [
                ('encode', 'batchFrameworkProcessing'),
                ('continueBatching', 'httpRes'),
            ]
        else:
            post_event_pairs = [
                ('continueBatching', 'AllTime'),
            ]
        cur_df = cur_df.sort_values(by='start_time(microsecond)')
        cur_df = postprocess_framework_df(cur_df, post_event_pairs, name)

        if cur_df.equals(pd.DataFrame()):
            logger.debug(f'{name}: cur_df is empty, continue')
            continue

        concat_df = pd.concat([concat_df, empty_row, cur_df], ignore_index=True)
    return concat_df


def process_exporter(framework_df, batch_size, batch_num, rid, name):
    # 划分组
    result_df = get_groups(framework_df, batch_size, rid, name)
    len_result_df = len(result_df)

    if len(result_df) == 0:
        logger.warning('%s: no batchFrameworkProcessing with batch_size %d' % (name, batch_size))
        logger.warning('%s: no batchFrameworkProcessing with rid %r' % (name, rid))
        return pd.DataFrame()
    
    cacl_num = len_result_df if len_result_df <= batch_num else batch_num
    cacl_num = min(cacl_num, MAX_BATCH_NUMBER)
    concat_df = get_batch_concat_df(result_df, framework_df, cacl_num, rid, name)
    return concat_df


def get_service_type(framework_df):
    name_set = set(list(framework_df['name']))
    if 'deserializeExecuteResponse' in name_set:
        service_type = 'mindie'
    else:
        service_type = 'vllm'
    return service_type