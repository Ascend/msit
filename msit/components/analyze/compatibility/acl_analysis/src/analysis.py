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

import os
from typing import Dict, List

from knowledge_base import Knowledge, KnowledgeGroup
from components.utils.log import logger
from components.utils.check.rule import Rule


def check_filetype(filename: str):
    return filename.endswith('.cpp') or filename.endswith('.py') or filename.endswith('.h')


def check_api(acl_api: str, line: str):
    line = line.strip()
    if line.startswith('#'):
        return False
    if line.startswith('//'):
        return False
    pos = line.find(acl_api)
    subline: str = line[pos + len(acl_api) :]
    subline = subline.strip()
    return subline.startswith('(')


def match_knowledge(line) -> Dict[str, List[Knowledge]]:
    # 遍历API变更迁移分析知识库
    result: Dict[str, List[Knowledge]] = {}
    for knowledge in KnowledgeGroup.get_knowledges():
        acl_apis = knowledge.apis
        for acl_api in acl_apis:
            if acl_api in line:
                if acl_api not in result and not check_api(acl_api, line):
                    continue
                if not knowledge.analysis(line):
                    continue
                if acl_api not in result:
                    result[acl_api] = []
                result.get(acl_api).append(knowledge)
    return result


def process_line(line: str, line_num: int, filepath: str, result: Dict[Knowledge, List[str]]):
    match_result = match_knowledge(line)
    if len(match_result) == 0:
        return
    
    for api, knowledges in match_result.items():
        for knowledge in knowledges:
            if knowledge not in result:
                result[knowledge] = []
            result[knowledge].append(f"{api} {filepath} Line: {line_num}")


def process_file(filepath: str, result: Dict[Knowledge, List[str]]):
    line_num = 0
    with open(filepath, encoding='UTF-8') as f:
        for line in f.readlines():
            line_num += 1
            process_line(line, line_num, filepath, result)


def process_directory(path: str, result: Dict[Knowledge, List[str]]):
    for root, _, files in os.walk(path):
        for filename in files:
            if not check_filetype(filename):
                continue

            filepath = os.path.join(root, filename)
            if not os.path.isfile(os.path.realpath(filepath)):
                continue

            check_res = Rule.input_file().check(file_path)
            if not check_res:
                logger.error("Failed to load file %r due to %s", file_path, check_res)

            process_file(file_path, result)


def analysis_310_to_310b(path: str) -> Dict[Knowledge, List[str]]:
    if os.path.islink(os.path.abspath(path)):
        raise PermissionError('Opening softlink directory is not permitted.')

    logger.info("[info] Start analysis.")
    # 遍历该目录下的所有code文件
    result: Dict[Knowledge, List[str]] = {}
    process_directory(path, result)
    
    logger.info("[info] Analysis finished.")
    return result
