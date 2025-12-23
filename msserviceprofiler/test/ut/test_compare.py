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
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch
from ms_service_profiler_ext.compare import main


# Test cases for main
def test_main_given_valid_args_when_run_then_success():
    # Arrange

    create_db("input", "ms_service_huawei-834623.db")
    create_db("golden", "ms_service_huawei-834618.db")

    args = ["input", "golden", "--output-path", "output", "--log-level", "info"]

    # Act
    with patch('sys.argv', ['compare.py'] + args):
        main()

    # Assert
    assert os.path.exists("output/span_comparation_result.csv")


def create_db(path_set, db_name):
    Path(path_set).mkdir(exist_ok=True)
    # 创建或连接到数据库文件
    db_file = f"{path_set}/{db_name}"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # 创建Mstx表
    create_mstx_table_sql = """
    CREATE TABLE IF NOT EXISTS Mstx (
    message TEXT,
    flag INTEGER,
    markId INTEGER,
    timestamp INTEGER,
    endTimestamp INTEGER,
    pid INTEGER,
    tid INTEGER
    );
    """
    cursor.execute(create_mstx_table_sql)
    # 插入Mstx示例数据
    mstx_sample_data = [
        ({"domain": "Connector", "name": "DeserializeRequests", "type": 2}, 2, 0, 84773025046, 4145584773047436,
         834623, 834632),
        ({"domain": "Connector", "name": "DeserializeRequests", "type": 2}, 2, 0, 84773025056, 4145584773047496,
         834623, 834632),
        ({"domain": "Connector", "name": "DeserializeRequests", "type": 2}, 2, 0, 84773026046, 4145584773057436,
         834623, 834632)
    ]

    # 将字典转换为 JSON 格式的字符串
    mstx_sample_json_data = [
        (json.dumps(data[0]), data[1], data[2], data[3], data[4], data[5], data[6])
        for data in mstx_sample_data
    ]

    cursor.executemany("""
    INSERT INTO Mstx (message, flag, markId, timestamp, endTimestamp, pid, tid)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, mstx_sample_json_data)

    # 创建Meta表
    create_meta_table_sql = """
        CREATE TABLE IF NOT EXISTS Meta (
        name TEXT,
        value TEXT
        );
        """
    cursor.execute(create_meta_table_sql)
    # 插入Meta示例数据
    meta_sample_data = [
        ("hostname", "huawei"),
        ("ppid", "834635")
    ]
    cursor.executemany("""
        INSERT INTO Meta (name, value)
        VALUES (?, ?)
        """, meta_sample_data)

    conn.commit()
    conn.close()