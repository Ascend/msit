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


class Constant:
    # env precheck
    DATA_DIR = "data"
    MAX_PATH_SIZE = 255  # Max path size
    MAX_FILE_BYTES = 64 * 1024 * 1024 * 1024  # Input file size limit 64 GB
    DIR_AUTHORITY = 0o750
    File_AUTHORITY = 0o640
    RATIO_THRESHOLD = 0.05
    BANDWIDTH_THRESHOLD = 0.30

