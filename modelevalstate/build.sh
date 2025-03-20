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

echo "=========== Step: 1.0 - Preparing Build Version ...... "

datetime=$(date +%m%d)
archive_dir_name=$(grep version pyproject.toml | awk -F'"' '{print $2}')
build_version="${archive_dir_name}_${datetime}"
echo "buildVersion=${build_version}"
echo "buildVersion=${build_version}" >buildInfo.properties
python3 -m pip install --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple poetry
echo "=========== Step: 2.0 - Build Package For Python...... "
python3 -m poetry build
echo "build success"