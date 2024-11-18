# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

from configparser import ConfigParser
from setuptools import setup, find_packages
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE

config = ConfigParser()
config.read('./components/config/config.ini')

abs_path = os.path.dirname(os.path.realpath(__file__))
with ms_open(os.path.join(abs_path, "requirements.txt")) as f:
    required = f.read().splitlines()

ait_sub_tasks = [
    {
        "name": "debug",
        "help_info": "debug a wide variety of model issues",
        "module": "components.debug.__init__",
        "attr": "debug_task",
    }
]

ait_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in ait_sub_tasks
]

setup(
    name='msit',
    version='7.0.0c912',
    description='msIT, MindStudio Inference Tools',
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={
        '': [
            'LICENSE',
            'README.md',
            '*.txt',
            '*.bat',
            '*.sh',
            '*.cpp',
            '*.h',
            '*.ini',
        ]
    },
    data_files=[('', ['requirements.txt'])],
    license='Apache-2.0',
    keywords='msit',
    python_requires='>=3.7',
    install_requires=required,
    entry_points={
        'console_scripts': ['ait=components.__main__:ait_main',
                            'msit=components.__main__:main'],
        'ait_sub_task': ait_sub_task_entry_points,
    },
)
