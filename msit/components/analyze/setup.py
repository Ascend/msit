# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from configparser import ConfigParser
from setuptools import setup, find_packages  # type: ignore
from components.utils.file_open_check import ms_open

config = ConfigParser()
config.read('../config/config.ini')


with ms_open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with ms_open('README.md', encoding='utf-8') as f:
    long_description = f.read()

ait_sub_tasks = [
    {
        "name": "analyze",
        "help_info": "Analyze tool to evaluate compatibility of model conversion",
        "module": "model_evaluation.__main__",
        "attr": "get_cmd_instance",
    }
]

ait_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}" for t in ait_sub_tasks
]

setup(
    name='msit-analyze',
    version='7.0.0c912',
    description='inference analyze tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={'model_evaluation': ['data/op_map/*.yaml']},
    license='Apache-2.0',
    keywords='analyze tool',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    python_requires='>=3.7',
    entry_points={
        'ait_sub_task': ait_sub_task_entry_points,
        'ait_sub_task_installer': ['msit-analyze=model_evaluation.__install__:AnalyzeInstall'],
    },
)
