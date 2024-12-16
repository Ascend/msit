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
from configparser import ConfigParser
from setuptools import setup, find_packages

__version__ = "8.0.0rc1230"

requirements_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(requirements_path, "requirements.txt")) as f:
    required = f.read().splitlines()

config = ConfigParser()
config.read("./msit/config.ini")

setup(
    name="msit", 
    version=__version__, 
    description="msit (MindStudio Inference Tools)", 
    long_description= """
    msit (MindStudio Inference Tools), [Powered by MindStudio].
    Providing one-site debugging and optimization toolkits for inference on Ascend devices.
    For any issue, refer FAQ first. Gitee repo: Ascend/msit, wiki.
    """,
    long_description_content_type="text/markdown", 
    url=config.get("URL", "msit_url"), 
    author="Ascend Team", 
    packages=find_packages(include=["msit", "msit.*"]), 
    package_data={
        "": ["LICENSE", "*.md", "*.txt", "*.cpp", "*.h"]
    }, 
    license="Apache-2.0", 
    keywords=["msit", "probe", "surgeon"], 
    python_requires=">=3.8", 
    install_requires=required, 
    entry_points={
        "console_scripts": ["msit=msit.__main__:main"], 
    }, 
)
