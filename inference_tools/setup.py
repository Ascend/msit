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

__version__ = "8.1.0rc0630"

import os
import sys
from configparser import ConfigParser
import platform
import subprocess

from setuptools import find_packages, setup

_COMPAT_REQUIREMENTS_MAP = {"tf": "requirements_tf.txt", "default": "requirements.txt"}


def parse_args():
    compat_flag = None
    if "--compat" in sys.argv:
        index = sys.argv.index("--compat")
        compat_flag = sys.argv[index + 1]
        sys.argv.remove("--compat")
        sys.argv.remove(compat_flag)
    return compat_flag


def get_requirements(compat_name=None):
    requirements_parent_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements")
    requirements_file = _COMPAT_REQUIREMENTS_MAP.get(compat_name, _COMPAT_REQUIREMENTS_MAP["default"])
    with open(os.path.join(requirements_parent_path, requirements_file)) as f:
        required_lines = f.read().splitlines()
    return required_lines


compat = parse_args()
required = get_requirements(compat)

config = ConfigParser()
config.read("./msit/config.ini")

arch = platform.machine()
build_cmd = f"bash ./build.sh -j16 -a {arch} -v {sys.version_info.major}.{sys.version_info.minor}"
p = subprocess.run(build_cmd.split(), shell=False)
if p.returncode != 0:
    raise RuntimeError(f"Failed to build source({p.returncode})")

setup(
    name="msit",
    version=__version__,
    description="msit (MindStudio Inference Tools)",
    long_description="""
    msit (MindStudio Inference Tools), [Powered by MindStudio].
    Providing one-site debugging and optimization toolkits for inference on Ascend devices.
    For any issue, refer FAQ first. Gitee repo: Ascend/msit, wiki.
    """,
    long_description_content_type="text/markdown",
    url=config.get("URL", "msit_url"),
    author="Ascend Team",
    packages=find_packages(include=["msit", "msit.*"]),
    package_data={"": ["LICENSE", "*.md", "*.txt", "*.cpp", "*.h", "*.json", "*.ini", "lib/*.so"]},
    license="Apache-2.0",
    keywords=["msit", "probe", "surgeon"],
    python_requires=">=3.7",
    install_requires=required,
    entry_points={"console_scripts": ["msit=msit.__main__:main"]},
)
