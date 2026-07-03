# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os

from configparser import ConfigParser
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

config = ConfigParser()
config.read("./components/config/config.ini")

abs_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(abs_path, "requirements.txt"), encoding="utf-8") as f:
    required = f.read().splitlines()


class DevelopWithShUmask(develop):
    def run(self):
        cur_umask, sh_mode = 0o027, 0o750
        os.umask(cur_umask)
        develop.run(self)
        for filepath in self.get_outputs():
            if filepath.endswith(".sh"):
                os.chmod(filepath, sh_mode)  # Has to be manually 750, cannot be stat with umask


class InstallWithShUmask(install):
    def run(self):
        cur_umask, sh_mode = 0o027, 0o750
        os.umask(cur_umask)
        install.run(self)
        for filepath in self.get_outputs():
            if filepath.endswith(".sh"):
                os.chmod(filepath, sh_mode)  # Has to be manually 750, cannot be stat with umask


setup(
    name="msit",
    version="8.2.0",
    description="msIT, MindStudio Inference Tools",
    long_description_content_type="text/markdown",
    url=config.get("URL", "msit_url"),
    packages=find_packages(),
    package_data={
        "": [
            "LICENSE",
            "README.md",
            "*.txt",
            "*.bat",
            "*.sh",
            "*.cpp",
            "*.h",
            "*.ini",
        ]
    },
    data_files=[("", ["requirements.txt"])],
    license='Mulan PSL v2',
    keywords="msit",
    python_requires=">=3.7",
    install_requires=required,
    entry_points={
        "console_scripts": ["msit=components.__main__:main"],
    },
    cmdclass={"develop": DevelopWithShUmask, "install": InstallWithShUmask},
)
