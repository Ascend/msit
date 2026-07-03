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
import sys
import subprocess  # nosec
from typing import Union

from components.utils.log import logger


def get_entry_points(entry_points_name):
    try:
        from importlib import metadata

        return metadata.entry_points().get(entry_points_name, [])
    except Exception:
        import pkg_resources

        return list(pkg_resources.iter_entry_points(entry_points_name))


def is_windows():
    return sys.platform == "win32"


def warning_in_windows(title):
    if is_windows():
        logger.warning(f"{title} is not support windows")
        return True
    return False


def get_base_path():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logger.info(base)
    return base


def get_real_pkg_path(pkg_path):
    return os.path.join(get_base_path(), pkg_path)


class AitInstaller:
    @staticmethod
    def check():
        return "OK"

    @staticmethod
    def build_extra(find_links):
        logger.info("there are no more extra dependencies to build")

    @staticmethod
    def download_extra(dest):
        logger.info("there are no more extra dependencies to download")


INSTALL_INFO_MAP = [
    {
        "arg-name": "llm",
        "pkg-name": "msit-llm",
        "pkg-path": "llm",
    },
    {
        "arg-name": "analyze",
        "pkg-name": "msit-analyze",
        "pkg-path": "analyze",
    },
    {
        "arg-name": "convert",
        "pkg-name": "msit-convert",
        "pkg-path": "convert",
    },
    {
        "arg-name": "profile",
        "pkg-name": "msit-profile",
        "pkg-path": "profile",
    },
    {"arg-name": "tensor-view", "pkg-name": "msit-tensor-view", "pkg-path": "tensor_view"},
    {
        "arg-name": "benchmark",
        "pkg-name": "msit-benchmark",
        "pkg-path": "benchmark",
    },
    {
        "arg-name": "graph",
        "pkg-name": "msit-graph",
        "pkg-path": "graph",
    },
    {"arg-name": "elb", "pkg-name": "msit-elb", "pkg-path": "expert_load_balancing"},
]


def get_install_info_follow_depends(install_infos):
    all_names = set()
    for info in install_infos:
        all_names.add(info.get("pkg-name"))
        all_names.update(info.get("depends", []))
    if len(all_names) == len(install_infos):
        return install_infos
    else:
        return list(filter(lambda info: info["pkg-name"] in all_names, INSTALL_INFO_MAP))


def install_tools(names, find_links):
    if names is None or len(names) == 0:
        logger.info(
            "You can specify the components you want to install, "
            "you can select more than one, "
            "or you can use install all to install all components."
        )
        return
    if "all" in names:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = list(filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP))

        install_infos = get_install_info_follow_depends(install_infos)

    for tool_info in install_infos:
        install_tool(tool_info, find_links)


def install_tool(tool_info, find_links):
    pkg_name = tool_info.get("pkg-name")
    arg_name = tool_info.get("arg-name")
    support_windows = tool_info.get("support_windows", False)
    if not support_windows and warning_in_windows(pkg_name):
        return
    logger.info(f"installing {pkg_name}")
    pkg_path = get_real_pkg_path(tool_info.get("pkg-path"))

    if find_links is not None:
        subprocess.run(  # nosec
            [sys.executable, "-m", "pip", "install", pkg_path, "--no-index", "-f", find_links], shell=False, check=False
        )
        subprocess.run(  # nosec
            [sys.executable, "-m", "components", "build-extra", arg_name, "-f", find_links], shell=False, check=False
        )
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_path], shell=False, check=False)  # nosec
        subprocess.run([sys.executable, "-m", "components", "build-extra", arg_name], shell=False, check=False)  # nosec


def get_installer(pkg_name) -> Union[AitInstaller, None]:
    entry_points = get_entry_points("msit_sub_task_installer")
    pkg_installer = None
    for entry_point in entry_points:
        if entry_point.name == pkg_name:
            pkg_installer = entry_point.load()()
            break
    if isinstance(pkg_installer, AitInstaller):
        return pkg_installer
    return None


def check_tools(names):
    if names is None or "all" in names or len(names) == 0:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP)

    for tool_info in install_infos:
        pkg_name = tool_info.get("pkg-name")
        logger.info(pkg_name)
        for msg in check_tool(pkg_name).split("\n"):
            logger.info(f"  {msg}")


def check_tool(pkg_name):
    logger.debug("checking %s", pkg_name)
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        return "not install yet."
    else:
        return pkg_installer.check()


def build_extra(name, find_links):
    pkg_name = None

    for pkg_info in INSTALL_INFO_MAP:
        if pkg_info.get("arg-name") == name:
            pkg_name = pkg_info.get("pkg-name")
            break

    if pkg_name is None:
        raise ValueError("unknow error, pkg_name not found.")
    # msit-convert 没有extra包，跳过构建extra提示
    if pkg_name != "msit-convert":
        logger.info(f"building extra of {pkg_name}")
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        pkg_installer = AitInstaller()
    return pkg_installer.build_extra(find_links)


def download_comps(names, dest):
    if names is None or "all" in names or len(names) == 0:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP)

    install_infos = get_install_info_follow_depends(list(install_infos))

    for tool_info in install_infos:
        download_comp(tool_info, dest)
    return install_infos


def download_comp(tool_info, dest):
    pkg_name = tool_info.get("pkg-name")
    support_windows = tool_info.get("support_windows", False)
    if not support_windows and warning_in_windows(pkg_name):
        return
    logger.info(f"installing {pkg_name}")
    pkg_path = get_real_pkg_path(tool_info.get("pkg-path"))

    subprocess.run([sys.executable, "-m", "pip", "download", "-d", dest, pkg_path], shell=False, check=False)  # nosec
    subprocess.run(  # nosec
        [sys.executable, "-m", "pip", "install", "--no-index", "-f", dest, pkg_path], shell=False, check=False
    )

    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        pkg_installer = AitInstaller()
    pkg_installer.download_extra(dest)


def get_public_url(url_name):
    if not isinstance(url_name, str):
        raise ValueError("%s is not a str." % url_name)

    from pkg_resources import resource_filename
    from configparser import ConfigParser

    config_path = resource_filename('components.config', 'config.ini')
    if not config_path:
        raise FileNotFoundError("Config file not found.")

    config = ConfigParser()
    config.read(config_path)

    if config.has_section('URL') and config.has_option('URL', url_name):
        result_url = config.get('URL', url_name)
        return result_url
    else:
        raise ValueError(f"url name '{url_name}' not found in config.ini")
