# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
import platform

from msguard.security import open_s

from .ascend_strategy import NPUSmi
from .base_strategy import CollectStrategy
from .weight_strategy import Weight
from ..utils import Utils, PathUtil, Output, LOGGER, PreFetch, Framework


class Image(CollectStrategy):

    def __init__(self, name: str = "image"):
        super().__init__(name)
        self._image_id = ""

    @staticmethod
    def _get_vllm_version():
        try:
            from importlib.metadata import version
            return version("vllm-ascend")
        except Exception:
            cmd = ["pip", "show", "vllm-ascend"]
            output = Utils.collect_data(cmd)
            if output == "--":
                Utils.log_error_and_exit(
                    f"Failed to execute command: {' '.join(cmd)}")
                return {}
            return Utils.grep_lines(output, "Version")

    @staticmethod
    def _get_npu_type():
        pci_device_id = NPUSmi('').execute().get("PCI Device ID", "")
        if '802' in pci_device_id:
            return "A2"
        if '803' in pci_device_id:
            return "A3"
        if '806' in pci_device_id:
            return "A5"
        return "Unknown"

    @staticmethod
    def _get_mindie_version():
        mindie_version_path = '/usr/local/Ascend/mindie/latest/version.info'
        if not os.path.exists(mindie_version_path):
            Utils.log_error_and_exit(f"MindIE version file {mindie_version_path} not exist.")
        try:
            with open_s(mindie_version_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip()
                for line in lines.splitlines():
                    if 'ascend-mindie' in line.lower():
                        return line.split(':')[-1].strip()
        except Exception as e:
            Utils.log_error_and_exit(f"Failed to get MindIE version: {e}")
            return ""

    def _get_images_from_resource(self) -> dict:
        resources_dir_path = PathUtil.get_resources_root_dir_path()
        resources_path = os.path.join(resources_dir_path, "images.json")
        if not os.path.exists(resources_path):
            Utils.log_error_and_exit(f"Resources file {resources_path} not exist.")
        resources_dir = Utils.load_json(resources_path)
        image_dir = (resources_dir.get(self._target.get("image_type", ""), {})
                     .get(self._target.get("version", ""), {})
                     .get(self._target.get("architecture", ""), {})
                     .get(self._target.get("npu_type", ""), {})
                     .get(self._target.get("os_name", "").lower(), {})
                     )
        if not image_dir:
            Utils.log_error_and_exit("Image (version:{}) from dumped file not found in image resource, "
                                     "please update the resource.".format(self._target.get("version", "")))
        return image_dir

    def _check_image_in_images_list(self, tag):
        import docker
        client = docker.from_env()
        try:
            images = client.images.list()
            for image in images:
                tags = image.tags[0] if image.tags else ''
                if tag == tags:
                    self._image_id = image.short_id.split(':')[-1]
                    return True
        except docker.errors.DockerException as e:
            Utils.log_error_and_exit(f"Failed to list docker images: {e}")
        finally:
            client.close()
        return False

    def _generate_docker_run_cmd(self, timestamp: str):
        data_dir = os.path.dirname(Weight().get_weight_dir())
        docker_run_cmd = f"""docker run -it --privileged --name=sync-{self._target.get('image_type', '')}-{timestamp} --net=host --shm-size=500g \\
--device=/dev/davinci_manager \\
--device=/dev/devmm_svm \\
--device=/dev/hisi_hdc \\
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \\
-v /usr/local/sbin/:/usr/local/sbin/ \\
-v /var/log/npu/slog/:/var/log/npu/slog \\
-v /var/log/npu/profiling/:/var/log/npu/profiling \\
-v /var/log/npu/dump/:/var/log/npu/dump \\
-v /var/log/npu/:/usr/slog \\
-v /etc/hccn.conf:/etc/hccn.conf \\
-v /home/:/home/ \\
-v /data/:{data_dir} \\
{self._image_id} /bin/bash"""
        Utils.print_line()
        Output.message(docker_run_cmd)
        Utils.print_line()
        Utils.log_error_and_exit("Please modify the docker run command above and start the container")

    def _sync_host(self, timestamp: str):
        image_dir = self._get_images_from_resource()
        if "fileArtifactoryPath" in image_dir:
            download_text = f"Please download image from: {image_dir.get('fileArtifactoryPath', '')}"
        elif "pull_cmd" in image_dir:
            download_text = f"Please pull image with command: {image_dir.get('pull_cmd', '')}"
        else:
            Utils.log_error_and_exit("Broken image file")

        tag = image_dir.get("tag", "")
        if not self._check_image_in_images_list(tag):
            Utils.log_error_and_exit(f"Image (tag:{tag}) not found in local docker images. {download_text}")
        else:
            LOGGER.info(f"Image (tag:{tag}) found in local docker images.")
            LOGGER.info("Image check passed.")
            self._generate_docker_run_cmd(timestamp)

    def _sync_container(self, current):
        target_image_type = self._target.get("image_type", "")
        current_image_type = current.get("image_type", "")
        if target_image_type != current_image_type:
            Utils.log_error_and_exit("Image type in container({}) not match with dumped file({}), "
                                     "please exit the container.".format(current_image_type, target_image_type))
        if self._target != current:
            target_version = self._target.get("version", "")
            current_version = current.get("version", "")
            LOGGER.warning("Image version({}) in container not match with dumped file({}), "
                           "which may lead to unpredictable issue.".format(current_version, target_version))
        else:
            LOGGER.info("Image check passed.")

    def execute(self):
        image_type = PreFetch.get_framework()
        if image_type == Framework.VLLM:
            version = self._get_vllm_version()
        elif image_type == Framework.MINDIE:
            version = self._get_mindie_version()
        else:
            return {}
        os_name = platform.freedesktop_os_release().get("ID", "")
        architecture = platform.machine()
        npu_type = self._get_npu_type()
        LOGGER.debug(f"Got image type: {image_type}, "
                     f"version id: {version}, "
                     f"architecture: {architecture}, "
                     f"NPU type: {npu_type}, "
                     f"os name: {os_name}, "
                     )
        results = {
            "image_type": image_type.value,
            "version": version,
            "architecture": architecture,
            "npu_type": npu_type,
            "os_name": os_name.lower(),
        }
        LOGGER.debug(f"Got image info: {results}")
        return results

    def sync(self, target_data: dict):
        super().sync(target_data)
        timestamp = target_data.get("timestamp", "")
        framework = PreFetch.get_framework()
        if framework == Framework.HOST:
            self._sync_host(timestamp)
        else:
            current = self.execute()
            self._sync_container(current)
