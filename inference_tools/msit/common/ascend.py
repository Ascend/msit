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

from msit.utils.constants import MsgConst, PathConst
from msit.utils.env import evars
from msit.utils.exceptions import MsitException
from msit.utils.log import logger
from msit.utils.path import SOFT_LINK_LEVEL_IGNORE, MsitPath, join_path
from msit.utils.toolkits import run_subprocess

_ENVVAR_ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME"
_DEFAULT_ASCEND_TOOLKIT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
_ENVVAR_ATB_HOME_PATH = "ATB_HOME_PATH"
_ATC_BIN_PATH = "compiler/bin/atc"
_OLD_ATC_BIN_PATH = "atc/bin/atc"
_ATC_MODE_OM2JSON = "1"
_ATC_MODE_GETXT2JSON = "5"


class CANN:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CANN, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.cann_home = evars.get(_ENVVAR_ASCEND_TOOLKIT_HOME, _DEFAULT_ASCEND_TOOLKIT_HOME)

    @property
    def _is_use_cxx11_for_atb(self):
        lib_atb_path = self._get_lib_atb_path()
        output = run_subprocess(["nm", "-D", lib_atb_path], check_interval=0, capture_output=True)
        abi_result = [line for line in (output or "").splitlines() if "Probe" in line and "cxx11" in line]
        return len(abi_result) > 0

    @staticmethod
    def _get_lib_atb_path():
        atb_home_path = evars.get(_ENVVAR_ATB_HOME_PATH)
        lib_atb_path = MsitPath(
            join_path(atb_home_path, "lib", "libatb.so"), PathConst.FILE, "r", PathConst.SIZE_20M
        ).check()
        return lib_atb_path

    def get_atb_probe_so_path(self):
        if self._is_use_cxx11_for_atb:
            atb_probe_so_path = join_path(self.cann_home, "tools", "ait_backend", "dump", "libatb_probe_abi1.so")
        else:
            atb_probe_so_path = join_path(self.cann_home, "tools", "ait_backend", "dump", "libatb_probe_abi0.so")
        atb_probe_so_path = MsitPath(
            atb_probe_so_path, PathConst.FILE, "r", PathConst.SIZE_20M, PathConst.SUFFIX_SO
        ).check(soft_link_level=SOFT_LINK_LEVEL_IGNORE)
        logger.info(f"The ATB probe is enabled by {atb_probe_so_path}.")
        return atb_probe_so_path

    def model2json(self, model_path: str, json_path: str):
        model_path = MsitPath(
            model_path, PathConst.FILE, "r", PathConst.SIZE_30G, PathConst.SUFFIX_CONVERT_MODEL
        ).check()
        json_path = MsitPath(json_path, PathConst.FILE, "w", suffix=PathConst.SUFFIX_JSON).check(path_exist=False)
        atc = self._get_atc_path()
        if model_path.endswith(PathConst.SUFFIX_OM):
            mode_type = _ATC_MODE_OM2JSON
        else:
            mode_type = _ATC_MODE_GETXT2JSON
        atc_cmd = [atc, "--mode=" + mode_type, "--om=" + model_path, "--json=" + json_path]
        logger.info("Start converting the model format to JSON.")
        logger.info(f"The ATC command line: {' '.join(atc_cmd)}.")
        run_subprocess(atc_cmd, check_interval=0)
        logger.info(f"The model has been converted to a JSON file, located at {json_path}.")

    def _get_atc_path(self):
        try:
            atc_path = MsitPath(
                join_path(self.cann_home, _ATC_BIN_PATH), PathConst.FILE, "e", PathConst.SIZE_20M
            ).check(soft_link_level=SOFT_LINK_LEVEL_IGNORE)
        except Exception as e1:
            logger.error(str(e1))
            try:
                atc_path = MsitPath(
                    join_path(self.cann_home, _OLD_ATC_BIN_PATH), PathConst.FILE, "e", PathConst.SIZE_20M
                ).check(soft_link_level=SOFT_LINK_LEVEL_IGNORE)
            except Exception as e2:
                raise MsitException(MsgConst.CANN_FAILED) from e2
        return atc_path


cann = CANN()
