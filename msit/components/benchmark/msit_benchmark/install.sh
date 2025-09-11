#!/bin/bash
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

declare -i ret_ok=0
declare -i ret_run_failed=1

WHL_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('whl_base_url'))")
TOOLS_BAS_URL_SUFFIX=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('tools_base_url'))")
TOOLS_BAS_URL="git+$TOOLS_BAS_URL_SUFFIX"

download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall aclruntime"
    elif [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"
    PLATFORM=$(uname -m)

    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_${PLATFORM}.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    wget -c "${WHL_BASE_URL}${WHL_NAME}"

    if [ "$PLATFORM" == "aarch64" ]; then
        if [ "$PYTHON3_MINI_VERSION" == "7" ]; then
            sha256Value="f1b6da04cb454bdf1f3f3373c346ec485dca7c425184806faa6b9f197e82016b"
        elif [ "$PYTHON3_MINI_VERSION" == "8" ]; then
            sha256Value="b7a15ac9fbb94f7f52e6cf574db4eb63101939a84c118e4c72dfd2bdf4f0c39a"
        elif [ "$PYTHON3_MINI_VERSION" == "9" ]; then
            sha256Value="af3be0a0fb0c74dabb0c3e7307dd2054673f635d22958011fa860f71a42d5dd4"
        elif [ "$PYTHON3_MINI_VERSION" == "10" ]; then
            sha256Value="2d7298ca9b0cf62c9914772f0d25c1326abef777bc451b3fdc345b0896835011"
        else
            echo "Unsupported python3 version"
            exit 1
        fi
    elif [ "$PLATFORM" == "x86_64" ]; then
        if [ "$PYTHON3_MINI_VERSION" == "7" ]; then
            sha256Value="a4be8a768e227f8b2db8f8b73a00e1541786cc9c42c726a5809f0a085bfd9168"
        elif [ "$PYTHON3_MINI_VERSION" == "8" ]; then
            sha256Value="c25e152364d3be7bff473162f8928547c3534f1826a35b867043372a58f0a380"
        elif [ "$PYTHON3_MINI_VERSION" == "9" ]; then
            sha256Value="9768354e32452c0073800064985f6b5169f7d7821452b008e3d95187162931b6"
        elif [ "$PYTHON3_MINI_VERSION" == "10" ]; then
            sha256Value="45d781faff585a92b58b7a6b4bd0564fc521bd0b578e31de6f852a19f069e664"
        else
            echo "Unsupported python3 version"
            exit 1
        fi
    else
        echo "Unsupported platform"
        exit 1
    fi

    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

download_and_install_ais_bench() {
    AIS_BENCH_VERSION=`pip3 show ais_bench | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall ais_bench"
    elif [ "$AIS_BENCH_VERSION" = "0.0.2" ]; then
        echo "ais_bench==0.0.2 already installed, skip"
        return
    fi

    WHL_NAME="ais_bench-0.0.2-py3-none-any.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    wget -c "${WHL_BASE_URL}${WHL_NAME}"

    sha256Value="ff55373a11d9975eaad497a230c9fb0d93856dc184790b8f168143c9c5f1cccd"
    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

ret=0
download_and_install_aclruntime
ret=$(( $ret + $? ))

download_and_install_ais_bench
ret=$(( $ret + $? ))

exit $ret