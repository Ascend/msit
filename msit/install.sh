#!/bin/bash
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

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
only_benchmark=
only_analyze=
only_convert=
only_profile=
only_llm=
only_tensor_view=
only_graph=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  --full) full_install=--full;;
  --benchmark) only_benchmark=true;;
  --analyze) only_analyze=true;;
  --convert) only_convert=true;;
  --profile) only_profile=true;;
  --llm) only_llm=true;;
  --tensor-view) only_tensor_view=true;;
  --graph) only_graph=true;;
  --uninstall) uninstall=true;;
  -y) all_uninstall=-y;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

if [ ! "$(command -v python3)" ]
then
  echo "Error: python3 is not installed" >&2
  exit 1;
fi

if [ ! "$(command -v pip3)" ]; then
  echo "Error: pip3 is not installed" >&2
  exit 1;
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h : Print help menu"
  echo " --benchmark : only install benchmark component"
  echo " --analyze : only install analyze component"
  echo " --convert : only install convert component"
  echo " --profile : only install profile component"
  echo " --llm : only install llm component"
  echo "--tensor-view: only install tensor-view component"
  echo "--graph: only install graph component"
  echo " --full : using with install, install all components and dependencies, may need sudo privileges"
  echo " --uninstall : uninstall"
  echo " -y : using with uninstall, don't ask for confirmation of uninstall deletions"
  exit;
fi


uninstall(){
  pip3 uninstall msit analyze_tool convert_tool auto_optimizer msprof ${all_uninstall}
  if [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_profile ] && [ -z $only_llm ] && [ -z $only_tensor_view ]
  then
    pip3 uninstall msit msit-analyze aclruntime ais_bench msit-benchmark msit-convert msit-profile msit-llm msit-tensor-view msit-graph ${all_uninstall}
  else
    if [ ! -z $only_benchmark ]
    then
      pip3 uninstall aclruntime ais_bench ${all_uninstall}
      pip3 uninstall msit-benchmark ${all_uninstall}
    fi

    if [ ! -z $only_analyze ]
    then
      pip3 uninstall msit-analyze ${all_uninstall}
    fi

    if [ ! -z $only_convert ]
    then
      pip3 uninstall msit-convert ${all_uninstall}
    fi

    if [ ! -z $only_profile ]
    then
      pip3 uninstall msit-profile ${all_uninstall}
    fi

    if [ ! -z $only_llm ]
    then
      pip3 uninstall msit-llm ${all_uninstall}
    fi

    if [ ! -z $only_tensor_view ]
    then
      pip3 uninstall msit-tensor-view ${all_uninstall}
    fi

    if [ ! -z $only_graph ]
    then
      pip3 uninstall msit-graph ${all_uninstall}
    fi
  fi
  exit;
}


build_opchecker_so() {
    echo ""
    echo "Try building libatb_speed_torch.so for msit llm."
    cd ${CURRENT_DIR}/components/llm/msit_llm/opcheck/atb_operators
    bash build.sh
    cd -
    echo ""
}


install(){
  pip3 install ${CURRENT_DIR} ${arg_force_reinstall}

  if [ ! -z $only_benchmark ]
  then
    bash ${CURRENT_DIR}/components/benchmark/msit_benchmark/install.sh
    pip3 install ${CURRENT_DIR}/components/benchmark ${arg_force_reinstall}
  fi

  if [ ! -z $only_analyze ]
  then
    pip3 install ${CURRENT_DIR}/components/analyze ${arg_force_reinstall}
  fi

  if [ ! -z $only_convert ]
  then
    pip3 install ${CURRENT_DIR}/components/convert ${arg_force_reinstall}

    bash ${CURRENT_DIR}/components/convert/build.sh
  fi

  if [ ! -z $only_profile ]
  then
    pip3 install ${CURRENT_DIR}/components/profile ${arg_force_reinstall}
  fi

  if [ ! -z $only_llm ]
  then
      pip3 install ${CURRENT_DIR}/components/llm ${arg_force_reinstall}
      build_opchecker_so
  fi

  if [ ! -z $only_tensor_view ]
    then
        pip3 install ${CURRENT_DIR}/components/tensor_view ${arg_force_reinstall}
        build_opchecker_so
    fi

  if [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_profile ] && [ -z $only_llm ] && [ -z $only_tensor_view ]
  then
    pip3 install ${CURRENT_DIR}/components/benchmark \
    ${CURRENT_DIR}/components/analyze \
    ${CURRENT_DIR}/components/convert \
    ${CURRENT_DIR}/components/profile \
    ${CURRENT_DIR}/components/llm \
    ${CURRENT_DIR}/components/tensor_view \
    ${arg_force_reinstall}

    bash ${CURRENT_DIR}/components/benchmark/msit_benchmark/install.sh
    bash ${CURRENT_DIR}/components/convert/build.sh

    build_opchecker_so
  fi

  rm -rf ${CURRENT_DIR}/msit.egg-info
}


if [ ! -z $uninstall ]
then
  uninstall
else
  install
fi
