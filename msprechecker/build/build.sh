#!/bin/bash
# SourceCode build msprechecker
set -o pipefail
CURRENT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(realpath "${CURRENT_DIR}/../")
BUILD_DIR=${PROJECT_DIR}/build
BIN_DIR=${PROJECT_DIR}/dist
SOURCE_PATH=${PROJECT_DIR}/resources
PICS_PATH=${PROJECT_DIR}/pics
README_PATH=${PROJECT_DIR}/README.md
VERSION_FILE=${PROJECT_DIR}/msprechecker/utils/const.py
# 流水线构建时基于version_number环境变量确定版本号
if [[ -z "${version_number}" ]]; then
	version_number="LocalBuild"
fi
APP_NAME="msprechecker"
TAR_NAME="${APP_NAME}-${version_number}-Linux-$(uname -m)"

function build_msprechecker() {
  echo "-----------------start build ${APP_NAME}-------------------------------------"
  cd $PROJECT_DIR
  cp $README_PATH $BUILD_DIR
  poetry config certificates.pypi-center.cert false
  poetry config certificates.pypi-mirrors.cert false
  poetry config certificates.pypi-center-repo.cert false
  poetry config virtualenvs.create false
  poetry install --no-root

  if [ -d "$BIN_DIR" ]; then
      rm -rf "$BIN_DIR"
  fi
  mkdir -m 700 $BIN_DIR

  # 增加入口
  cp ${BUILD_DIR}/msprechecker.py ${PROJECT_DIR}
  main_path=${PROJECT_DIR}/msprechecker.py
  dist_dir=${BIN_DIR}/${TAR_NAME}
  mkdir -m 700 $dist_dir
  # 复制文档到当前目录
  cp -r $SOURCE_PATH $PICS_PATH $dist_dir

  chmod 600 $dist_dir/*
  # 创建文件夹
  pyinstaller -F ${main_path} \
  --distpath ${dist_dir} \
  --name ${APP_NAME} \
  --noconfirm \
  --runtime-tmpdir . \
  --hidden-import yaml \
  --hidden-import psutil \
  --hidden-import docker \
  --collect-submodules colorama \
  --collect-submodules packaging \
  --collect-submodules msguard \
  --exclude-module "*test*" \
  --clean

  chmod 500 ${dist_dir}/${APP_NAME}

  cd $BIN_DIR
  tar -zcvf ${TAR_NAME}.tar.gz ${TAR_NAME} --owner=root --group=root
  chmod 600 ${TAR_NAME}.tar.gz
  echo "------------------build ${APP_NAME} success-----------------------------------"
}

function replace_text() {
  # 替换html内容
  SOURCE_FILE="$1"
  TARGET_FILE="$2"
  # 检查文件是否存在
  if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误：源文件 $SOURCE_FILE 不存在"
    return 1
  fi

  if [ ! -f "$TARGET_FILE" ]; then
    echo "错误：目标文件 $TARGET_FILE 不存在"
    return 1
  fi

  # 使用 Perl 进行替换（直接传递文件路径）
  perl -i -pe '
    BEGIN {
      # 读取源文件内容
      $source_file = shift;
      open(SRC, "<", $source_file) or die "无法打开源文件: $!";
      $replacement = do { local $/; <SRC> };
      close(SRC);
    }

    # 使用 index/substr 进行纯字符串替换
    my $placeholder = "% d3MinData %";
    my $pos = index($_, $placeholder);
    if ($pos != -1) {
      substr($_, $pos, length($placeholder), $replacement);
    }
  ' "$SOURCE_FILE" "$TARGET_FILE"

  echo "替换完成：$TARGET_FILE 中的 % d3MinData % 已被 $SOURCE_FILE 的内容替换"
}

 build_msprechecker