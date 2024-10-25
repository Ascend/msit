# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import pandas as pd
import utils
from components.utils.log import logger
from components.utils.check.rule import Rule


analysis_apis = {
    "Crop": "acldvppVpcCropAsync",
    "Resize": "acldvppVpcResizeAsync",
    "CropResize": "acldvppVpcCropResizeAsync",
    "CropPaste": "acldvppVpcCropAndPasteAsync",
    "CropResizePaste": "acldvppVpcCropResizePasteAsync",
    "MakeBorder": "acldvppVpcMakeBorderAsync",
    "CropBatch": "acldvppVpcBatchCropAsync",
    "CropResizeBatch": "acldvppVpcBatchCropResizeAsync",
    "CropPasteBatch": "acldvppVpcBatchCropAndPasteAsync",
    "CropResizePasteBatch": "acldvppVpcBatchCropResizePasteAsync",
    "MakeBorderBatch": "acldvppVpcBatchCropResizeMakeBorderAsync",
    "VdecSF": "aclvdecSendFrame",
    "VdecSkip": "aclvdecSendSkippedFrame",
    "VdecCCA": "acldvppVpcConvertColorAsync",
    "AippInputFormat_py": "acl.mdl.set_aipp_input_format",
    "AippInputFormat_cpp": "aclmdlSetAIPPInputFormat",
    "AippCscParams_py": "acl.mdl.set_aipp_csc_params",
    "AippCscParams_cpp": "aclmdlSetAIPPCscParams",
}


def evaluate(path):
    prof_path = os.path.join(path, 'profiling')
    prof_path = utils.check_profiling_data(prof_path)

    # dvpp vpc接口选择和优化
    analyze_dvpp_vpc(prof_path, analysis_apis)

    # dvpp vdec接口选择和优化
    analyze_dvpp_vdec(prof_path, analysis_apis)


def analyze_dvpp_vpc(profiling_path, api):
    acl_statistic_data = utils.get_statistic_profile_data_path(profiling_path)
    if not Rule.input_file().check(acl_statistic_data):
        logger.error("read csv file failed, please check %r", acl_statistic_data)
        raise OSError
    data = pd.read_csv(acl_statistic_data)
    count_crop = 0
    count_resize = 0
    count_crop_resize = 0
    count_crop_paste = 0
    count_crop_resize_paste = 0
    count_make_border = 0
    has_suggestion = 0

    for line in data.itertuples():
        if len(line) > 5:
            if line[1] == api["Crop"]:
                count_crop = line[5]
            elif line[1] == api["Resize"]:
                count_resize = line[5]
            elif line[1] == api["CropResize"]:
                count_crop_resize = line[5]
            elif line[1] == api["CropPaste"]:
                count_crop_paste = line[5]
            elif line[1] == api["CropResizePaste"]:
                count_crop_resize_paste = line[5]
            elif line[1] == api["MakeBorder"]:
                count_make_border = line[5]
        else:
            raise IndexError("Index out of range: The data row does not have enough columns.")

    if count_crop != 0 or count_resize != 0:
        has_suggestion = 1
    elif count_crop_resize != 0 or count_crop_paste != 0:
        has_suggestion = 1
    elif count_crop_resize_paste != 0 or count_make_border != 0:
        has_suggestion = 1
    if has_suggestion != 0:
        if count_crop >= 2:
            logger.info(f'检测到使用{api["Crop"]}接口，循环处理图片，建议使用{api["CropBatch"]}接口。')
        if count_crop >= 2 and count_resize >= 2 and count_crop == count_resize:
            logger.info(
                f'检测到连续调用{api["Crop"]}和{api["Resize"]}接口，同时循环处理多张图，建议使用{api["CropResizeBatch"]}接口。'
            )
        if count_crop_resize >= 2:
            logger.info(f'检测到连续调用{api["CropResize"]}接口，建议使用{api["CropResizeBatch"]}接口。')
        if count_crop_paste >= 2:
            logger.info(f'检测到循环调用{api["CropPaste"]}接口，建议使用{api["CropPasteBatch"]}接口。')
        if count_crop_resize_paste >= 2:
            logger.info(f'检测到循环调用{api["CropResizePaste"]}接口，建议使用{api["CropResizePasteBatch"]}接口。')
        if count_make_border >= 2 and (count_crop != 0 or count_resize != 0):
            if count_make_border == count_crop or count_make_border == count_resize:
                logger.info(
                    f'检测到循环调用{api["Crop"]}和{api["Resize"]}和{api["MakeBorder"]}接口，建议使用{api["MakeBorderBatch"]}接口。'
                )
    else:
        logger.info("在这个AI处理器上，可能没有使用VPCAPI接口，因而在这个方向上，知识库暂时没有调优建议。")


def analyze_dvpp_vdec(profiling_path, api):
    acl_statistic_data = utils.get_statistic_profile_data_path(profiling_path)
    if not Rule.input_file().check(acl_statistic_data):
        logger.error("read csv file failed, please check %r", acl_statistic_data)
        raise OSError
    data = pd.read_csv(acl_statistic_data)
    count_vpc_cca = 0
    count_vdec_sf = 0
    for line in data.itertuples():
        if line[1] == api["VdecCCA"]:
            count_vpc_cca = line[5]
        if line[1] == api["VdecSF"]:
            count_vdec_sf = line[5]
    if count_vpc_cca != 0 or count_vdec_sf != 0:
        if count_vpc_cca >= 1 & count_vdec_sf == 0:
            logger.info(f'检测使用了{api["VdecCCA"]}接口。')
            logger.info(
                f'如果您使用的是昇腾710 AI处理器，该处理器视频解码接口{api["VdecSF"]}'
                '支持输出YUV420SP格式或RGB888格式，可设置接口参数输出不同的格式，'
                '建议省去调用{api["VdecCCA"]}进行格式转换的步骤，减少接口调用。'
            )
            logger.info(
                f'同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，'
                '推荐您使用{api["VdecSkip"]}接口进行解码，不输出解码结果。'
            )
        if count_vdec_sf >= 1 & count_vpc_cca == 0:
            logger.info(f'检测使用了{api["VdecSF"]}接口。')
            logger.info(
                '在昇腾710 AI处理器上，VPC图像处理功能支持输出YUV400格式（灰度图像）,'
                '如果模型推理的输入图像是灰度图像，建议您直接使用VPC功能，无需再使用AIPP色域转换功能。'
            )
            logger.info(
                '同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，'
                '推荐您使用{api["VdecSkip"]}接口进行解码，不输出解码结果。'
            )
        if count_vpc_cca >= 1 & count_vdec_sf >= 1:
            logger.info(f'检测同时使用了{api["VdecCCA"]}接口以及{api["VdecSF"]}接口。')
            logger.info(
                f'在昇腾710 AI处理器上，视频解码接口{api["VdecSF"]}'
                '支持输出YUV420SP格式或RGB888格式，可设置接口参数输出不同的格式，'
                '建议省去调用{api["VdecCCA"]}进行格式转换的步骤，减少接口调用。'
            )
            logger.info(
                f'同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，'
                '推荐您使用{api["VdecSkip"]}接口进行解码，不输出解码结果。'
            )
    else:
        logger.info('在此 AI 处理器上，并没有使用到 VDECAPI接口。所以在这个方向上，知识库并没有调优建议。')
        logger.info(
            f'但是在视频解码+模型推理的场景下，如果用户视频的帧数很大并且不是每一帧都需要推断，'
            '建议您使用{api["VdecSkip"]}接口以提升使用体验。'
        )
