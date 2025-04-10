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

from msit.lib.msit_c import acl
from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger


class ACLConst:
    ACL_SUCCESS = 0

    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2


class ACLResourceManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ACLResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.ptr_context = None

    def initlize(self, device_id=0):
        ret = acl.init()
        if ret != ACLConst.ACL_SUCCESS:
            raise MsitException(MsgConst.CALL_FAILED, f"Acl init failed! ErrorCode = {ret}.")
        logger.info("Acl init success!")

        ret = acl.rt_set_device(device_id)
        if ret != ACLConst.ACL_SUCCESS:
            raise MsitException(MsgConst.CALL_FAILED, f"Acl set device:{device_id} failed! ErrorCode = {ret}.")
        logger.info(f"Set device:{device_id} success!")

        self.ptr_context, ret = acl.rt_create_context(device_id)
        if ret != ACLConst.ACL_SUCCESS:
            raise MsitException(MsgConst.CALL_FAILED, f"Acl create context failed! ErrorCode = {ret}.")
        logger.info("Create new context success!")

    def destroy_resource(self, device_id):
        if self.ptr_context is not None:
            ret = acl.rt_destroy_context(self.ptr_context)
            if ret != ACLConst.ACL_SUCCESS:
                logger.error(f"Destroy context failed! ErrorCode = {ret}.")

        ret = acl.rt_reset_device(device_id)
        if ret != ACLConst.ACL_SUCCESS:
            logger.error(f"Reset deivce failed! DeviceId = {device_id}, ErrorCode = {ret}.")
        logger.info(f"End to reset device:{device_id}.")

        ret = acl.finalize()
        if ret != ACLConst.ACL_SUCCESS:
            logger.error(f"Finalize failed! ErrorCode = {ret}.")
        logger.info("End to finalize.")


acl_resource_manager = ACLResourceManager()
