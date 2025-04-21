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
from msit.module.probe.base import OfflineModelActuator
from msit.module.probe.dump.acl_manager import ACLConst, acl_resource_manager
from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.io import load_om_model
from msit.utils.log import logger

_BUFFER_METHOD_MAP = {"input": acl.get_input_size_by_index, "output": acl.get_output_size_by_index}


class OmModelActuator(OfflineModelActuator):
    def __init__(self, model_path, input_shape, input_path, device_id, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)
        self.device_id = device_id

        self.ptr_model_desc = None
        self.ptr_input_dataset = None
        self.ptr_output_dataset = None

        self.model_id = None
        self.input_size = 0
        self.output_size = 0

        self.input_ptr_size = []
        self.output_ptr_size = []
        acl_resource_manager.initlize(device_id)

    def load_model(self):
        self.model_id = load_om_model(self.model_path)

    def get_input_tensor_info(self):
        inputs_tensor_info = []
        self._get_model_info()
        for index in range(self.input_size):
            name = acl.get_input_name_by_index(self.ptr_model_desc, index)
            if name is None:
                raise MsitException(MsgConst.CALL_FAILED, f"Get input name by index:{index} failed!")

            shape, ret = acl.get_input_dims(self.ptr_model_desc, index)
            if shape is None or ret != ACLConst.ACL_SUCCESS:
                raise MsitException(MsgConst.CALL_FAILED, f"Get input shape by index:{index} failed!")

            dtype = acl.get_input_data_type(self.ptr_model_desc, index)
            if dtype is None:
                raise MsitException(MsgConst.CALL_FAILED, f"Get input type by index:{index} failed!")

            inputs_tensor_info.append({"name": name, "shape": shape["dims"], "type": dtype})
        logger.info(f"Model input tensor info: {inputs_tensor_info}.")
        return inputs_tensor_info

    def infer(self, input_map):
        self._create_data_buffer()
        self._copy_data_from_host_to_device(input_map)
        self._run()
        self._destroy_data_buffer()
        self._destroy_resource()
        acl_resource_manager.destroy_resource(self.device_id)

    def _run(self):
        ret = acl.execute(self.model_id, self.ptr_input_dataset, self.ptr_output_dataset)
        if ret != ACLConst.ACL_SUCCESS:
            raise MsitException(MsgConst.CALL_FAILED, f"Model execute failed! ErrorCode = {ret}.")
        logger.info("Model execute success!")

    def _get_model_info(self):
        self.ptr_model_desc = acl.create_desc()
        if self.ptr_model_desc is None:
            raise MsitException(MsgConst.CALL_FAILED, "Create model description Failed!")

        ret = acl.get_desc(self.ptr_model_desc, self.model_id)
        if ret != ACLConst.ACL_SUCCESS:
            raise MsitException(MsgConst.CALL_FAILED, f"Get model description failed! ErrorCode = {ret}.")

        self.input_size = acl.get_num_inputs(self.ptr_model_desc)
        if self.input_size is None:
            raise MsitException(MsgConst.CALL_FAILED, "Get input nums failed!")

        self.output_size = acl.get_num_outputs(self.ptr_model_desc)
        if self.output_size is None:
            raise MsitException(MsgConst.CALL_FAILED, "Get output nums failed!")

        logger.info("Create model description Success!")

    def _create_data_buffer(self):
        for mode in ["input", "output"]:
            ptr_dataset = getattr(self, f"ptr_{mode}_dataset", None)
            data_size = getattr(self, f"{mode}_size", 0)
            ptr_size_map = getattr(self, f"{mode}_ptr_size", [])

            ptr_dataset = acl.create_dataset()
            if ptr_dataset is None:
                raise MsitException(MsgConst.CALL_FAILED, f"Create {mode} dataset failed!")

            for index in range(data_size):
                temp_buffer_size = _BUFFER_METHOD_MAP.get(mode)(self.ptr_model_desc, index)
                if temp_buffer_size is None:
                    raise MsitException(MsgConst.CALL_FAILED, f"Get {mode} size by index:{index} failed!")

                temp_ptr, ret = acl.rt_malloc(temp_buffer_size)
                if ret != ACLConst.ACL_SUCCESS:
                    raise MsitException(MsgConst.CALL_FAILED, f"{mode.title()} malloc failed! ErrorCode = {ret}.")

                ptr_size_map.append({"buffer": temp_ptr, "size": temp_buffer_size})

                temp_buffer = acl.create_databuffer(temp_ptr, temp_buffer_size)
                if temp_buffer is None:
                    acl.rt_free(temp_ptr)
                    raise MsitException(MsgConst.CALL_FAILED, f"Create {mode} buffer failed!")

                ret = acl.add_dataset_buffer(ptr_dataset, temp_buffer)
                if ret != ACLConst.ACL_SUCCESS:
                    acl.rt_free(temp_ptr)
                    raise MsitException(
                        MsgConst.CALL_FAILED, f"Add {mode} buffer to dataset failed! ErrorCode = {ret}."
                    )
            setattr(self, f"ptr_{mode}_dataset", ptr_dataset)
            setattr(self, f"{mode}_size_map", ptr_size_map)

    def _destroy_resource(self):
        ret = acl.unload(self.model_id)
        if ret != ACLConst.ACL_SUCCESS:
            logger.error(f"Unload model failed! ErrorCode = {ret}.")
        logger.info("End to unload model.")

        if self.ptr_model_desc is not None:
            ret = acl.destroy_desc(self.ptr_model_desc)
            if ret != ACLConst.ACL_SUCCESS:
                logger.error(f"Destroy model description failed! ErrorCode = {ret}.")

    def _destroy_data_buffer(self):
        for mode in ["input", "output"]:
            dataset = getattr(self, f"ptr_{mode}_dataset", None)
            ptr_size_map = getattr(self, f"{mode}_ptr_size", [])

            if dataset is None or not ptr_size_map:
                return

            buffer_nums = acl.get_dataset_num_buffers(dataset)
            if buffer_nums is None:
                logger.error(f"Get dataset num buffers failed!")
                return

            for index in range(buffer_nums):
                data_buffer = acl.get_dataset_buffer(dataset, index)
                if data_buffer is None:
                    logger.error(f"From {mode} dataset get dataBuffer failed!")
                    continue
                ret = acl.destroy_databuffer(data_buffer)
                if ret != ACLConst.ACL_SUCCESS:
                    logger.error(f"Destroy dataBuffer failed! ErrorCode = {ret}.")

            ret = acl.destroy_dataset(dataset)
            if ret != ACLConst.ACL_SUCCESS:
                logger.error(f"Destroy {mode} dataset failed! ErrorCode = {ret}.")

            for items in ptr_size_map:
                ptr = items.get("buffer", None)
                ret = acl.rt_free(ptr)
                if ret != ACLConst.ACL_SUCCESS:
                    logger.error(f"Free Failed! ErrorCode = {ret}.")

    def _copy_data_from_host_to_device(self, input_map):
        if len(input_map) != len(self.input_ptr_size):
            logger.warning(f"input_map size:{len(input_map)} not equal input_ptr_size:{len(self.input_ptr_size)}")
            return

        for index, (_, input_data) in enumerate(input_map.items()):
            dest_ptr = self.input_ptr_size[index].get("buffer", None)
            dest_size = self.input_ptr_size[index].get("size", 0)
            byte_data = input_data.tobytes()
            ret = acl.rt_memcpy(dest_ptr, dest_size, byte_data, len(byte_data), ACLConst.ACL_MEMCPY_HOST_TO_DEVICE)
            if ret != ACLConst.ACL_SUCCESS:
                logger.error(f"Memcpy Input data from host to device failed! ErrorCode = {ret}.")
                return
