/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MSIT_ACLDUMP_ACLAPI_H
#define MSIT_ACLDUMP_ACLAPI_H

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <map>

#define ACL_MAX_DIM_CNT 128
#define ACL_MAX_TENSOR_NAME_LEN 128

extern "C" {

using aclError = int;
using aclrtContext = void *;
using aclmdlDesc = struct aclmdlDesc;
using aclmdlDataset = struct aclmdlDataset;
using aclDataBuffer = struct aclDataBuffer;
typedef enum {
    ACL_DT_UNDEFINED = -1, // 未知数据类型，默认值
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
} aclDataType;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
    ACL_MEM_TYPE_LOW_BAND_WIDTH = 0x0100,
    ACL_MEM_TYPE_HIGH_BAND_WIDTH = 0x1000
} aclrtMemMallocPolicy;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,     // Host内的内存复制
    ACL_MEMCPY_HOST_TO_DEVICE,   // Host到Device的内存复制
    ACL_MEMCPY_DEVICE_TO_HOST,   // Device到Host的内存复制
    ACL_MEMCPY_DEVICE_TO_DEVICE, // Device内或Device间的内存复制
    ACL_MEMCPY_DEFAULT,          // 由系统根据源、目的内存地址自行判断拷贝方向
} aclrtMemcpyKind;

typedef struct aclmdlIODims {
    char name[ACL_MAX_TENSOR_NAME_LEN]; /**< tensor name */
    size_t dimCount;                    /**Shape中的维度个数，如果为标量，则维度个数为0*/
    int64_t dims[ACL_MAX_DIM_CNT];      /**< 维度信息 */
} aclmdlIODims;

}

namespace MSIT_C {
typedef struct LoadFileResult {
    uint32_t modelId;
    aclError ret;
} LoadFileResult;

class AclApi {
public:
    static AclApi &GetInstance();
    AclApi();
    aclError ACLAPI_AclInit(const char *cfg = nullptr);
    aclError ACLAPI_AclRtSetDevice(int32_t deviceId);
    aclError ACLAPI_AclRtCreateContext(aclrtContext *context, int32_t deviceId);
    LoadFileResult ACLAPI_AclMdlLoadFromFile(const char *modelPath);
    aclmdlDesc *ACLAPI_AclMdlCreateDesc();
    aclError ACLAPI_AclMdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);
    aclDataBuffer *ACLAPI_AclCreateDataBuffer(void *data, size_t size);
    size_t ACLAPI_AclMdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index);
    size_t ACLAPI_AclMdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index);
    size_t ACLAPI_AclMdlGetNumInputs(aclmdlDesc *modelDesc);
    size_t ACLAPI_AclMdlGetNumOutputs(aclmdlDesc *modelDesc);
    const char *ACLAPI_AclMdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index);
    aclDataType ACLAPI_AclMdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index);
    aclError ACLAPI_AclMdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
    aclError ACLAPI_AclRtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
    aclmdlDataset *ACLAPI_AclMdlCreateDataset();
    aclError ACLAPI_AclMdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *databuffer);
    aclError ACLAPI_AclMdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);
    aclError ACLAPI_AclRtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
    size_t ACLAPI_AclMdlGetDatasetNumBuffers(const aclmdlDataset *dataset);
    aclDataBuffer *ACLAPI_AclMdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index);
    aclError ACLAPI_AclDestroyDataBuffer(const aclDataBuffer *dataBuffer);
    aclError ACLAPI_AclMdlDestroyDataset(const aclmdlDataset *dataset);
    aclError ACLAPI_AclRtFree(void *devPtr);
    aclError ACLAPI_AclFinalize();
    aclError ACLAPI_AclMdlUnload(uint32_t modelId);
    aclError ACLAPI_AclMdlDestroyDesc(aclmdlDesc *modelDesc);
    aclError ACLAPI_AclRtDestroyContext(aclrtContext context);
    aclError ACLAPI_AclRtResetDevice(int32_t deviceId);

private:
    void LoadAclApi();
};

#define CALL_ACL_API(func, ...) MSIT_C::AclApi::GetInstance().ACLAPI_##func(__VA_ARGS__)

}

#endif
