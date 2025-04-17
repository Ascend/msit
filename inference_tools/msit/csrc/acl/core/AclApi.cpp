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

#include "acl/include/AclApi.h"
#include <iostream>
#include "utils/log.h"

namespace MSIT_C {

constexpr const char *ascendAclName = "libascendcl.so";

using aclInitFuncType = aclError (*)(const char *);
using aclrtSetDeviceFuncType = aclError (*)(int32_t);
using aclrtCreateContextFuncType = aclError (*)(aclrtContext *, int32_t);
using aclmdlLoadFromFileFuncType = aclError (*)(const char *, uint32_t *);
using aclmdlCreateDescFuncType = aclmdlDesc *(*)();
using aclmdlGetDescFuncType = aclError (*)(aclmdlDesc *, uint32_t);
using aclmdlGetInputSizeByIndexFuncType = size_t (*)(aclmdlDesc *, size_t);
using aclmdlGetOutputSizeByIndexFuncType = size_t (*)(aclmdlDesc *, size_t);
using aclmdlGetNumInputsFuncType = size_t (*)(aclmdlDesc *);
using aclmdlGetNumOutputsFuncType = size_t (*)(aclmdlDesc *);
using aclmdlGetInputNameByIndexFuncType = const char *(*)(const aclmdlDesc *, size_t);
using aclmdlGetInputDataTypeFuncType = aclDataType (*)(const aclmdlDesc *, size_t);
using aclmdlGetInputDimsFuncType = aclError (*)(const aclmdlDesc *, size_t, aclmdlIODims *);
using aclrtMallocFuncType = aclError (*)(void **, size_t, aclrtMemMallocPolicy);
using aclmdlCreateDatasetFuncType = aclmdlDataset *(*)();
using aclCreateDataBufferFuncType = aclDataBuffer *(*)(void *, size_t);
using aclmdlAddDatasetBufferFuncType = aclError (*)(aclmdlDataset *, aclDataBuffer *);
using aclmdlExecuteFuncType = aclError (*)(uint32_t, const aclmdlDataset *, aclmdlDataset *);
using aclrtMemcpyFuncType = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
using aclmdlGetDatasetNumBuffersFuncType = size_t (*)(const aclmdlDataset *);
using aclmdlGetDatasetBufferFuncType = aclDataBuffer *(*)(const aclmdlDataset *, size_t);
using aclDestroyDataBufferFuncType = aclError (*)(const aclDataBuffer *);
using aclmdlDestroyDatasetFuncType = aclError (*)(const aclmdlDataset *);
using aclrtFreeFuncType = aclError (*)(void *);
using aclFinalizeFuncType = aclError (*)();
using aclmdlUnloadFuncType = aclError (*)(uint32_t);
using aclmdlDestroyDescFuncType = aclError (*)(aclmdlDesc *);
using aclrtDestroyContextFuncType = aclError (*)(aclrtContext);
using aclrtResetDeviceFuncType = aclError (*)(int32_t);

static aclInitFuncType aclInitFunc = nullptr;
static aclrtSetDeviceFuncType aclrtSetDeviceFunc = nullptr;
static aclrtCreateContextFuncType aclrtCreateContextFunc = nullptr;
static aclmdlLoadFromFileFuncType aclmdlLoadFromFileFunc = nullptr;
static aclmdlCreateDescFuncType aclmdlCreateDescFunc = nullptr;
static aclmdlGetDescFuncType aclmdlGetDescFunc = nullptr;
static aclmdlGetInputSizeByIndexFuncType aclmdlGetInputSizeByIndexFunc = nullptr;
static aclmdlGetOutputSizeByIndexFuncType aclmdlGetOutputSizeByIndexFunc = nullptr;
static aclmdlGetNumInputsFuncType aclmdlGetNumInputsFunc = nullptr;
static aclmdlGetNumOutputsFuncType aclmdlGetNumOutputsFunc = nullptr;
static aclmdlGetInputNameByIndexFuncType aclmdlGetInputNameByIndexFunc = nullptr;
static aclmdlGetInputDataTypeFuncType aclmdlGetInputDataTypeFunc = nullptr;
static aclmdlGetInputDimsFuncType aclmdlGetInputDimsFunc = nullptr;
static aclrtMallocFuncType aclrtMallocFunc = nullptr;
static aclmdlCreateDatasetFuncType aclmdlCreateDatasetFunc = nullptr;
static aclCreateDataBufferFuncType aclCreateDataBufferFunc = nullptr;
static aclmdlAddDatasetBufferFuncType aclmdlAddDatasetBufferFunc = nullptr;
static aclmdlExecuteFuncType aclmdlExecuteFunc = nullptr;
static aclrtMemcpyFuncType aclrtMemcpyFunc = nullptr;
static aclmdlGetDatasetNumBuffersFuncType aclmdlGetDatasetNumBuffersFunc = nullptr;
static aclmdlGetDatasetBufferFuncType aclmdlGetDatasetBufferFunc = nullptr;
static aclDestroyDataBufferFuncType aclDestroyDataBufferFunc = nullptr;
static aclmdlDestroyDatasetFuncType aclmdlDestroyDatasetFunc = nullptr;
static aclrtFreeFuncType aclrtFreeFunc = nullptr;
static aclFinalizeFuncType aclFinalizeFunc = nullptr;
static aclmdlUnloadFuncType aclmdlUnloadFunc = nullptr;
static aclmdlDestroyDescFuncType aclmdlDestroyDescFunc = nullptr;
static aclrtDestroyContextFuncType aclrtDestroyContextFunc = nullptr;
static aclrtResetDeviceFuncType aclrtResetDeviceFunc = nullptr;

const std::map<const char *, void **> functionMap = {
    {"aclInit", reinterpret_cast<void **>(&aclInitFunc)},
    {"aclrtSetDevice", reinterpret_cast<void **>(&aclrtSetDeviceFunc)},
    {"aclrtCreateContext", reinterpret_cast<void **>(&aclrtCreateContextFunc)},
    {"aclmdlLoadFromFile", reinterpret_cast<void **>(&aclmdlLoadFromFileFunc)},
    {"aclmdlCreateDesc", reinterpret_cast<void **>(&aclmdlCreateDescFunc)},
    {"aclmdlGetDesc", reinterpret_cast<void **>(&aclmdlGetDescFunc)},
    {"aclmdlGetNumInputs", reinterpret_cast<void **>(&aclmdlGetNumInputsFunc)},
    {"aclmdlGetNumOutputs", reinterpret_cast<void **>(&aclmdlGetNumOutputsFunc)},
    {"aclmdlGetInputNameByIndex", reinterpret_cast<void **>(&aclmdlGetInputNameByIndexFunc)},
    {"aclmdlGetInputSizeByIndex", reinterpret_cast<void **>(&aclmdlGetInputSizeByIndexFunc)},
    {"aclmdlGetInputDataType", reinterpret_cast<void **>(&aclmdlGetInputDataTypeFunc)},
    {"aclmdlGetInputDims", reinterpret_cast<void **>(&aclmdlGetInputDimsFunc)},
    {"aclmdlGetOutputSizeByIndex", reinterpret_cast<void **>(&aclmdlGetOutputSizeByIndexFunc)},
    {"aclrtMalloc", reinterpret_cast<void **>(&aclrtMallocFunc)},
    {"aclmdlCreateDataset", reinterpret_cast<void **>(&aclmdlCreateDatasetFunc)},
    {"aclCreateDataBuffer", reinterpret_cast<void **>(&aclCreateDataBufferFunc)},
    {"aclmdlAddDatasetBuffer", reinterpret_cast<void **>(&aclmdlAddDatasetBufferFunc)},
    {"aclmdlExecute", reinterpret_cast<void **>(&aclmdlExecuteFunc)},
    {"aclrtMemcpy", reinterpret_cast<void **>(&aclrtMemcpyFunc)},
    {"aclmdlGetDatasetNumBuffers", reinterpret_cast<void **>(&aclmdlGetDatasetNumBuffersFunc)},
    {"aclmdlGetDatasetBuffer", reinterpret_cast<void **>(&aclmdlGetDatasetBufferFunc)},
    {"aclDestroyDataBuffer", reinterpret_cast<void **>(&aclDestroyDataBufferFunc)},
    {"aclmdlDestroyDataset", reinterpret_cast<void **>(&aclmdlDestroyDatasetFunc)},
    {"aclrtFree", reinterpret_cast<void **>(&aclrtFreeFunc)},
    {"aclFinalize", reinterpret_cast<void **>(&aclFinalizeFunc)},
    {"aclmdlUnload", reinterpret_cast<void **>(&aclmdlUnloadFunc)},
    {"aclmdlDestroyDesc", reinterpret_cast<void **>(&aclmdlDestroyDescFunc)},
    {"aclrtDestroyContext", reinterpret_cast<void **>(&aclrtDestroyContextFunc)},
    {"aclrtResetDevice", reinterpret_cast<void **>(&aclrtResetDeviceFunc)},
};

AclApi &AclApi::GetInstance()
{
    static AclApi instance;
    return instance;
}

AclApi::AclApi()
{
    LoadAclApi();
}

void AclApi::LoadAclApi()
{
    static void *libAscendcl = nullptr;

    if (libAscendcl != nullptr)
    {
        INFO_LOG("No need to load acl api again.");
        return;
    }

    libAscendcl = dlopen(ascendAclName, RTLD_LAZY);
    if (libAscendcl == nullptr)
    {
        ERROR_LOG("Failed to search libascendcl.so." + std::string(dlerror()));
        return;
    }

    for (auto &iter : functionMap)
    {
        if (*(iter.second) != nullptr)
        {
            continue;
        }
        *(iter.second) = dlsym(libAscendcl, iter.first);
        if (*(iter.second) == nullptr)
        {
            ERROR_LOG("Failed to load function " + std::string(iter.first) +
                        " from libascendcl.so." + std::string(dlerror()));
            dlclose(libAscendcl);
            libAscendcl = nullptr;
            return;
        }
        DEBUG_LOG("Load function " + std::string(iter.first) + " from libascendcl.so.");
    }
}

aclError AclApi::ACLAPI_AclInit(const char *cfg)
{
    if (aclInitFunc == nullptr)
    {
        throw std::runtime_error("API aclInit does not have a definition.");
    }
    return aclInitFunc(cfg);
}

aclError AclApi::ACLAPI_AclRtSetDevice(int32_t deviceId)
{
    if (aclrtSetDeviceFunc == nullptr)
    {
        throw std::runtime_error("API aclrtSetDevice does not have a definition.");
    }
    return aclrtSetDeviceFunc(deviceId);
}

aclError AclApi::ACLAPI_AclRtCreateContext(aclrtContext *context, int32_t deviceId)
{
    if (aclrtCreateContextFunc == nullptr)
    {
        throw std::runtime_error("API aclrtCreateContext does not have a definition.");
    }
    return aclrtCreateContextFunc(context, deviceId);
}

LoadFileResult AclApi::ACLAPI_AclMdlLoadFromFile(const char *modelPath)
{
    uint32_t modelId;
    if (aclmdlLoadFromFileFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlLoadFromFile does not have a definition.");
    }
    aclError ret = aclmdlLoadFromFileFunc(modelPath, &modelId);
    LoadFileResult loadFileResult;
    loadFileResult.modelId = modelId;
    loadFileResult.ret = ret;
    return loadFileResult;
}

aclmdlDesc *AclApi::ACLAPI_AclMdlCreateDesc()
{
    if (aclmdlCreateDescFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlCreateDesc does not have a definition.");
    }
    return aclmdlCreateDescFunc();
}

aclError AclApi::ACLAPI_AclMdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId)
{
    if (aclmdlGetDescFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetDesc does not have a definition.");
    }
    return aclmdlGetDescFunc(modelDesc, modelId);
}

size_t AclApi::ACLAPI_AclMdlGetNumInputs(aclmdlDesc *modelDesc)
{
    if (aclmdlGetNumInputsFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetNumInputs does not have a definition.");
    }
    return aclmdlGetNumInputsFunc(modelDesc);
}

size_t AclApi::ACLAPI_AclMdlGetNumOutputs(aclmdlDesc *modelDesc)
{
    if (aclmdlGetNumOutputsFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetNumInputs does not have a definition.");
    }
    return aclmdlGetNumOutputsFunc(modelDesc);
}

const char *AclApi::ACLAPI_AclMdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index)
{
    if (aclmdlGetInputNameByIndexFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetInputNameByIndex does not have a definition.");
    }
    return aclmdlGetInputNameByIndexFunc(modelDesc, index);
}

size_t AclApi::ACLAPI_AclMdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    if (aclmdlGetInputSizeByIndexFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetInputSizeByIndex does not have a definition.");
    }
    return aclmdlGetInputSizeByIndexFunc(modelDesc, index);
}

size_t AclApi::ACLAPI_AclMdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    if (aclmdlGetOutputSizeByIndexFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetInputSizeByIndex does not have a definition.");
    }
    return aclmdlGetOutputSizeByIndexFunc(modelDesc, index);
}

aclDataType AclApi::ACLAPI_AclMdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index)
{
    if (aclmdlGetInputDataTypeFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetInputDataType does not have a definition.");
    }
    return aclmdlGetInputDataTypeFunc(modelDesc, index);
}

aclError AclApi::ACLAPI_AclMdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    if (aclmdlGetInputDimsFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetInputDims does not have a definition.");
    }
    return aclmdlGetInputDimsFunc(modelDesc, index, dims);
}

aclError AclApi::ACLAPI_AclRtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    if (aclrtMallocFunc == nullptr)
    {
        throw std::runtime_error("API aclrtMalloc does not have a definition.");
    }
    return aclrtMallocFunc(devPtr, size, policy);
}

aclmdlDataset *AclApi::ACLAPI_AclMdlCreateDataset()
{
    if (aclmdlCreateDatasetFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlCreateDataset does not have a definition.");
    }
    return aclmdlCreateDatasetFunc();
}

aclDataBuffer *AclApi::ACLAPI_AclCreateDataBuffer(void *data, size_t size)
{
    if (aclCreateDataBufferFunc == nullptr)
    {
        throw std::runtime_error("API aclCreateDataBuffer does not have a definition.");
    }
    return aclCreateDataBufferFunc(data, size);
}

aclError AclApi::ACLAPI_AclMdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *databuffer)
{
    if (aclmdlAddDatasetBufferFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlAddDatasetBuffer does not have a definition.");
    }
    return aclmdlAddDatasetBufferFunc(dataset, databuffer);
}

aclError AclApi::ACLAPI_AclMdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output)
{
    if (aclmdlExecuteFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlExecute does not have a definition.");
    }
    return aclmdlExecuteFunc(modelId, input, output);
}

aclError AclApi::ACLAPI_AclRtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    if (aclrtMemcpyFunc == nullptr)
    {
        throw std::runtime_error("API aclrtMemcpy does not have a definition.");
    }
    return aclrtMemcpyFunc(dst, destMax, src, count, kind);
}

size_t AclApi::ACLAPI_AclMdlGetDatasetNumBuffers(const aclmdlDataset *dataset)
{
    if (aclmdlGetDatasetNumBuffersFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetDatasetNumBuffers does not have a definition.");
    }
    return aclmdlGetDatasetNumBuffersFunc(dataset);
}

aclDataBuffer *AclApi::ACLAPI_AclMdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index)
{
    if (aclmdlGetDatasetBufferFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlGetDatasetBuffer does not have a definition.");
    }
    return aclmdlGetDatasetBufferFunc(dataset, index);
}

aclError AclApi::ACLAPI_AclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    if (aclDestroyDataBufferFunc == nullptr)
    {
        throw std::runtime_error("API aclDestroyDataBuffer does not have a definition.");
    }
    return aclDestroyDataBufferFunc(dataBuffer);
}

aclError AclApi::ACLAPI_AclMdlDestroyDataset(const aclmdlDataset *dataset)
{
    if (aclmdlDestroyDatasetFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlDestroyDataset does not have a definition.");
    }
    return aclmdlDestroyDatasetFunc(dataset);
}

aclError AclApi::ACLAPI_AclRtFree(void *devPtr)
{
    if (aclrtFreeFunc == nullptr)
    {
        throw std::runtime_error("API aclrtFree does not have a definition.");
    }
    return aclrtFreeFunc(devPtr);
}

aclError AclApi::ACLAPI_AclFinalize()
{
    if (aclFinalizeFunc == nullptr)
    {
        throw std::runtime_error("API aclFinalize does not have a definition.");
    }
    return aclFinalizeFunc();
}

aclError AclApi::ACLAPI_AclMdlUnload(uint32_t modelId)
{
    if (aclmdlUnloadFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlUnload does not have a definition.");
    }
    return aclmdlUnloadFunc(modelId);
}

aclError AclApi::ACLAPI_AclMdlDestroyDesc(aclmdlDesc *modelDesc)
{
    if (aclmdlDestroyDescFunc == nullptr)
    {
        throw std::runtime_error("API aclmdlDestroyDesc does not have a definition.");
    }
    return aclmdlDestroyDescFunc(modelDesc);
}

aclError AclApi::ACLAPI_AclRtDestroyContext(aclrtContext context)
{
    if (aclrtDestroyContextFunc == nullptr)
    {
        throw std::runtime_error("API aclrtDestroyContext does not have a definition.");
    }
    return aclrtDestroyContextFunc(context);
}

aclError AclApi::ACLAPI_AclRtResetDevice(int32_t deviceId)
{
    if (aclrtResetDeviceFunc == nullptr)
    {
        throw std::runtime_error("API aclrtResetDevice does not have a definition.");
    }
    return aclrtResetDeviceFunc(deviceId);
}

}
