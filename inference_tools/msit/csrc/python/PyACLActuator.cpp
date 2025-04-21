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

#include "PyACLActuator.h"
#include "acl/include/AclApi.h"

namespace MSIT_C {

PyDoc_STRVAR(ACLInferfaceModuleDoc,
    "The part of the module acl actuator that is implemented in CXX.\n\
...");

static PyObject *AclApi_AclInit(PyObject *module)
{
    int ret = CALL_ACL_API(AclInit);
    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtSetDevice(PyObject *module, PyObject *arg)
{
    int ret = -1;
    if (!PyLong_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"deviceId\" should be a integer.");
        return PyLong_FromLong(ret);
    }

    int32_t num = static_cast<int32_t>(PyLong_AsLong(arg));
    if (PyErr_Occurred())
    {
        return PyLong_FromLong(ret);
    }
    ret = CALL_ACL_API(AclRtSetDevice, num);
    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtCreateContext(PyObject *module, PyObject *arg)
{
    int ret = -1;
    aclrtContext context;
    PyObject *pyContext = nullptr;
    if (!PyLong_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"deviceId\" should be a integer.");
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    int32_t num = static_cast<int32_t>(PyLong_AsLong(arg));
    if (PyErr_Occurred())
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }
    ret = CALL_ACL_API(AclRtCreateContext, &context, num);
    pyContext = PyCapsule_New(context, "aclrtContext", nullptr);
    if (pyContext == nullptr)
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    return Py_BuildValue("(Oi)", pyContext, ret);
}

static PyObject *AclApi_AclMdlLoadFromFile(PyObject *module, PyObject *arg)
{
    int ret = -1;
    if (!PyUnicode_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"modelPath\" should be a string.");
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    const char *path = PyUnicode_AsUTF8(arg);
    if (path == nullptr)
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    LoadFileResult cTuple = CALL_ACL_API(AclMdlLoadFromFile, path);
    return Py_BuildValue("(Ii)", cTuple.modelId, cTuple.ret);
}

static PyObject *AclApi_AclMdlCreateDesc(PyObject *module)
{
    aclmdlDesc *modelDesc = CALL_ACL_API(AclMdlCreateDesc);
    PyObject *pyModelDesc = PyCapsule_New(modelDesc, "aclmdlDesc", nullptr);
    if (pyModelDesc == nullptr)
    {
        Py_RETURN_NONE;
    }
    return pyModelDesc;
}

static PyObject *AclApi_AclMdlGetDesc(PyObject *module, PyObject *args)
{
    int ret = -1;
    /* 预期2个参数，modelDesc和modelId */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetDesc\' expects 2 arguments.");
        return PyLong_FromLong(ret);
    }

    PyObject *pyModelDesc = nullptr;
    uint32_t modelId;
    if (!PyArg_ParseTuple(args, "OI", &pyModelDesc, &modelId))
    {
        return PyLong_FromLong(ret);
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    ret = CALL_ACL_API(AclMdlGetDesc, modelDesc, modelId);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlGetNumInputs(PyObject *module, PyObject *args)
{
    size_t inputSize;
    PyObject *pyModelDesc = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyModelDesc))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    inputSize = CALL_ACL_API(AclMdlGetNumInputs, modelDesc);

    return PyLong_FromSize_t(inputSize);
}

static PyObject *AclApi_AclMdlGetNumOutputs(PyObject *module, PyObject *args)
{
    size_t outputSize;
    PyObject *pyModelDesc = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyModelDesc))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    outputSize = CALL_ACL_API(AclMdlGetNumOutputs, modelDesc);

    return PyLong_FromSize_t(outputSize);
}

static PyObject *AclApi_AclMdlGetInputNameByIndex(PyObject *module, PyObject *args)
{
    const char *name;
    /* 预期2个参数，modelDesc和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetInputNameByIndex\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyModelDesc = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyModelDesc, &index))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    name = CALL_ACL_API(AclMdlGetInputNameByIndex, modelDesc, index);

    return PyUnicode_FromString(name);
}

static PyObject *AclApi_AclMdlGetInputSizeByIndex(PyObject *module, PyObject *args)
{
    size_t inputSize;
    /* 预期2个参数，modelDesc和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetInputSizeByIndex\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyModelDesc = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyModelDesc, &index))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    inputSize = CALL_ACL_API(AclMdlGetInputSizeByIndex, modelDesc, index);

    return PyLong_FromSize_t(inputSize);
}

static PyObject *AclApi_AclMdlGetOutputSizeByIndex(PyObject *module, PyObject *args)
{
    size_t outputSize;
    /* 预期2个参数，modelDesc和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetOutputSizeByIndex\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyModelDesc = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyModelDesc, &index))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    outputSize = CALL_ACL_API(AclMdlGetOutputSizeByIndex, modelDesc, index);

    return PyLong_FromSize_t(outputSize);
}

static PyObject *AclApi_AclMdlGetInputDataType(PyObject *module, PyObject *args)
{
    /* 预期2个参数，modelDesc和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetInputDataType\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyModelDesc = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyModelDesc, &index))
    {
        Py_RETURN_NONE;
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    int type = CALL_ACL_API(AclMdlGetInputDataType, modelDesc, index);

    return PyLong_FromLong(type);
}

static PyObject *AclApi_AclMdlGetInputDims(PyObject *module, PyObject *args)
{
    int ret = -1;
    /* 预期2个参数，modelDesc和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetInputDims\' expects 2 arguments.");
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    PyObject *pyModelDesc = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyModelDesc, &index))
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    aclmdlIODims ioDims;
    ret = CALL_ACL_API(AclMdlGetInputDims, modelDesc, index, &ioDims);

    PyObject *dimsDict = PyDict_New();
    PyObject *pyDims = PyList_New(ioDims.dimCount);
    for (size_t i = 0; i < ioDims.dimCount; ++i)
    {
        PyList_SET_ITEM(pyDims, i, PyLong_FromSize_t(ioDims.dims[i]));
    }

    PyDict_SetItemString(dimsDict, "name", PyUnicode_FromString(ioDims.name));
    PyDict_SetItemString(dimsDict, "dimCount", PyLong_FromSize_t(ioDims.dimCount));
    PyDict_SetItemString(dimsDict, "dims", pyDims);

    return Py_BuildValue("(Oi)", dimsDict, ret);
}

static PyObject *AclApi_AclRtMalloc(PyObject *module, PyObject *args)
{
    int ret = -1;
    void *ptr = nullptr;
    size_t bufferSize;
    PyObject *pyMallocPtr = nullptr;
    if (!PyArg_ParseTuple(args, "K", &bufferSize))
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }

    ret = CALL_ACL_API(AclRtMalloc, &ptr, bufferSize);
    pyMallocPtr = PyCapsule_New(ptr, "aclrtMallocPtr", nullptr);
    if (pyMallocPtr == nullptr)
    {
        return Py_BuildValue("(Oi)", Py_None, ret);
    }
    return Py_BuildValue("(Oi)", pyMallocPtr, ret);
}

static PyObject *AclApi_AclMdlCreateDataset(PyObject *module)
{
    aclmdlDataset *dataset = CALL_ACL_API(AclMdlCreateDataset);
    PyObject *pyDataset = PyCapsule_New(dataset, "aclmdlDataset", nullptr);
    if (pyDataset == nullptr)
    {
        Py_RETURN_NONE;
    }
    return pyDataset;
}

static PyObject *AclApi_AclCreateDataBuffer(PyObject *module, PyObject *args)
{
    /* 预期2个参数，ptr和buffersize */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclCreateDataBuffer\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyMallocPtr = nullptr;
    size_t bufferSize;
    if (!PyArg_ParseTuple(args, "OK", &pyMallocPtr, &bufferSize))
    {
        Py_RETURN_NONE;
    }

    void *ptr = reinterpret_cast<void *>(PyCapsule_GetPointer(pyMallocPtr, "aclrtMallocPtr"));
    aclDataBuffer *dataBuffer = CALL_ACL_API(AclCreateDataBuffer, ptr, bufferSize);

    PyObject *pyDataBuffer = PyCapsule_New(dataBuffer, "aclDataBuffer", nullptr);
    if (pyDataBuffer == nullptr)
    {
        Py_RETURN_NONE;
    }
    return pyDataBuffer;
}

static PyObject *AclApi_AclMdlAddDatasetBuffer(PyObject *module, PyObject *args)
{
    int ret = -1;
    /* 预期2个参数，dataset和buffer */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlAddDatasetBuffer\' expects 2 arguments.");
        return PyLong_FromLong(ret);
    }

    PyObject *pyDataset = nullptr;
    PyObject *pyDataBuffer = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pyDataset, &pyDataBuffer))
    {
        return PyLong_FromLong(ret);
    }

    aclDataBuffer *buffer = reinterpret_cast<aclDataBuffer *>(PyCapsule_GetPointer(pyDataBuffer, "aclDataBuffer"));
    aclmdlDataset *dataset = reinterpret_cast<aclmdlDataset *>(PyCapsule_GetPointer(pyDataset, "aclmdlDataset"));
    ret = CALL_ACL_API(AclMdlAddDatasetBuffer, dataset, buffer);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlExecute(PyObject *module, PyObject *args)
{
    int ret = -1;
    /* 预期3个参数，modelId, inputDataset和outputDataset */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 3)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlExecute\' expects 3 arguments.");
        return PyLong_FromLong(ret);
    }

    PyObject *pyInputDataset = nullptr;
    PyObject *pyOutputDataset = nullptr;
    uint32_t modelId;
    if (!PyArg_ParseTuple(args, "IOO", &modelId, &pyInputDataset, &pyOutputDataset))
    {
        return PyLong_FromLong(ret);
    }

    aclmdlDataset *inputDataset = reinterpret_cast<aclmdlDataset *>(PyCapsule_GetPointer(pyInputDataset, "aclmdlDataset"));
    aclmdlDataset *outputDataset = reinterpret_cast<aclmdlDataset *>(PyCapsule_GetPointer(pyOutputDataset, "aclmdlDataset"));
    ret = CALL_ACL_API(AclMdlExecute, modelId, inputDataset, outputDataset);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtMemcpy(PyObject *module, PyObject *args)
{
    int ret = -1;
    /* 预期5个参数，dest, destMax, src, count, kind */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 5)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclrtMemcpy\' expects 5 arguments.");
        return PyLong_FromLong(ret);
    }

    PyObject *pyDst = nullptr;
    PyObject *pySrc = nullptr;
    size_t destCount;
    size_t srcCount;
    int kind;
    if (!PyArg_ParseTuple(args, "OKOKi", &pyDst, &destCount, &pySrc, &srcCount, &kind))
    {
        return PyLong_FromLong(ret);
    }

    const void *src = nullptr;
    if (PyCapsule_CheckExact(pySrc))
    {
        src = reinterpret_cast<const void *>(PyCapsule_GetPointer(pySrc, "aclrtMallocPtr"));
    }
    else if (PyBytes_Check(pySrc))
    {
        src = reinterpret_cast<const void *>(PyBytes_AsString(pySrc));
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\'pySrc\' invalid type.");
    }

    void *dst = reinterpret_cast<void *>(PyCapsule_GetPointer(pyDst, "aclrtMallocPtr"));
    aclrtMemcpyKind cKind = static_cast<aclrtMemcpyKind>(kind);
    ret = CALL_ACL_API(AclRtMemcpy, dst, destCount, src, srcCount, cKind);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlGetDatasetNumBuffers(PyObject *module, PyObject *args)
{
    size_t bufferNum;
    PyObject *pyDataset = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyDataset))
    {
        Py_RETURN_NONE;
    }

    const aclmdlDataset *dataset = reinterpret_cast<const aclmdlDataset *>(PyCapsule_GetPointer(pyDataset, "aclmdlDataset"));
    bufferNum = CALL_ACL_API(AclMdlGetDatasetNumBuffers, dataset);

    return PyLong_FromSize_t(bufferNum);
}

static PyObject *AclApi_AclMdlGetDatasetBuffer(PyObject *module, PyObject *args)
{
    /* 预期2个参数，dataset和index */
    if (args == nullptr || PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "\'aclmdlGetDatasetBuffer\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject *pyDataset = nullptr;
    size_t index;
    if (!PyArg_ParseTuple(args, "OK", &pyDataset, &index))
    {
        Py_RETURN_NONE;
    }

    const aclmdlDataset *dataset = reinterpret_cast<const aclmdlDataset *>(PyCapsule_GetPointer(pyDataset, "aclmdlDataset"));
    aclDataBuffer *dataBuffer = CALL_ACL_API(AclMdlGetDatasetBuffer, dataset, index);

    PyObject *pyDataBuffer = PyCapsule_New(dataBuffer, "aclDataBuffer", nullptr);
    if (pyDataBuffer == nullptr)
    {
        Py_RETURN_NONE;
    }
    return pyDataBuffer;
}

static PyObject *AclApi_AclDestroyDataBuffer(PyObject *module, PyObject *args)
{
    int ret = -1;

    PyObject *pyDataBuffer = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyDataBuffer))
    {
        return PyLong_FromLong(ret);
    }

    const aclDataBuffer *buffer = reinterpret_cast<const aclDataBuffer *>(PyCapsule_GetPointer(pyDataBuffer, "aclDataBuffer"));
    ret = CALL_ACL_API(AclDestroyDataBuffer, buffer);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlDestroyDataset(PyObject *module, PyObject *args)
{
    int ret = -1;

    PyObject *pyDataset = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyDataset))
    {
        return PyLong_FromLong(ret);
    }

    const aclmdlDataset *dataset = reinterpret_cast<const aclmdlDataset *>(PyCapsule_GetPointer(pyDataset, "aclmdlDataset"));
    ret = CALL_ACL_API(AclMdlDestroyDataset, dataset);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtFree(PyObject *module, PyObject *args)
{
    int ret = -1;

    PyObject *pyMallocPtr = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyMallocPtr))
    {
        return PyLong_FromLong(ret);
    }

    void *ptr = reinterpret_cast<void *>(PyCapsule_GetPointer(pyMallocPtr, "aclrtMallocPtr"));
    ret = CALL_ACL_API(AclRtFree, ptr);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclFinalize(PyObject *module)
{
    int ret = CALL_ACL_API(AclFinalize);
    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlUnload(PyObject *module, PyObject *arg)
{
    int ret = -1;
    if (!PyLong_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"modelId\" should be a integer.");
        return PyLong_FromLong(ret);
    }

    int32_t num = static_cast<int32_t>(PyLong_AsLong(arg));
    if (PyErr_Occurred())
    {
        return PyLong_FromLong(ret);
    }
    ret = CALL_ACL_API(AclMdlUnload, num);
    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclMdlDestroyDesc(PyObject *module, PyObject *args)
{
    int ret = -1;

    PyObject *pyModelDesc = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyModelDesc))
    {
        return PyLong_FromLong(ret);
    }

    aclmdlDesc *modelDesc = reinterpret_cast<aclmdlDesc *>(PyCapsule_GetPointer(pyModelDesc, "aclmdlDesc"));
    ret = CALL_ACL_API(AclMdlDestroyDesc, modelDesc);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtDestroyContext(PyObject *module, PyObject *args)
{
    int ret = -1;

    PyObject *pyContext = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pyContext))
    {
        return PyLong_FromLong(ret);
    }

    aclrtContext modelDesc = reinterpret_cast<aclrtContext>(PyCapsule_GetPointer(pyContext, "aclrtContext"));
    ret = CALL_ACL_API(AclRtDestroyContext, modelDesc);

    return PyLong_FromLong(ret);
}

static PyObject *AclApi_AclRtResetDevice(PyObject *module, PyObject *arg)
{
    int ret = -1;
    if (!PyLong_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"deviceId\" should be a integer.");
        return PyLong_FromLong(ret);
    }

    int32_t num = static_cast<int32_t>(PyLong_AsLong(arg));
    if (PyErr_Occurred())
    {
        return PyLong_FromLong(ret);
    }
    ret = CALL_ACL_API(AclRtResetDevice, num);
    return PyLong_FromLong(ret);
}

static PyMethodDef ACLActuatorMethods[] = {
    {"init", reinterpret_cast<PyCFunction>(AclApi_AclInit), METH_NOARGS, nullptr},
    {"rt_set_device", reinterpret_cast<PyCFunction>(AclApi_AclRtSetDevice), METH_O, nullptr},
    {"rt_create_context", reinterpret_cast<PyCFunction>(AclApi_AclRtCreateContext), METH_O, nullptr},
    {"load_from_file", reinterpret_cast<PyCFunction>(AclApi_AclMdlLoadFromFile), METH_O, nullptr},
    {"create_desc", reinterpret_cast<PyCFunction>(AclApi_AclMdlCreateDesc), METH_NOARGS, nullptr},
    {"get_desc", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetDesc), METH_VARARGS, nullptr},
    {"get_num_inputs", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetNumInputs), METH_VARARGS, nullptr},
    {"get_num_outputs", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetNumOutputs), METH_VARARGS, nullptr},
    {"get_input_name_by_index", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetInputNameByIndex), METH_VARARGS, nullptr},
    {"get_input_size_by_index", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetInputSizeByIndex), METH_VARARGS, nullptr},
    {"get_output_size_by_index", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetOutputSizeByIndex), METH_VARARGS, nullptr},
    {"get_input_data_type", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetInputDataType), METH_VARARGS, nullptr},
    {"get_input_dims", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetInputDims), METH_VARARGS, nullptr},
    {"rt_malloc", reinterpret_cast<PyCFunction>(AclApi_AclRtMalloc), METH_VARARGS, nullptr},
    {"create_dataset", reinterpret_cast<PyCFunction>(AclApi_AclMdlCreateDataset), METH_NOARGS, nullptr},
    {"create_databuffer", reinterpret_cast<PyCFunction>(AclApi_AclCreateDataBuffer), METH_VARARGS, nullptr},
    {"add_dataset_buffer", reinterpret_cast<PyCFunction>(AclApi_AclMdlAddDatasetBuffer), METH_VARARGS, nullptr},
    {"execute", reinterpret_cast<PyCFunction>(AclApi_AclMdlExecute), METH_VARARGS, nullptr},
    {"rt_memcpy", reinterpret_cast<PyCFunction>(AclApi_AclRtMemcpy), METH_VARARGS, nullptr},
    {"get_dataset_num_buffers", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetDatasetNumBuffers), METH_VARARGS, nullptr},
    {"get_dataset_buffer", reinterpret_cast<PyCFunction>(AclApi_AclMdlGetDatasetBuffer), METH_VARARGS, nullptr},
    {"destroy_databuffer", reinterpret_cast<PyCFunction>(AclApi_AclDestroyDataBuffer), METH_VARARGS, nullptr},
    {"destroy_dataset", reinterpret_cast<PyCFunction>(AclApi_AclMdlDestroyDataset), METH_VARARGS, nullptr},
    {"rt_free", reinterpret_cast<PyCFunction>(AclApi_AclRtFree), METH_VARARGS, nullptr},
    {"finalize", reinterpret_cast<PyCFunction>(AclApi_AclFinalize), METH_NOARGS, nullptr},
    {"unload", reinterpret_cast<PyCFunction>(AclApi_AclMdlUnload), METH_O, nullptr},
    {"destroy_desc", reinterpret_cast<PyCFunction>(AclApi_AclMdlDestroyDesc), METH_VARARGS, nullptr},
    {"rt_destroy_context", reinterpret_cast<PyCFunction>(AclApi_AclRtDestroyContext), METH_VARARGS, nullptr},
    {"rt_reset_device", reinterpret_cast<PyCFunction>(AclApi_AclRtResetDevice), METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef g_ACLActuatorModule = {
    PyModuleDef_HEAD_INIT,
    "msit_c.acl",          /*   m_name    */
    ACLInferfaceModuleDoc, /*   m_doc     */
    -1,                    /*   m_size    */
    ACLActuatorMethods,    /*   m_methods */
};

PyObject *GetACLActuatorModule()
{
    return PyModule_Create(&g_ACLActuatorModule);
}

}
