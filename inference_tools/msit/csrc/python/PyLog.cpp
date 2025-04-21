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

#include "PyLog.h"
#include "utils/log.h"

namespace MSIT_C {

const uint8_t PRINT_LOG_ARGS_SIZE = 2;

PyDoc_STRVAR(LogModuleDoc,
    "The part of the module log that is implemented in CXX.\n\
...");

static PyObject *Print_Log(PyObject *module, PyObject *args)
{
    int logLevel;
    const char *msg;
    if (args == nullptr || PyTuple_GET_SIZE(args) != PRINT_LOG_ARGS_SIZE)
    {
        PyErr_SetString(PyExc_TypeError, "\'Print_Log\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTuple(args, "is", &logLevel, &msg))
    {
        PyErr_SetString(PyExc_TypeError, "\'Print_Log\' should input a integer and a string.");
        Py_RETURN_NONE;
    }

    switch (logLevel)
    {
    case static_cast<int>(Utility::LogLevel::DEBUG):
        DEBUG_LOG(msg);
        break;
    case static_cast<int>(Utility::LogLevel::INFO):
        INFO_LOG(msg);
        break;
    case static_cast<int>(Utility::LogLevel::WARNING):
        WARNING_LOG(msg);
        break;
    case static_cast<int>(Utility::LogLevel::ERROR):
        ERROR_LOG(msg);
        break;
    default:
        break;
    }

    Py_RETURN_NONE;
}

static PyObject *Set_Log_Level(PyObject *module, PyObject *arg)
{
    if (!PyLong_Check(arg))
    {
        PyErr_SetString(PyExc_TypeError, "\"LogLevel\" should be a integer.");
        Py_RETURN_NONE;
    }

    int logLevel = static_cast<int>(PyLong_AsLong(arg));
    if (PyErr_Occurred())
    {
        Py_RETURN_NONE;
    }

    Utility::Log::GetInstance().SetLogLevel(logLevel);
    Py_RETURN_NONE;
}

static PyObject *Get_Log_Level(PyObject *module)
{
    int logLevel = Utility::Log::GetInstance().GetLogLevel();
    return PyLong_FromLong(logLevel);
}

static PyMethodDef LogMethods[] = {
    {"print_log", reinterpret_cast<PyCFunction>(Print_Log), METH_VARARGS, nullptr},
    {"set_log_level", reinterpret_cast<PyCFunction>(Set_Log_Level), METH_O, nullptr},
    {"get_log_level", reinterpret_cast<PyCFunction>(Get_Log_Level), METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef g_LogModule = {
    PyModuleDef_HEAD_INIT,
    "msit_c.log", /*   m_name    */
    LogModuleDoc, /*   m_doc     */
    -1,           /*   m_size    */
    LogMethods,   /*   m_methods */
};

PyObject *GetLogModule()
{
    return PyModule_Create(&g_LogModule);
}

}
