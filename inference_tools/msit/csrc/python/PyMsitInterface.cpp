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

#include <Python.h>
#include "PyACLActuator.h"
#include "PyLog.h"

namespace MSIT_C {

PyDoc_STRVAR(MsitInferfaceModuleDoc,
    "The part of the module msit acl that is implemented in CXX.\n\
...");

static struct PyModuleDef g_MsitInterfaceModule = {
    PyModuleDef_HEAD_INIT,
    "msit_c",               /*   m_name    */
    MsitInferfaceModuleDoc, /*   m_doc     */
    -1,                     /*   m_size    */
    nullptr,                /*   m_methods */
};

}

PyMODINIT_FUNC PyInit_msit_c(void)
{
    PyObject *m = PyModule_Create(&MSIT_C::g_MsitInterfaceModule);
    if (m == nullptr)
    {
        return nullptr;
    }

    PyObject *cpyACLActuator = MSIT_C::GetACLActuatorModule();
    if (cpyACLActuator == nullptr)
    {
        PyErr_SetString(PyExc_ImportError, "Failed to create submodule ACLActuatorModule.");
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "acl", cpyACLActuator) < 0)
    {
        PyErr_SetString(PyExc_ImportError, "Failed to bind submodule ACLActuatorModule.");
        Py_DECREF(m);
        return nullptr;
    }
    Py_INCREF(cpyACLActuator);

    PyObject *cpyLog = MSIT_C::GetLogModule();
    if (cpyLog == nullptr)
    {
        PyErr_SetString(PyExc_ImportError, "Failed to create submodule LogModule.");
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "log", cpyLog) < 0)
    {
        PyErr_SetString(PyExc_ImportError, "Failed to bind submodule LogModule.");
        Py_DECREF(m);
        return nullptr;
    }
    Py_INCREF(cpyLog);

    return m;
}
