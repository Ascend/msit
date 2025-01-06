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

import importlib.util
import sys
import importlib.abc
from abc import ABC, abstractmethod
import uuid


REPLACE = 0
PRE_HOOK = 1
POST_HOOK = 2

def hijack(*, stub: callable, module: str, cls: str='', function: str='', action=REPLACE, priority: int=100) -> str:
    '''
    Hijack module-import process or function execution process.
    Support attaching pre/post hooks to the process, or replacing function implementations.

    .. target::
        When only set 'module': module
        When set 'module' and 'function': function in module
        When set 'module', 'cls' and 'function': function in class

    .. warning::
        The pre-hook of the module-import process will only take effect if it is set before the module is imported.
        If the module is modified in its post-hook, the impact cannot be restored even if the hijacking is released.

    Parameters
    ----------
    stub: Callable object.
        Follow different format under different target and action.
        ---------------------------------------------------------------------------------------------------------------
        |  target  |  action   |           format             |                   description                         |
        |-------------------------------------------------------------------------------------------------------------|
        | module   | pre-hook  | callable()                   | Called before module import.                          |
        |-------------------------------------------------------------------------------------------------------------|
        | module   | post-hook | callable(m)                  | Called after module import. 'm' is the module.        |
        |-------------------------------------------------------------------------------------------------------------|
        | function | replace   | ret = callable(*args, **kws) | Replace original object.                              |
        |-------------------------------------------------------------------------------------------------------------|
        | function | pre-hook  | args, kws =                  | Called before function execution, and the return will |
        |          |           |    callable(*args)           | replace original input of the target function.        |
        |-------------------------------------------------------------------------------------------------------------|
        | function | post-hook | ret = callable(ret, *args,   | Called after function execution, and the return will  |
        |          |           |           **kws)             | replace original return of target function.           |
        ---------------------------------------------------------------------------------------------------------------
    module: str
        Full name of target module.
    cls: str, optional
        Full name of target class.
    function: str, optional
        Name of target function.
    action: enum, optional
        Choose between REPLACE, PRE_HOOK, and POST_HOOK.
    priority: int, optional
        The smaller the value is, the higher the priority is. When multiple hooks are set on the same target, they will
        be excuted by priority.

    Returns
    -------
    hander:
        Handler to a hijacking.
    '''
    HiJackerManager.initialize()
    return HiJackerManager.add_unit(HijackerUnit(stub, module, cls, function, action, priority))

def release(handler):
    '''
    Cancel a hijacking. 'handler' is returned by function 'hijack'.
    '''
    HiJackerManager.remove_unit(handler)

class HijackerUnit:
    def __init__(self, stub, module, cls, function, action, priority):
        self.stub = stub
        self.module = module
        self.cls = cls
        self.function = function
        self.action = action
        self.priority = priority
        self.target = f'{module}-{cls}-{function}'

        self._check_para_valid()

    def _check_para_valid(self):
        if not isinstance(self.module, str):
            raise TypeError("\'module\' should be a str")
        if not self.module:
            raise ValueError("\'module\' is required")
        if self.cls and not isinstance(self.cls, str):
            raise TypeError("\'cls\' should be a str")
        if self.function and not isinstance(self.function, str):
            raise TypeError("\'function\' should be a str")
        if self.action not in (REPLACE, PRE_HOOK, POST_HOOK):
            raise ValueError("\'action\' should be REPLACE, PRE_HOOK, or POST_HOOK")
        if not isinstance(self.priority, int):
            raise TypeError("\'priority\' should be an int")
        if self.cls and not self.function:
            raise ValueError("\'function\' should be used when \'cls\' used")

        if not self.cls and not self.function and self.action == REPLACE:
            raise NotImplementedError("replacement of a module is not supported")
        if not callable(self.stub):
            raise TypeError("\'stub\' should be callable")

class HiJackerWrapperObj(ABC):
    def __init__(self, name):
        self.name = name
        self.ori_obj = None
        self.replacement = []
        self.pre_hooks = []
        self.post_hooks = []
        self.mod_name, self.class_name, self.func_name = name.split('-')

    def add_unit(self, unit):
        if unit.action == REPLACE:
            self.replacement.append(unit)
            self.replacement.sort(key=lambda x : x.priority)
        elif unit.action == PRE_HOOK:
            self.pre_hooks.append(unit)
            self.pre_hooks.sort(key=lambda x : x.priority)
        else:
            self.post_hooks.append(unit)
            self.post_hooks.sort(key=lambda x : x.priority)

    def remove_unit(self, unit):
        if unit.action == REPLACE:
            self.replacement.remove(unit)
        elif unit.action == PRE_HOOK:
            self.pre_hooks.remove(unit)
        else:
            self.post_hooks.remove(unit)

    def is_empty(self):
        return not self.replacement and not self.pre_hooks and not self.post_hooks

    def set_ori_obj(self, obj):
        self.ori_obj = obj

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass

class HiJackerWrapperModule(HiJackerWrapperObj):
    def __init__(self, name):
        super().__init__(name)

    def exec_pre_hook(self):
        for unit in self.pre_hooks:
            unit.stub()

    def exec_post_hook(self, m):
        self.set_ori_obj(m)
        for unit in self.post_hooks:
            unit.stub(m)

    def add_unit(self, unit):
        super().add_unit(unit)
        if unit.action == POST_HOOK:
            m = sys.modules.get(self.mod_name)
            if m:
                unit.stub(m)

    def activate(self):
        HiJackerPathFinder.add_mod(self.mod_name)

    def deactivate(self):
        HiJackerPathFinder.remove_mod(self.mod_name)

class HiJackerWrapperFunction(HiJackerWrapperObj):
    def __init__(self, name):
        super().__init__(name)
        self.mod_hijacker = None

    def activate(self):
        def replace_closure(class_name, func_name, wrapper):
            def modify_module(m):
                parent_obj = m
                class_chain = class_name.split('.') if class_name else []
                for c in class_chain:
                    if not hasattr(parent_obj, c):
                        return
                    parent_obj = getattr(parent_obj, c)
                if parent_obj and hasattr(parent_obj, func_name):
                    ori_obj = getattr(parent_obj, func_name)
                    self.set_ori_obj(ori_obj)
                    setattr(parent_obj, func_name, wrapper)
                return
            return modify_module

        self.mod_hijacker = hijack(stub = replace_closure(self.class_name, self.func_name, self._get_wrapper()),
                                   module = self.mod_name,
                                   action=POST_HOOK,
                                   priority=0)
        return

    def deactivate(self):
        if self.mod_hijacker:
            release(self.mod_hijacker)
            self.mod_hijacker = None
        mod = sys.modules.get(self.mod_name)
        if mod and self.ori_obj:
            parent_obj = mod
            class_chain = self.class_name.split('.') if self.class_name else []
            for c in class_chain:
                if not hasattr(parent_obj, c):
                    self.ori_obj = None
                    return
                parent_obj = getattr(parent_obj, c)
            if parent_obj and hasattr(parent_obj, self.func_name):
                setattr(parent_obj, self.func_name, self.ori_obj)
        self.ori_obj = None
        return

    def _get_wrapper(self):
        def wrapper(*args, **kws):
            if not self.ori_obj:
                raise RuntimeError
            for unit in self.pre_hooks:
                args, kws = unit.stub(*args, **kws)
            f = self.ori_obj
            if self.replacement:
                f = self.replacement[0].stub
            ret = f(*args, **kws)
            for unit in self.post_hooks:
                ret = unit.stub(ret, *args, **kws)
            return ret
        return wrapper

class HiJackerManager:
    _initialized = False
    _hijacker_units = {}
    _hijacker_wrappers = {}

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        sys.meta_path.insert(0, HiJackerPathFinder())
        cls._initialized = True

    @classmethod
    def add_unit(cls, unit):
        handler = uuid.uuid4().hex
        cls._hijacker_units[handler] = unit
        wrapper_obj = cls._hijacker_wrappers.get(unit.target)
        if not wrapper_obj:
            wrapper_obj = cls._build_wrapper_obj(unit.target)
            cls._hijacker_wrappers[unit.target] = wrapper_obj
            wrapper_obj.activate()
        wrapper_obj.add_unit(unit)
        return handler

    @classmethod
    def remove_unit(cls, handler):
        unit = cls._hijacker_units.get(handler)
        if not unit:
            return
        wrapper_obj = cls._hijacker_wrappers.get(unit.target)
        wrapper_obj.remove_unit(unit)
        if wrapper_obj.is_empty():
            wrapper_obj.deactivate()
            del cls._hijacker_wrappers[unit.target]
        del cls._hijacker_units[handler]

    @classmethod
    def get_module_wrapper(cls, name):
        return cls._hijacker_wrappers.get(f'{name}--')

    @classmethod
    def _build_wrapper_obj(cls, name):
        m, c, f = name.split('-')
        if f:
            return HiJackerWrapperFunction(name)
        else:
            return HiJackerWrapperModule(name)


class HiJackerPathFinder(importlib.abc.MetaPathFinder):
    _modules_of_insterest = set()

    @classmethod
    def add_mod(cls, name):
        cls._modules_of_insterest.add(name)

    @classmethod
    def remove_mod(cls, name):
        cls._modules_of_insterest.discard(name)

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._modules_of_insterest:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, HiJackerPathFinder):
                continue
            spec = finder.find_spec(fullname, path, target)
            if not spec:
                continue
            return importlib.util.spec_from_loader(fullname, HiJackerLoader(spec))
        return None

    def find_module(self, fullname, path=None):
        if fullname not in self._modules_of_insterest:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, HiJackerPathFinder):
                continue
            loader = finder.find_module(fullname, path)
            if not loader:
                continue
            return HiJackerLoader(importlib.util.spec_from_loader(fullname, loader))
        return None

class HiJackerLoader(importlib.abc.Loader):
    def __init__(self, ori_spec):
        self.ori_spec = ori_spec

    def create_module(self, spec):
        module = importlib.util.module_from_spec(self.ori_spec)
        return module

    def load_module(self, fullname):
        module = self.ori_spec.loader.load_module(fullname)
        return module

    def exec_module(self, module):
        wrapper = HiJackerManager.get_module_wrapper(module.__name__)
        if wrapper:
            wrapper.exec_pre_hook()
        self.ori_spec.loader.exec_module(module)
        if wrapper:
            wrapper.exec_post_hook(module)
