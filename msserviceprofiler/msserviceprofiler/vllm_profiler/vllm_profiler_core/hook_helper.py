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
import importlib
import inspect
import logging


class HookHelper:
    def __init__(self, ori_function_define, new_function):
        self.new_function = new_function
        self.ori_function = None
        self.location = None
        self.attr_name = None
        if ori_function_define is None:
            return
        elif inspect.isfunction(ori_function_define) or inspect.ismethod(ori_function_define):
            self.ori_function = ori_function_define
            self.location, self.attr_name = self.get_location(self.ori_function)
        elif callable(ori_function_define):
            self.ori_function = ori_function_define.__call__
            self.location, self.attr_name = self.get_location(self.ori_function)

        if not all((self.ori_function, self.location, self.attr_name, self.new_function)):
            warn_msg = "{} replace failed.".format(ori_function_define)
            logging.error(warn_msg)
            raise ValueError(warn_msg)

    @staticmethod
    def get_location(function_ins):
        if not hasattr(function_ins, "__module__"):
            warning_msg = "function {} do not has attr __module__.".format(str(function_ins))
            logging.error(warning_msg)
            raise ValueError(warning_msg)
        module = importlib.import_module(function_ins.__module__)
        qualified_name = function_ins.__qualname__.split(".")
        classes = qualified_name[:-1]
        attr_name = qualified_name[-1]
        location = module
        for class_name in classes:
            location = getattr(location, class_name, None)
            if location is None:
                break

        if location is None:
            warning_msg = "{} do not exists".format(".".join(classes))
            logging.error(warning_msg)
            raise ValueError(warning_msg)
        return location, attr_name

    def replace(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.new_function))

    def recover(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.ori_function))

    def _get_method(self, func):
        if inspect.isclass(self.location):
            func_cls_name = inspect.getattr_static(self.location, self.attr_name).__class__.__name__
            if func_cls_name in ("staticmethod", "classmethod"):
                return staticmethod(func)
        return func
