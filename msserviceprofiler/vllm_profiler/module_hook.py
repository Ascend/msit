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

import traceback
import importlib
import inspect
import functools
import traceback
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional, Callable, Any
from packaging.version import Version
from .logger import logger
from .registry import get_hook_registry, add_to_hook_registry


def import_object_from_string(import_path: str, module_path: str) -> Any:
    """根据点分隔的路径导入对象（模块、类、函数等）。
    
    支持多级嵌套属性，如 "module.submodule.ClassName.method_name"
    
    Args:
        import_path: 模块导入路径
        module_path: 对象在模块中的路径
        
    Returns:
        Any: 导入的对象，失败时返回 None
    """
    if not import_path:
        logger.error(f"Module import_path is empty")
        return None

    try:
        module = importlib.import_module(import_path)
    except ImportError as e:
        logger.error(f"Module import failed for %s: %s", module_path, e)
        return None

    current = module
    for part in module_path.split("."):
        prev = current  # For log
        current = getattr(current, part, None)
        if current is None:
            logger.error(f"Module {prev} doesn't has attribute {part}")
            return None
    return current


class HookHelper:
    """辅助类，用于函数替换操作。
    
    该类封装了函数替换的核心逻辑，包括：
    - 获取原始函数的位置信息
    - 执行函数替换
    - 恢复原始函数
    
    Attributes:
        new_function: 新的替换函数
        ori_function: 原始函数
        location: 函数所在的位置对象
        attr_name: 函数属性名称
    """
    
    def __init__(self, ori_function_define, new_function):
        """初始化 HookHelper。
        
        Args:
            ori_function_define: 原始函数定义
            new_function: 新的替换函数
        
        Raises:
            ValueError: 当初始化失败时抛出
        """
        self.new_function = new_function
        self.ori_function = None
        self.location = None
        self.attr_name = None

        if ori_function_define is None:
            return

        if inspect.isfunction(ori_function_define) or inspect.ismethod(ori_function_define):
            self.ori_function = ori_function_define
            self.location, self.attr_name = self.get_location(self.ori_function)
        elif callable(ori_function_define):
            self.ori_function = ori_function_define.__call__
            self.location, self.attr_name = self.get_location(self.ori_function)

        # Allow initialization without new_function; validation will occur in replace()/recover()
        if not all([self.ori_function, self.location, self.attr_name]):
            warn_msg = f"{ori_function_define} initialization failed."
            logger.error(warn_msg)
            raise ValueError(warn_msg)

    @staticmethod
    def get_location(function_ins):
        """获取函数的位置信息。
        
        Args:
            function_ins: 函数实例
            
        Returns:
            Tuple: (location, attr_name) 元组
            
        Raises:
            ValueError: 当无法获取位置信息时抛出
        """
        if not hasattr(function_ins, "__module__"):
            warning_msg = f"function {str(function_ins)} has no __module__."
            logger.error(warning_msg)
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
            warning_msg = f"{'.'.join(classes)} does not exist"
            logger.error(warning_msg)
            raise ValueError(warning_msg)

        return location, attr_name

    def replace(self):
        """执行函数替换操作。
        
        Raises:
            Exception: 当替换失败时抛出
        """
        if all([self.ori_function, self.location, self.attr_name, self.new_function]):
            try:
                setattr(self.location, self.attr_name, self._get_method(self.new_function))
                logger.debug(f"Successfully replaced {self.attr_name} in {self.location}")
            except Exception as e:
                logger.error(f"Failed to replace {self.attr_name} in {self.location}: {e}")
                raise

    def recover(self):
        """恢复原始函数。
        
        Raises:
            Exception: 当恢复失败时抛出
        """
        if all([self.ori_function, self.location, self.attr_name, self.new_function]):
            try:
                setattr(self.location, self.attr_name, self._get_method(self.ori_function))
                logger.debug(f"Successfully recovered {self.attr_name} in {self.location}")
            except Exception as e:
                logger.error(f"Failed to recover {self.attr_name} in {self.location}: {e}")
                raise

    def _get_method(self, func):
        """获取适当的方法类型（staticmethod 或 classmethod）。
        
        Args:
            func: 函数对象
            
        Returns:
            适当的方法类型包装
        """
        if inspect.isclass(self.location):
            try:
                func_cls_name = inspect.getattr_static(self.location, self.attr_name).__class__.__name__
                if func_cls_name in ("staticmethod", "classmethod"):
                    return staticmethod(func)
            except AttributeError:
                pass
        return func


def get_parents_name(ori_func, index=1):
    """获取调用栈中的父级函数名。
    
    Args:
        ori_func: 原始函数
        index: 调用栈索引，默认为 1
        
    Returns:
        Optional[str]: 父级函数名，失败时返回 None
    """
    gen = traceback.walk_stack(None)
    try:
        for _ in range(index + 1):
            f = next(gen)
        return f[0].f_code.co_name
    except StopIteration:
        return None


class VLLMHookerBase(ABC):
    """Hooker基类，提供核心功能。
    
    该类是所有 hooker 的基类，提供：
    - 版本支持检查
    - hook 应用和恢复
    - 异常处理和回退机制
    - 异步和同步函数支持
    
    Attributes:
        vllm_version (Tuple[Optional[str], Optional[str]]): 支持的 vLLM 版本范围
        applied_hook_func_name (str): 应用的 hook 函数名称
        hooks (List): hook 实例列表
        hook_list (List[Tuple[str, str]]): hook 点列表
        caller_filter (Optional[str]): 调用者过滤条件
        hook_func (Optional[Callable]): hook 处理函数
    """
    
    vllm_version = (None, None)  # (min_version, max_version)
    applied_hook_func_name = ""
    
    def __init__(self):
        """初始化 VLLMHookerBase。"""
        self.hooks = []
        # 记录文本形式的 hook 点（未解析成对象前的 (import_path, attr_path) 列表）
        self.hook_list: List[Tuple[str, str]] = []
        self.caller_filter: Optional[str] = None
        # 对应的 hook 处理函数，用于配置化时复用
        self.hook_func: Optional[Callable] = None
        
    @abstractmethod
    def init(self):
        """初始化hook点，子类必须实现。
        
        该方法由子类实现，用于初始化具体的 hook 点。
        """
        pass

    def replace_func(self, ori_func, pname, profiler_func, on_recover: Optional[Callable] = None):
        """创建替换函数，带失败回退机制。
        
        功能特性：
        - 任一 hook 执行抛异常时，不影响业务逻辑：返回原始函数结果
        - 连续失败达到 5 次，自动回退为原始函数（若提供 on_recover 则调用以执行恢复）
        - 支持 context manager 的特殊处理
        
        Args:
            ori_func: 原始函数
            pname: 调用者过滤名称
            profiler_func: profiler 函数
            on_recover: 恢复回调函数，可选
            
        Returns:
            Callable: 替换函数
        """
        # 检测函数类型
        is_context_manager_func = inspect.isgeneratorfunction(profiler_func)
        is_async_func = inspect.iscoroutinefunction(ori_func) or inspect.iscoroutinefunction(profiler_func)

        if is_async_func:
            return self._create_async_wrapper(ori_func, profiler_func, pname, on_recover, is_context_manager_func)
        else:
            return self._create_sync_wrapper(ori_func, profiler_func, pname, on_recover, is_context_manager_func)

    def do_hook(self, hook_points, profiler_func_maker, pname=None):
        """执行实际的hook操作。
        
        Args:
            hook_points: hook 点列表
            profiler_func_maker: profiler 函数生成器
            pname: 调用者过滤名称，可选
        """
        for ori_func in hook_points:
            if ori_func is None:
                continue
            profiler_func = profiler_func_maker(ori_func)
            cur_hook = HookHelper(ori_func, None)

            def _recover_current(cur_hook_ref=cur_hook):
                try:
                    cur_hook_ref.recover()
                except Exception as e:
                    logger.error(f"Recover call failed: {e}")
            
            wrapped = self.replace_func(ori_func, pname, profiler_func, on_recover=_recover_current)
            cur_hook.new_function = wrapped
            cur_hook.replace()
            self.hooks.append(cur_hook)
            logger.debug(f"replacing {ori_func} with {self.applied_hook_func_name}")

    def support_version(self, version):
        """检查当前版本是否支持。
        
        Args:
            version: 要检查的版本号
            
        Returns:
            bool: 如果支持则返回 True，否则返回 False
        """
        min_version = self.vllm_version[0]
        max_version = self.vllm_version[1]

        if min_version is not None and Version(min_version) > Version(version):
            logger.debug(f"min_version={min_version} is less than version={version}, skip")
            return False
        if max_version is not None and Version(max_version) < Version(version):
            logger.debug(f"max_version={max_version} is larger than version={version}, skip")
            return False
        return True

    def register(self):
        """注册hooker到全局注册表。"""
        add_to_hook_registry(self)

    def _log_hook_exception(self, ori_func, e: Exception, failures: int):
        """记录 hook 执行异常。
        
        Args:
            ori_func: 原始函数
            e: 异常对象
            failures: 失败次数
        """
        logger.error(f"Hook execution failed ({failures}/5) for {ori_func}: {e}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    def _handle_context_manager_failure(self, ori_func, on_recover: Optional[Callable], is_async: bool):
        """处理 context manager 函数失败的情况。
        
        Args:
            ori_func: 原始函数
            on_recover: 恢复回调函数
            is_async: 是否为异步函数
            
        Returns:
            Callable: 返回原始函数的包装器
        """
        logger.warning(f"Context manager hook failed, immediately reverting to original for {ori_func}")
        try:
            if callable(on_recover):
                on_recover()
        except Exception as re:
            logger.error(f"Failed to recover original function for {ori_func}: {re}")
        
        # 对于 context manager，直接返回原始函数的结果
        if is_async:
            async def _return_original_async(*args, **kwargs):
                return await ori_func(*args, **kwargs)
            return _return_original_async
        else:
            def _return_original_sync(*args, **kwargs):
                return ori_func(*args, **kwargs)
            return _return_original_sync

    def _handle_permanent_fallback(self, ori_func, on_recover: Optional[Callable], failures: int):
        """处理永久回退的情况。
        
        Args:
            ori_func: 原始函数
            on_recover: 恢复回调函数
            failures: 失败次数
        """
        if failures >= 5:
            try:
                if callable(on_recover):
                    on_recover()
                logger.warning(f"Hook permanently reverted to original for {ori_func} after 5 failures")
            except Exception as re:
                logger.error(f"Failed to recover original function for {ori_func}: {re}")

    def _create_async_wrapper(self, ori_func, profiler_func, pname: Optional[str], 
                             on_recover: Optional[Callable], is_context_manager_func: bool):
        """创建异步函数包装器。
        
        Args:
            ori_func: 原始函数
            profiler_func: profiler 函数
            pname: 调用者过滤名称
            on_recover: 恢复回调函数
            is_context_manager_func: 是否为 context manager 函数
            
        Returns:
            Callable: 异步包装器函数
        """
        failures = 0
        
        @functools.wraps(ori_func)
        async def async_wrapper(*args, **kwargs):
            # 检查调用者过滤
            if pname is not None and get_parents_name(ori_func) != pname:
                logger.debug(f"calling {ori_func}")
                return await ori_func(*args, **kwargs)

            nonlocal failures
            if failures >= 5:
                return await ori_func(*args, **kwargs)

            try:
                logger.debug(f"calling profiler_func={self.applied_hook_func_name} for {ori_func}")
                return await profiler_func(*args, **kwargs)
            except Exception as e:
                failures += 1
                self._log_hook_exception(ori_func, e, failures)
                
                # 对于 context manager 函数，第一次失败就立即回退
                if is_context_manager_func:
                    return await self._handle_context_manager_failure(ori_func, on_recover, True)(*args, **kwargs)
                
                # 异步函数，调用原始函数获取结果
                try:
                    result = await ori_func(*args, **kwargs)
                except Exception as orig_e:
                    logger.error(f"Original function also failed for {ori_func}: {orig_e}")
                    raise orig_e
                
                # 检查是否需要永久回退
                self._handle_permanent_fallback(ori_func, on_recover, failures)
                return result
        
        return async_wrapper

    def _create_sync_wrapper(self, ori_func, profiler_func, pname: Optional[str], 
                            on_recover: Optional[Callable], is_context_manager_func: bool):
        """创建同步函数包装器。
        
        Args:
            ori_func: 原始函数
            profiler_func: profiler 函数
            pname: 调用者过滤名称
            on_recover: 恢复回调函数
            is_context_manager_func: 是否为 context manager 函数
            
        Returns:
            Callable: 同步包装器函数
        """
        failures = 0
        
        @functools.wraps(ori_func)
        def wrapper(*args, **kwargs):
            # 检查调用者过滤
            if pname is not None and get_parents_name(ori_func) != pname:
                logger.debug(f"calling {ori_func}")
                return ori_func(*args, **kwargs)

            nonlocal failures
            if failures >= 5:
                return ori_func(*args, **kwargs)

            try:
                logger.debug(f"calling profiler_func={self.applied_hook_func_name} for {ori_func}")
                return profiler_func(*args, **kwargs)
            except Exception as e:
                failures += 1
                self._log_hook_exception(ori_func, e, failures)
                
                # 对于 context manager 函数，第一次失败就立即回退
                if is_context_manager_func:
                    return self._handle_context_manager_failure(ori_func, on_recover, False)(*args, **kwargs)
                
                # 普通函数，调用原始函数获取结果
                try:
                    result = ori_func(*args, **kwargs)
                except Exception as orig_e:
                    logger.error(f"Original function also failed for {ori_func}: {orig_e}")
                    raise orig_e
                
                # 检查是否需要永久回退
                self._handle_permanent_fallback(ori_func, on_recover, failures)
                return result
        
        return wrapper


def vllm_hook(
    hook_points: Union[Tuple[str, str], List[Tuple[str, str]]],
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    caller_filter: Optional[str] = None
) -> Callable:
    """装饰器工厂函数，用于简化hooker创建。
    
    Args:
        hook_points: hook点列表，格式为(模块名, 属性路径)
        min_version: 支持的最小版本
        max_version: 支持的最大版本
        caller_filter: 调用者过滤条件
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(hook_func):
        logger.debug(f"Handling {hook_func}")
        
        class AutoHooker(VLLMHookerBase):
            vllm_version = (min_version, max_version)
            applied_hook_func_name = getattr(hook_func, "__name__", str(hook_func))

            def init(self):
                hook_list = [hook_points] if isinstance(hook_points, tuple) else hook_points
                points = [import_object_from_string(import_path, func_path) for import_path, func_path in hook_list]
                self.do_hook(
                    hook_points=points,
                    profiler_func_maker=lambda ori_func: lambda *args, **kwargs: hook_func(ori_func, *args, **kwargs),
                    pname=caller_filter,
                )
        hooker = AutoHooker()
        # 在注册阶段即缓存文本点位与处理函数，便于配置化复用
        try:
            hook_list_local = [hook_points] if isinstance(hook_points, tuple) else list(hook_points)
        except Exception:
            hook_list_local = []
        hooker.hook_list = hook_list_local
        hooker.caller_filter = caller_filter
        hooker.hook_func = hook_func
        hooker.register()
        return hook_func

    return decorator
