#  -*- coding: utf-8 -*-
#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Optional, Type, Any, TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    """
    处理器注册表基类，用于管理和注册处理器。
    
    该类提供了一个中央注册表，用于存储和管理不同类型的处理器。
    支持通过装饰器方式注册处理器，并提供查询和获取处理器的功能。
    处理器以配置类型字符串为键进行注册，不同的处理器用于处理不同的配置类型。
    匹配逻辑基于配置类型的字符串名称。
    """

    def __init__(self) -> None:
        # 存储所有注册的处理器
        self._types_by_name: Dict[str, Type[T]] = {}
        self._types_by_type: Dict[Type[Any], Type[T]] = {}

    def register(self, name: str = None):
        """
        处理器注册装饰器。
        
        该装饰器用于注册处理器类，可以指定处理器的名称和适用的配置类型。
        
        参数:
            name: 处理器的名称，如果不指定，则使用处理器类的名称
        返回:
            装饰器函数，用于注册处理器类
        """

        def decorator(handler_cls: Type[T]) -> Type[T]:
            self._types_by_name[name] = handler_cls
            return handler_cls

        return decorator

    def register_by_name(self, name: str):
        """
        根据名称注册处理器类。
        
        参数:
            name: 处理器的名称
        返回:
            装饰器函数，用于注册处理器类
        """

        def decorator(handler_cls: Type[T]) -> Type[T]:
            self._types_by_name[name] = handler_cls
            return handler_cls

        return decorator

    def register_by_type(self, type_: Type[Any]):
        """
        根据类型注册处理器类。
        
        参数:
            type: 处理器的类型
        """

        def decorator(handler_cls: Type[T]) -> Type[T]:
            self._types_by_type[type_] = handler_cls
            return handler_cls

        return decorator

    def get_by_name(self, name: str) -> Optional[Type[T]]:
        """
        根据名称获取处理器类。
        
        参数:
            name: 处理器的名称
            
        返回:
            Type[BaseHandler]: 处理器类，如果不存在则返回None
        """
        return self._types_by_name.get(name, None)

    def get_by_type(self, type_: Type[Any]) -> Optional[Type[T]]:
        """
        根据类型获取处理器类。
        
        参数:
            type: 处理器的类型
        """
        return self._types_by_type.get(type_, None)

    def get_all(self) -> Dict[str, Type[T]]:
        """
        获取所有注册的处理器。
        
        返回:
            Dict[str, Type[BaseHandler]]: 处理器名称到处理器类的映射
        """
        return self._types_by_name.copy()

    def clear(self):
        """
        清空注册表。
        
        该方法会清空所有注册的处理器和配置类型映射。
        """
        self._types_by_name.clear()
