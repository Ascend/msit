# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch

from msmodelslim.model.factory import ModelFactory, model_map
from msmodelslim.utils.exception import ToDoError, UnsupportedError


class DummyInterface:
    """模拟接口类"""
    pass


class DummyModelA(DummyInterface):
    """模拟模型A"""
    pass


class DummyModelB:
    """模拟模型B，不实现DummyInterface"""
    pass


class TestModelFactory(unittest.TestCase):

    def setUp(self):
        """每个测试前清空model_map"""
        model_map.clear()

    def tearDown(self):
        """每个测试后清空model_map"""
        model_map.clear()

    def test_register_new_model(self):
        """测试注册新模型"""
        @ModelFactory.register("TestModel")
        class TestModel:
            pass
        
        # 验证模型被注册到model_map
        self.assertIn("TestModel", model_map)
        self.assertEqual(model_map["TestModel"], TestModel)

    def test_register_duplicate_model(self):
        """测试注册重复模型抛出异常"""
        @ModelFactory.register("TestModel")
        class TestModel1:
            pass
        
        # 尝试注册同名模型应该抛出ToDoError
        with self.assertRaises(ToDoError) as context:
            @ModelFactory.register("TestModel")
            class TestModel2:
                pass
        
        self.assertIn("already registered", str(context.exception))

    def test_register_non_class_object(self):
        """测试注册非类对象抛出异常"""
        with self.assertRaises(ToDoError) as context:
            @ModelFactory.register("NotAClass")
            def some_function():
                pass
        
        self.assertIn("not a class", str(context.exception))

    def test_create_registered_model(self):
        """测试创建已注册的模型"""
        @ModelFactory.register("RegisteredModel")
        class RegisteredModel:
            pass
        
        result = ModelFactory.create("RegisteredModel")
        
        # 验证返回正确的类
        self.assertEqual(result, RegisteredModel)

    def test_create_model_with_interface_check_success(self):
        """测试创建模型时接口检查成功"""
        @ModelFactory.register("ModelWithInterface")
        class ModelWithInterface(DummyInterface):
            pass
        
        result = ModelFactory.create("ModelWithInterface", interface=DummyInterface)
        
        # 验证返回正确的类
        self.assertEqual(result, ModelWithInterface)

    def test_create_model_with_interface_check_failure(self):
        """测试创建模型时接口检查失败"""
        @ModelFactory.register("ModelWithoutInterface")
        class ModelWithoutInterface:
            pass
        
        # 尝试要求DummyInterface但模型未实现，应该抛出UnsupportedError
        with self.assertRaises(UnsupportedError) as context:
            ModelFactory.create("ModelWithoutInterface", interface=DummyInterface)
        
        self.assertIn("not implements", str(context.exception))

    def test_create_unregistered_model_with_default(self):
        """测试创建未注册的模型时使用default"""
        @ModelFactory.register("default")
        class DefaultModel:
            pass
        
        with patch('msmodelslim.model.factory.get_logger') as mock_logger:
            result = ModelFactory.create("UnregisteredModel")
            
            # 验证返回default模型
            self.assertEqual(result, DefaultModel)
            
            # 验证警告被记录
            mock_logger().warning.assert_called_once()
            warning_msg = mock_logger().warning.call_args[0][0]
            self.assertIn("UnregisteredModel", warning_msg)
            self.assertIn("default", warning_msg)

    def test_create_unregistered_model_without_default(self):
        """测试创建未注册的模型且没有default时抛出异常"""
        # 不注册任何模型，包括default
        
        with self.assertRaises(UnsupportedError) as context:
            ModelFactory.create("NonExistentModel")
        
        self.assertIn("not found", str(context.exception))

    def test_register_multiple_models(self):
        """测试注册多个模型"""
        @ModelFactory.register("Model1")
        class Model1:
            pass
        
        @ModelFactory.register("Model2")
        class Model2:
            pass
        
        @ModelFactory.register("Model3")
        class Model3:
            pass
        
        # 验证所有模型都被注册
        self.assertEqual(len(model_map), 3)
        self.assertIn("Model1", model_map)
        self.assertIn("Model2", model_map)
        self.assertIn("Model3", model_map)

    def test_register_returns_class(self):
        """测试register装饰器返回原始类"""
        original_class = type('OriginalClass', (), {})
        
        decorated_class = ModelFactory.register("TestModel")(original_class)
        
        # 验证装饰器返回原始类
        self.assertIs(decorated_class, original_class)

    def test_create_with_none_interface(self):
        """测试create方法interface为None时不进行接口检查"""
        @ModelFactory.register("AnyModel")
        class AnyModel:
            pass
        
        # interface为None时，不应该进行接口检查
        result = ModelFactory.create("AnyModel", interface=None)
        
        # 验证返回正确的类
        self.assertEqual(result, AnyModel)
