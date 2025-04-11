import sys
import unittest
from unittest.mock import ANY, MagicMock, Mock, call, patch

from msit.utils.exceptions import MsitException
from msit.utils.hijack import (
    POST_HOOK,
    PRE_HOOK,
    REPLACE,
    HiJackerManager,
    HiJackerPathFinder,
    HijackerUnit,
    HiJackerWrapperFunction,
    HiJackerWrapperModule,
    HiJackerWrapperObj,
    HijackHandler,
    hijacker,
    release,
)


class TestHijackerUnit(unittest.TestCase):
    def test_valid_parameters(self):
        stub = MagicMock()
        unit = HijackerUnit(stub, "module", "cls", "func", REPLACE, 100)
        self.assertEqual(unit.stub, stub)
        self.assertEqual(unit.module, "module")
        self.assertEqual(unit.cls, "cls")
        self.assertEqual(unit.function, "func")
        self.assertEqual(unit.action, REPLACE)
        self.assertEqual(unit.priority, 100)

    def test_invalid_stub(self):
        with self.assertRaises(MsitException):
            HijackerUnit("not_callable", "module", "", "", REPLACE, 100)

    def test_missing_module(self):
        with self.assertRaises(MsitException):
            HijackerUnit(MagicMock(), "", "", "", REPLACE, 100)

    def test_invalid_action(self):
        with self.assertRaises(MsitException):
            HijackerUnit(MagicMock(), "module", "", "", 999, 100)

    def test_replace_module_error(self):
        with self.assertRaises(MsitException):
            HijackerUnit(MagicMock(), "module", "", "", REPLACE, 100)


class TestHijackerUnit(unittest.TestCase):

    def test_valid_parameters(self):
        mock_stub = MagicMock()
        unit = HijackerUnit(mock_stub, "module_name", "ClassName", "function_name", REPLACE, 1)
        self.assertEqual(unit.module, "module_name")

    def test_invalid_stub(self):
        with self.assertRaises(MsitException) as context:
            HijackerUnit("not_callable", "module_name", "ClassName", "function_name", REPLACE, 1)
        self.assertIn('"stub" should be callable.', str(context.exception))

    def test_missing_module(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, None, "ClassName", "function_name", REPLACE, 1)
        self.assertIn('"module" is required.', str(context.exception))

    def test_invalid_module_type(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, 123, "ClassName", "function_name", REPLACE, 1)
        self.assertIn('"module" should be a str.', str(context.exception))

    def test_invalid_cls_type(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", 123, "function_name", REPLACE, 1)
        self.assertIn('"cls" should be a str.', str(context.exception))

    def test_missing_function_when_cls_present(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", "ClassName", None, REPLACE, 1)
        self.assertIn('"function" should be used when "cls" used.', str(context.exception))

    def test_invalid_function_type(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", "ClassName", 123, REPLACE, 1)
        self.assertIn('"function" should be a str.', str(context.exception))

    def test_invalid_action(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", "ClassName", "function_name", "INVALID_ACTION", 1)
        self.assertIn('"action" should be REPLACE, PRE_HOOK, or POST_HOOK.', str(context.exception))

    def test_module_replacement_not_supported(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", None, None, REPLACE, 1)
        self.assertIn("replacement of a module is not supported", str(context.exception))

    def test_invalid_priority_type(self):
        mock_stub = MagicMock()
        with self.assertRaises(MsitException) as context:
            HijackerUnit(mock_stub, "module_name", "ClassName", "function_name", REPLACE, "high")
        self.assertIn('"priority" should be an int.', str(context.exception))


class TestRelease(unittest.TestCase):
    @patch("msit.utils.hijack.HiJackerManager")
    def test_release_valid_handler(self, mock_manager):
        handler = MagicMock(spec=HijackHandler)
        handler.released = False
        handler.unit = "test_unit"
        release(handler)
        self.assertTrue(handler.released)
        mock_manager.remove_unit.assert_called_once_with("test_unit")

    def test_release_with_invalid_handler_type(self):
        invalid_handler = "not_a_handler"
        with self.assertRaises(MsitException) as context:
            release(invalid_handler)
        self.assertIn("Handler must be an instance of HijackHandler.", str(context.exception))


class TestHijackerManager(unittest.TestCase):
    def setUp(self):
        HiJackerManager._initialized = False
        HiJackerManager._hijacker_units = {}
        HiJackerManager._hijacker_wrappers = {}

    @patch("sys.meta_path", [])
    def test_initialize(self):
        HiJackerManager.initialize()
        self.assertTrue(HiJackerManager._initialized)
        self.assertIsInstance(sys.meta_path[0], HiJackerPathFinder)

    def test_add_and_remove_unit(self):
        stub = MagicMock()
        unit = HijackerUnit(stub, "test_module", "", "test_func", REPLACE, 100)
        handler = HiJackerManager.add_unit(unit)
        self.assertIn(handler, HiJackerManager._hijacker_units)
        wrapper = HiJackerManager._hijacker_wrappers.get("test_module--test_func")
        self.assertIsInstance(wrapper, HiJackerWrapperFunction)
        self.assertEqual(len(wrapper.replacement), 1)
        HiJackerManager.remove_unit(handler)
        self.assertNotIn(handler, HiJackerManager._hijacker_units)
        self.assertNotIn("test_module--test_func", HiJackerManager._hijacker_wrappers)


class ConcreteHiJackerWrapper(HiJackerWrapperObj):
    def activate(self):
        pass

    def deactivate(self):
        pass


class TestRemoveUnit(unittest.TestCase):
    def setUp(self):
        self.hijacker = ConcreteHiJackerWrapper("mod-class-func")
        self.unit_replace = Mock(action=REPLACE, priority=1)
        self.unit_pre = Mock(action=PRE_HOOK, priority=2)
        self.unit_post = Mock(action=POST_HOOK, priority=3)

    def test_remove_replace_unit(self):
        self.hijacker.replacement.append(self.unit_replace)
        self.hijacker.remove_unit(self.unit_replace)
        self.assertNotIn(self.unit_replace, self.hijacker.replacement)
        self.assertEqual(len(self.hijacker.replacement), 0)

    def test_remove_pre_hook_unit(self):
        self.hijacker.pre_hooks.append(self.unit_pre)
        self.hijacker.remove_unit(self.unit_pre)
        self.assertNotIn(self.unit_pre, self.hijacker.pre_hooks)
        self.assertEqual(len(self.hijacker.pre_hooks), 0)

    def test_remove_post_hook_unit(self):
        self.hijacker.post_hooks.append(self.unit_post)
        self.hijacker.remove_unit(self.unit_post)
        self.assertNotIn(self.unit_post, self.hijacker.post_hooks)
        self.assertEqual(len(self.hijacker.post_hooks), 0)

    def test_remove_non_existent_unit_raises_error(self):
        with self.assertRaises(ValueError):
            self.hijacker.remove_unit(self.unit_replace)
        self.hijacker.pre_hooks.append(Mock(action=PRE_HOOK))
        with self.assertRaises(ValueError):
            self.hijacker.remove_unit(self.unit_pre)
        self.hijacker.post_hooks.append(Mock(action=POST_HOOK))
        with self.assertRaises(ValueError):
            self.hijacker.remove_unit(self.unit_post)

    def test_remove_from_multiple_units(self):
        unit1 = Mock(action=REPLACE, priority=1)
        unit2 = Mock(action=REPLACE, priority=2)
        self.hijacker.replacement = [unit1, unit2]
        self.hijacker.remove_unit(unit1)
        self.assertEqual(self.hijacker.replacement, [unit2])

    def test_remove_does_not_affect_other_lists(self):
        self.hijacker.replacement.append(self.unit_replace)
        self.hijacker.pre_hooks.append(self.unit_pre)
        self.hijacker.post_hooks.append(self.unit_post)

        self.hijacker.remove_unit(self.unit_replace)
        self.assertIn(self.unit_pre, self.hijacker.pre_hooks)
        self.assertIn(self.unit_post, self.hijacker.post_hooks)


class TestHijackerWrapperModule(unittest.TestCase):
    def setUp(self):
        self.wrapper = HiJackerWrapperModule("test_module--")

    @patch("msit.utils.hijack.HiJackerPathFinder.add_mod")
    def test_activate(self, mock_add_mod):
        self.wrapper.activate()
        mock_add_mod.assert_called_once_with("test_module")

    def test_exec_pre_post_hooks(self):
        pre_unit = MagicMock(action=PRE_HOOK, stub=MagicMock())
        post_unit = MagicMock(action=POST_HOOK, stub=MagicMock())
        self.wrapper.add_unit(pre_unit)
        self.wrapper.add_unit(post_unit)
        mock_module = MagicMock()
        self.wrapper.exec_pre_hook()
        pre_unit.stub.assert_called_once()
        self.wrapper.exec_post_hook(mock_module)
        post_unit.stub.assert_called_once_with(mock_module)


class TestHiJackerWrapperFunction(unittest.TestCase):
    def setUp(self):
        self.target_name = "test_mod-TestClass-test_method"
        self.wrapper = HiJackerWrapperFunction(self.target_name)

        self.mock_module = MagicMock()
        self.mock_class = MagicMock()
        self.original_method = MagicMock()
        self.mock_module.TestClass = self.mock_class
        self.mock_class.test_method = self.original_method

    def test_initialization(self):
        self.assertEqual(self.wrapper.mod_name, "test_mod")
        self.assertEqual(self.wrapper.class_name, "TestClass")
        self.assertEqual(self.wrapper.func_name, "test_method")

    @patch("msit.utils.hijack.hijacker")
    @patch.dict("msit.utils.hijack.sys.modules", {"test_mod": None})
    def test_activate_module_not_loaded(self, mock_hijacker):
        self.wrapper.activate()
        mock_hijacker.assert_called_once_with(stub=ANY, module="test_mod", action=POST_HOOK, priority=0)

    def test_wrapper_execution_flow(self):
        pre_hook = MagicMock()
        pre_hook.stub = MagicMock(return_value=(("modified_args",), {"new_kw": 1}))
        replacement = MagicMock()
        replacement.stub = MagicMock(return_value="replaced_result")
        post_hook = MagicMock()
        post_hook.stub = MagicMock(return_value="final_result")

        self.wrapper.pre_hooks = [pre_hook]
        self.wrapper.replacement = [replacement]
        self.wrapper.post_hooks = [post_hook]
        self.wrapper.ori_obj = MagicMock()

        result = self.wrapper._get_wrapper()("arg1", kw1=2)

        pre_hook.stub.assert_called_once_with("arg1", kw1=2)
        replacement.stub.assert_called_once_with("modified_args", new_kw=1)
        post_hook.stub.assert_called_once_with("replaced_result", "modified_args", new_kw=1)
        self.assertEqual(result, "final_result")

    def test_pre_hook_type_check(self):
        invalid_hook = MagicMock()
        invalid_hook.stub = MagicMock(return_value="invalid_type")
        self.wrapper.pre_hooks = [invalid_hook]
        self.wrapper.ori_obj = MagicMock()

        with self.assertRaises(MsitException) as cm:
            self.wrapper._get_wrapper()("arg1")
        self.assertIn("Pre-hook must return a tuple", str(cm.exception))

    @patch("msit.utils.hijack.release")
    @patch.dict("msit.utils.hijack.sys.modules", {"test_mod": sys})
    def test_deactivate_with_missing_class(self, mock_release):
        self.wrapper.class_name = "NonExistentClass"
        self.wrapper.ori_obj = self.original_method
        self.wrapper.mod_hijacker = MagicMock()
        self.wrapper.deactivate()
        self.assertIsNone(self.wrapper.ori_obj)
        mock_release.assert_called_once()
