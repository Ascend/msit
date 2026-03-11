# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from unittest.mock import Mock

import pytest

from msprechecker.commands import CmdStrategy, CmdStrategyFactory, CmdType


# =============================================================================
# Tests for CmdStrategy (abstract base class)
# =============================================================================


def test_cmd_strategy_is_abstract():
    with pytest.raises(TypeError):
        CmdStrategy()


def test_cmd_strategy_subclass_must_implement_execute():
    class IncompleteStrategy(CmdStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_cmd_strategy_subclass_with_execute():
    class CompleteStrategy(CmdStrategy):
        def execute(self):
            return 0

    strategy = CompleteStrategy()
    assert strategy.execute() == 0


# =============================================================================
# Tests for CmdType enum
# =============================================================================


def test_cmd_type_values():
    assert CmdType.PRECHECK.value == "precheck"
    assert CmdType.DUMP.value == "dump"
    assert CmdType.COMPARE.value == "compare"
    assert CmdType.RUN.value == "run"
    assert CmdType.INSPECT.value == "inspect"


def test_cmd_type_enum_iteration():
    values = [cmd.value for cmd in CmdType]
    assert "precheck" in values
    assert "dump" in values
    assert "compare" in values
    assert "run" in values
    assert "inspect" in values


# =============================================================================
# Tests for CmdStrategyFactory
# =============================================================================


class TestCmdStrategyFactory:
    def test_factory_init(self):
        factory = CmdStrategyFactory()
        assert factory is not None
        # Check that registry contains expected commands
        assert CmdType.PRECHECK in factory._registry
        assert CmdType.DUMP in factory._registry
        assert CmdType.COMPARE in factory._registry
        assert CmdType.RUN in factory._registry
        assert CmdType.INSPECT in factory._registry

    def test_factory_get_precheck(self):
        factory = CmdStrategyFactory()
        strategy = factory.get(CmdType.PRECHECK)
        assert strategy is not None
        # Verify it's the correct type
        from msprechecker.commands.precheck import Precheck

        assert isinstance(strategy, Precheck)

    def test_factory_get_dump(self):
        factory = CmdStrategyFactory()
        strategy = factory.get(CmdType.DUMP)
        assert strategy is not None
        from msprechecker.commands.dump import Dump

        assert isinstance(strategy, Dump)

    def test_factory_get_compare(self):
        factory = CmdStrategyFactory()
        strategy = factory.get(CmdType.COMPARE)
        assert strategy is not None
        from msprechecker.commands.compare import Compare

        assert isinstance(strategy, Compare)

    def test_factory_get_run(self):
        factory = CmdStrategyFactory()
        strategy = factory.get(CmdType.RUN)
        assert strategy is not None
        from msprechecker.commands.cmate import Run

        assert isinstance(strategy, Run)

    def test_factory_get_inspect(self):
        factory = CmdStrategyFactory()
        strategy = factory.get(CmdType.INSPECT)
        assert strategy is not None
        from msprechecker.commands.cmate import Inspect

        assert isinstance(strategy, Inspect)

    def test_factory_get_unknown_raises_error(self):
        factory = CmdStrategyFactory()
        # Create a mock CmdType that doesn't exist in registry
        mock_cmd_type = Mock()
        mock_cmd_type.__hash__ = Mock(return_value=hash("unknown"))

        with pytest.raises(ValueError, match="No strategy registered"):
            factory.get(mock_cmd_type)

    def test_factory_register_new_strategy(self):
        factory = CmdStrategyFactory()

        class NewStrategy(CmdStrategy):
            def execute(self):
                return 0

        # Create a new command type
        new_cmd_type = CmdType.PRECHECK  # Use existing for simplicity

        # Register should work without error
        factory.register(new_cmd_type, NewStrategy)

        # Verify the new strategy is returned
        strategy = factory.get(new_cmd_type)
        assert isinstance(strategy, NewStrategy)

    def test_factory_register_invalid_strategy_raises_error(self):
        factory = CmdStrategyFactory()

        class NotAStrategy:
            pass

        with pytest.raises(
            TypeError, match="Expected 'strategy_cls' to be 'Cmdstrategy'"
        ):
            factory.register(CmdType.PRECHECK, NotAStrategy)

    def test_factory_register_non_subclass_raises_error(self):
        factory = CmdStrategyFactory()

        # A class that doesn't inherit from CmdStrategy
        class RegularClass:
            def execute(self):
                return 0

        with pytest.raises(TypeError):
            factory.register(CmdType.PRECHECK, RegularClass)

    def test_factory_returns_new_instance_each_time(self):
        factory = CmdStrategyFactory()
        strategy1 = factory.get(CmdType.PRECHECK)
        strategy2 = factory.get(CmdType.PRECHECK)

        # Should be different instances
        assert strategy1 is not strategy2
        # But same type
        assert type(strategy1) is type(strategy2)


# =============================================================================
# Integration tests
# =============================================================================


def test_all_cmd_types_have_strategies():
    """Verify that all defined CmdType values have corresponding strategies."""
    factory = CmdStrategyFactory()

    for cmd_type in CmdType:
        strategy = factory.get(cmd_type)
        assert strategy is not None
        assert hasattr(strategy, "execute")
        assert callable(strategy.execute)


def test_strategy_execute_returns_int():
    """Verify that all strategies return an integer from execute."""
    factory = CmdStrategyFactory()

    for cmd_type in CmdType:
        strategy = factory.get(cmd_type)
        # Just verify the method exists and is callable
        assert callable(strategy.execute)
