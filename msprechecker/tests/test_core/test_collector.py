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

from msprechecker.core.collector import Collector, CollectStrategy


@pytest.fixture
def collector():
    return Collector()


@pytest.mark.parametrize(
    "invalid_strategies",
    [
        pytest.param(2, id="integer"),
        pytest.param(2.5, id="float"),
        pytest.param(True, id="boolean_true"),
        pytest.param(False, id="boolean_false"),
        pytest.param(object(), id="object_instance"),
    ],
)
def test_collector_given_invalid_type_not_iterable_when_initializing_then_raises_type_error(
    invalid_strategies, caplog
):
    with pytest.raises(TypeError, match="is not iterable"):
        Collector(collect_strategies=invalid_strategies)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        assert (
            caplog.records[0].message
            == f"collect_strategies must be an iterable. Got {invalid_strategies} instead"
        )


@pytest.mark.parametrize(
    "invalid_strategies",
    [
        pytest.param("str", id="string"),
        pytest.param([1], id="list_integers"),
        pytest.param((1.5,), id="tuple_floats"),
        pytest.param({True, False}, id="set_booleans"),
    ],
)
def test_collector_given_invalid_iterable_of_non_strategy_when_initializing_then_raises_type_error(
    invalid_strategies,
):
    with pytest.raises(TypeError, match="must be instances of CollectStrategy"):
        Collector(collect_strategies=invalid_strategies)


@pytest.mark.parametrize(
    "invalid_strategies",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="emtpy_list"),
        pytest.param((), id="emtpy_tuple"),
        pytest.param({}, id="emtpy_set"),
    ],
)
def test_collector_given_none_or_empty_iterable_when_initializing_then_succeeds(
    invalid_strategies,
):
    collector = Collector(collect_strategies=invalid_strategies)
    assert isinstance(collector, Collector)


@pytest.mark.parametrize(
    "invalid_strategy",
    [
        pytest.param(2, id="integer"),
        pytest.param(2.5, id="float"),
        pytest.param(True, id="boolean_true"),
        pytest.param(False, id="boolean_false"),
        pytest.param(object(), id="object_instance"),
        pytest.param("str", id="string"),
        pytest.param([1], id="list_integers"),
        pytest.param((1.5,), id="tuple_floats"),
        pytest.param({True, False}, id="set_booleans"),
        pytest.param(None, id="none"),
        pytest.param([], id="emtpy_list"),
        pytest.param((), id="emtpy_tuple"),
        pytest.param({}, id="emtpy_set"),
    ],
)
def test_add_strategy_given_non_strategy_instance_when_calling_then_raises_type_error(
    collector, invalid_strategy
):
    with pytest.raises(TypeError, match="must be an instance of CollectStrategy"):
        collector.add_strategy(invalid_strategy)


def test_init_given_valid_strategy_list_when_initializing_then_strategies_set_correctly(
    collector,
):
    strategy1 = Mock(spec=CollectStrategy)
    strategy2 = Mock(spec=CollectStrategy)

    collector = Collector(collect_strategies=[strategy1, strategy2])
    assert collector._collect_strategies == [strategy1, strategy2]


def test_collect_given_no_strategies_when_called_then_returns_empty_dict(collector):
    assert collector.collect() == {}


def test_collect_given_single_strategy_when_called_then_returns_dict_with_strategy_name_as_key(
    collector,
):
    strategy = Mock(spec=CollectStrategy)
    strategy.name = "test_name"
    strategy.execute.return_value = {"a": "b"}
    collector.add_strategy(strategy)

    assert collector.collect() == {strategy.name: strategy.execute.return_value}


def test_collect_with_duplicate_strategy_names_then_overwrites_with_last_result(
    collector,
):
    strategy1 = Mock(spec=CollectStrategy)
    strategy1.name = "same_name"
    strategy1.execute.return_value = {"a": "b"}
    strategy2 = Mock(spec=CollectStrategy)
    strategy2.name = "same_name"
    strategy2.execute.return_value = {"c": "d"}

    collector.add_strategy(strategy1)
    collector.add_strategy(strategy2)

    assert collector.collect() == {strategy2.name: strategy2.execute.return_value}
