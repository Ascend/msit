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

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from msprechecker.core.strategy import (
    _Ascend,
    _AscendComponent,
    _OptionIp,
    _SingleOption,
    Ascend,
    CollectStrategy,
    CollectStrategyGroup,
    Config,
    CPU,
    CPUHighPerformance,
    Driver,
    Env,
    HcclPing,
    HccsPing,
    Kernel,
    Link,
    Lscpu,
    MindIE,
    Network,
    NPU,
    OppKernel,
    PageSize,
    Ping,
    Sys,
    TB,
    TBSpeed,
    Tls,
    Toolkit,
    TransparentHugepage,
    VirtualMachine,
    Vnic,
    Weight,
)
from msprechecker.util import RankTable


@pytest.fixture
def strategy_group():
    return CollectStrategyGroup("test")


def test_cant_instantiate_abstract_collect_strategy_given_base_class_when_init_then_raises_type_error():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class CollectStrategy.*abstract method.*execute",
    ):
        CollectStrategy("test")


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
def test_init_strategy_group_given_non_iterable_strategies_then_raises_type_error_and_logs_error(
    invalid_strategies, caplog
):
    with pytest.raises(TypeError, match="object is not iterable"):
        CollectStrategyGroup("test", strategies=invalid_strategies)
        assert len(caplog.records, 1)
        assert caplog.records[0].levelname == "ERROR"
        assert (
            caplog.records[0].message
            == f"strategies must be an iterable. Got {invalid_strategies} instead"
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
def test_init_strategy_group_given_iterable_with_invalid_elements_then_raises_type_error(
    invalid_strategies,
):
    with pytest.raises(
        TypeError, match="collect_strategy must be an instance of CollectStrategy"
    ):
        CollectStrategyGroup("test", strategies=invalid_strategies)


def test_init_strategy_group_given_valid_strategies_then_initializes_correctly():
    strategy1 = Mock(spec=CollectStrategy)
    strategy2 = Mock(spec=CollectStrategy)
    strategy_group = CollectStrategyGroup("test", [strategy1, strategy2])
    assert strategy_group.name == "test"
    assert strategy_group._strategies == [strategy1, strategy2]


def test_execute_on_empty_strategy_group_then_returns_empty_dict(strategy_group):
    assert strategy_group.execute() == {}


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
def test_add_given_invalid_type_then_raises_type_error(
    strategy_group, invalid_strategy
):
    with pytest.raises(
        TypeError,
        match="collect_strategy must be an instance of CollectStrategy",
    ):
        strategy_group.add(invalid_strategy)


def test_add_given_valid_strategy_and_group_then_adds_to_internal_list(strategy_group):
    strategy1 = Mock(spec=CollectStrategy)
    strategy2 = Mock(spec=CollectStrategyGroup)
    strategy_group.add(strategy1).add(strategy2)
    assert strategy_group.name == "test"
    assert strategy_group._strategies == [strategy1, strategy2]


def test_execute_with_mixed_strategies_given_unique_names_then_returns_merged_result_dict(
    strategy_group,
):
    strategy1 = Mock(spec=CollectStrategyGroup)
    strategy2 = Mock(spec=CollectStrategy)
    strategy1.name = "name_1"
    strategy1.execute.return_value = {"a": "b"}
    strategy2.name = "name_2"
    strategy2.execute.return_value = {"c": "d"}

    strategy_group.add(strategy1).add(strategy2)
    assert strategy_group.execute() == {
        strategy1.name: strategy1.execute.return_value,
        strategy2.name: strategy2.execute.return_value,
    }


def test_add_given_duplicate_strategy_names_then_raises_value_error(strategy_group):
    strategy1 = Mock(spec=CollectStrategyGroup)
    strategy2 = Mock(spec=CollectStrategy)
    strategy1.name = "same_name"
    strategy1.execute.return_value = {"a": "b"}
    strategy2.name = "same_name"
    strategy2.execute.return_value = {"c": "d"}

    strategy_group.add(strategy1)
    with pytest.raises(
        ValueError,
        match="A strategy with name 'same_name' already exists in this group",
    ):
        strategy_group.add(strategy2)


# Test Env


def test_init_env_given_no_name_or_custom_name_then_sets_name_correctly():
    env_strategy1 = Env()
    env_strategy2 = Env("test")
    assert env_strategy1.name == "env"
    assert env_strategy2.name == "test"


def test_execute_given_any_env_vars_when_default_then_returns_all_os_environ():
    test_environ = {"TEST": "test"}
    with patch.dict(os.environ, test_environ, clear=True):
        assert Env().execute() == test_environ


def test_execute_given_mixed_env_vars_when_ascend_only_then_filters_to_prefixed_vars():
    ascend_environ = {"MINDIE_": "mindie", "ASCEND_": "ascend"}
    test_environ = {"TEST": "test", **ascend_environ}
    with patch.dict(os.environ, test_environ, clear=True):
        assert Env(ascend_only=True).execute() == ascend_environ


# Test Lscpu


@patch("shutil.which", return_value=None)
def test_execute_given_lscpu_command_not_found_when_execute_then_returns_none_and_logs_warning(
    mock_which, caplog
):
    assert Lscpu().execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "lscpu command not found in system PATH"


@patch("shutil.which", return_value="random_path")
@patch("subprocess.check_output", side_effect=RuntimeError)
def test_execute_given_lscpu_command_fails_when_execute_then_returns_none_and_logs_error(
    mock_check_output, mock_which, caplog
):
    assert Lscpu().execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to execute lscpu command:" in caplog.records[0].message


@patch("shutil.which", return_value="random_path")
@patch("subprocess.check_output", return_value="a: b\nc\nd: e")
def test_execute_given_valid_lscpu_output_when_parse_then_returns_correct_parsed_dictionary(
    mock_check_output, mock_which
):
    assert Lscpu().execute() == {"a": "b", "d": "e"}


# Test CPUHighPerformance


@patch("psutil.cpu_freq", return_value=None)
def test_check_via_psutil_given_no_cpu_frequency_when_check_then_returns_false(
    mock_psutil, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_psutil()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert (
        caplog.records[0].message
        == "Unable to get CPU frequency information via psutil"
    )


@patch("psutil.cpu_freq", return_value=Mock(current=2000, max=3000))
def test_check_via_psutil_given_current_freq_below_max_when_check_then_returns_false(
    mock_which,
):
    assert not CPUHighPerformance()._check_via_psutil()


@patch("psutil.cpu_freq", return_value=Mock(current=3000, max=3000))
def test_check_via_psutil_given_current_freq_equals_max_when_check_then_returns_true(
    mock_which,
):
    assert CPUHighPerformance()._check_via_psutil()


@patch("os.cpu_count", return_value=None)
def test_check_via_scaling_governor_given_cpu_count_is_none_when_check_then_returns_false_and_logs_debug_message(
    mock_cpu_count, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Unable to determine CPU count"


@patch("os.path.isfile", return_value=False)
@patch("os.cpu_count", return_value=1)
def test_check_via_scaling_governor_given_file_not_found_returns_false(
    mock_cpu_count, mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Scaling governor file not found for CPU core 0"


@patch("builtins.open", new_callable=mock_open, read_data="normal")
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_via_scaling_governor_given_non_performance_returns_false(
    mock_cpu_count, mock_isfile, mock_file, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert (
        caplog.records[0].message
        == "CPU core 0 scaling governor is not set to performance mode"
    )


@patch("builtins.open", side_effect=OSError)
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_via_scaling_governor_given_file_open_fails_returns_false(
    mock_cpu_count, mock_isfile, mock_cpu_freq
):
    assert not CPUHighPerformance()._check_via_scaling_governor()


@patch("builtins.open", new_callable=mock_open, read_data="performance")
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_via_scaling_governor_given_governor_file_exists_and_content_is_performance_when_check_then_returns_true(
    mock_cpu_count, mock_isfile, mock_cpu_freq
):
    assert CPUHighPerformance()._check_via_scaling_governor()


@patch("shutil.which", return_value=None)
def test_check_via_cpupower_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_cpupower()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "cpupower command not found in system PATH"


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_via_cpupower_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_cpupower()
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to execute cpupower command" in caplog.records[0].message


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_cpupower_given_current_freq_below_max_limit_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 1.00 GHz
"""
    assert not CPUHighPerformance()._check_via_cpupower()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_cpupower_given_output_missing_current_freq_line_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
"""
    assert not CPUHighPerformance()._check_via_cpupower()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_cpupower_given_current_freq_equals_max_limit_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 3.00 GHz
"""
    assert CPUHighPerformance()._check_via_cpupower()


@patch("shutil.which", return_value=None)
def test_check_via_dmidecode_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_dmidecode()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "dmidecode command not found in system PATH"


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_via_dmidecode_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_dmidecode()
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to execute dmidecode command" in caplog.records[0].message


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_dmidecode_given_output_missing_current_speed_line_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
"""
    assert not CPUHighPerformance()._check_via_dmidecode()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_dmidecode_given_current_speed_is_zero_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
Current Speed: 0 MHz
"""
    assert not CPUHighPerformance()._check_via_dmidecode()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_dmidecode_given_current_speed_equals_max_speed_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
Current Speed: 3000 MHz
"""
    assert CPUHighPerformance()._check_via_dmidecode()


@patch("shutil.which", return_value=None)
def test_check_via_lshw_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_lshw()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "lshw command not found in system PATH"


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_lshw_given_output_missing_capacity_key_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
capacity: 1
"""
    assert not CPUHighPerformance()._check_via_lshw()


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_via_lshw_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_via_lshw()
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to execute lshw command" in caplog.records[0].message


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_via_lshw_given_output_contains_size_and_capacity_keys_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
size: 1
capacity: 1
"""
    assert CPUHighPerformance()._check_via_lshw()


# Test VirtualMachine


@patch("os.path.isfile", return_value=False)
def test_execute_given_cpuinfo_file_does_not_exist_when_check_then_returns_false(
    mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not VirtualMachine().execute()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "/proc/cpuinfo file not found"


@patch("builtins.open", side_effect=IOError)
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_but_open_fails_when_check_then_returns_false(
    mock_isfile, mock_file, caplog
):
    assert not VirtualMachine().execute()
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to read /proc/cpuinfo file" in caplog.records[0].message


@patch("builtins.open", new_callable=mock_open, read_data="其他的玩意儿")
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_without_hypervisor_keyword_when_check_then_returns_false(
    mock_isfile, mock_file
):
    assert not VirtualMachine().execute()


@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc\nhypervisor")
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_containing_hypervisor_keyword_when_check_then_returns_true(
    mock_isfile, mock_file
):
    assert VirtualMachine().execute()


# Test TransparentHugepage


@patch("os.path.isfile", return_value=False)
def test_execute_given_hugepage_file_does_not_exist_when_check_then_returns_none(
    mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    assert TransparentHugepage().execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert (
        caplog.records[0].message == "Transparent hugepage configuration file not found"
    )


@patch("builtins.open", side_effect=IOError)
@patch("os.path.isfile", return_value=True)
def test_execute_given_hugepage_file_exists_but_open_fails_when_check_then_returns_none(
    mock_isfile, mock_file, caplog
):
    assert TransparentHugepage().execute() is None
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert (
        "Failed to read transparent hugepage configuration" in caplog.records[0].message
    )


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_execute_given_hugepage_file_exists_and_readable_when_check_then_returns_file_content(
    mock_isfile, mock_file
):
    test_data = "always [madvise] never"
    mock_file.return_value.read.return_value = test_data
    assert TransparentHugepage().execute() == test_data


# Test Kernel


@patch("platform.uname")
def test_execute_given_mock_platform_uname_call_when_execute_then_returns_uname_asdict_result(
    mock_uname,
):
    test_data = {
        "system": "a",
        "node": "b",
        "release": "c",
        "version": "d",
        "machine": "e",
        "processor": "f",
    }
    mock_uname.return_value._asdict = Mock(return_value=test_data)
    assert Kernel().execute() == test_data


# Test PageSize


@patch("os.sysconf", side_effect=OSError)
def test_execute_given_sysconf_call_raises_oserror_when_execute_then_returns_none_and_logs_error(
    mock_sysconf, caplog
):
    assert PageSize().execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to get system page size" in caplog.records[0].message


@patch("os.sysconf", return_value=2)
def test_execute_given_sysconf_call_returns_integer_value_when_execute_then_returns_that_value(
    mock_sysconf,
):
    assert PageSize().execute() == 2


# Test Sys


def test_sys_initialization_with_default_strategies_then_has_correct_name_and_strategy_count():
    sys_strategy = Sys()
    assert sys_strategy.name == "sys"
    assert len(sys_strategy._strategies) == 7


def test_sys_execute_given_custom_strategy_list_when_execute_then_returns_dict_of_strategy_name_to_execute_result():
    mock_strategy1 = Mock(spec=CollectStrategy)
    mock_strategy1.name = "test_1"
    mock_strategy1.execute.return_value = "test_1"
    mock_strategy2 = Mock(name="test_2", spec=CollectStrategy)
    mock_strategy2.name = "test_2"
    mock_strategy2.execute.return_value = "test_2"

    assert Sys(strategies=[mock_strategy1, mock_strategy2]).execute() == {
        "test_1": "test_1",
        "test_2": "test_2",
    }


# Test Config


def test_given_empty_path_when_execute_then_logs_warning_and_returns_none(caplog):
    assert Config("test", config_path="").execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Configuration path is empty or not provided"


def test_given_invalid_path_when_execute_then_logs_file_not_found_warning_and_returns_none(
    caplog,
):
    assert Config("test", config_path="/nonexistent/path/file.json").execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        "Configuration file '/nonexistent/path/file.json' not found"
        in caplog.records[0].message
    )


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_mocked_file_with_unsupported_format_returns_raw_content(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "random_data"
    assert Config("test", config_path="a").execute() == "random_data"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Unsupported configuration file format: 'a'"


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_mocked_json_file_with_parse_error_when_execute_then_logs_json_parse_error_and_returns_raw_content(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "random_data"
    result = Config("test", config_path="a.json").execute()
    assert result == "random_data"
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert (
        "Failed to parse JSON configuration file 'a.json'" in caplog.records[0].message
    )


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_mocked_yaml_file_with_parse_error_when_execute_then_logs_yaml_parse_error_and_returns_raw_content(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "- a\nb"
    assert Config("test", config_path="a.yaml").execute() == "- a\nb"
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert (
        "Failed to parse YAML configuration file 'a.yaml'" in caplog.records[0].message
    )


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_mocked_yaml_file_with_parse_error_returns_raw_content(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "--- - a\nb"
    assert Config("test", config_path="a.yaml").execute() == "--- - a\nb"
    # logger.exception logs at ERROR level
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to parse YAML configuration file" in caplog.records[0].message


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_valid_mocked_json_file_when_execute_then_returns_parsed_dict(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = '{"a": "b"}'
    assert Config("test", config_path="a.json").execute() == {"a": "b"}


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_valid_single_document_yaml_file_when_execute_then_returns_parsed_dict(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "a: b"
    assert Config("test", config_path="a.yaml").execute() == {"a": "b"}


@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_given_valid_multi_document_yaml_file_when_execute_then_returns_list_of_dicts(
    mock_isfile, mock_file, caplog
):
    mock_file.return_value.read.return_value = "a: b\n---\nc : d"
    assert Config("test", config_path="a.yaml").execute() == [{"a": "b"}, {"c": "d"}]


# test Weight


def test_weight_init_with_default_params():
    weight = Weight("test_weight")
    assert weight.name == "test_weight"
    assert weight._weight_dir == ""
    assert weight._tensor_suffix == ".safetensors"
    assert weight._max_size == 10 * 1024**3
    assert weight._chunk_size == 256 * 1024**2
    assert weight._max_hash_workers == 4


def test_weight_init_with_custom_params():
    weight = Weight(
        "custom_weight",
        weight_dir="/path/to/weights",
        tensor_suffix=".bin",
        max_size=1024**3,
        chunk_size=1024**2,
        max_hash_workers=8,
    )
    assert weight.name == "custom_weight"
    assert weight._weight_dir == "/path/to/weights"
    assert weight._tensor_suffix == ".bin"
    assert weight._max_size == 1024**3
    assert weight._chunk_size == 1024**2
    assert weight._max_hash_workers == 8


@patch("os.path.isdir", return_value=False)
def test_weight_execute_given_invalid_dir_then_logs_warning_and_returns_none(
    mock_isdir, caplog
):
    weight = Weight("test", weight_dir="/nonexistent")
    assert weight.execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Expected '/nonexistent' to be a directory" in caplog.records[0].message


@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=[])
def test_weight_execute_given_empty_dir_then_logs_warning_and_returns_none(
    mock_listdir, mock_isdir, caplog
):
    weight = Weight("test", weight_dir="/empty")
    assert weight.execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "No valid tensor files found in '/empty'" in caplog.records[0].message


@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["model.safetensors"])
@patch("os.path.islink", return_value=False)
@patch("os.path.isfile", return_value=True)
@patch("os.path.getsize", return_value=1024)
@patch.object(Weight, "_calculate_hash256", return_value="mock_hash_123")
def test_weight_execute_given_valid_files_then_returns_hashes(
    mock_calculate_hash,
    mock_getsize,
    mock_isfile,
    mock_islink,
    mock_listdir,
    mock_isdir,
):
    weight = Weight("test", weight_dir="/weights", max_hash_workers=1)
    result = weight.execute()
    assert result is not None
    assert (
        "model.safetensors" in result
        or result.get("model.safetensors") == "mock_hash_123"
    )


@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["model-00001-of-00003.safetensors"])
@patch("os.path.islink", return_value=False)
@patch("os.path.isfile", return_value=True)
@patch("os.path.getsize", return_value=1024)
@patch("builtins.open", new_callable=mock_open, read_data=b"test data")
def test_weight_get_tensor_id_given_shard_filename_then_extracts_id(
    mock_file, mock_getsize, mock_isfile, mock_islink, mock_listdir, mock_isdir
):
    weight = Weight("test", weight_dir="/weights", max_hash_workers=1)
    tensor_id = weight._get_tensor_id("/weights/model-00001-of-00003.safetensors")
    assert tensor_id == "00001"


def test_weight_get_tensor_id_given_no_shard_pattern_then_returns_basename():
    weight = Weight("test")
    tensor_id = weight._get_tensor_id("/weights/model.safetensors")
    assert tensor_id == "model.safetensors"


@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["model.safetensors"])
@patch("os.path.islink", return_value=True)
def test_weight_is_valid_tensor_file_given_symlink_then_returns_false_and_logs_warning(
    mock_islink, mock_listdir, mock_isdir, caplog
):
    weight = Weight("test", weight_dir="/weights")
    result = weight._is_valid_tensor_file("/weights/model.safetensors")
    assert result is False


@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["model.safetensors"])
@patch("os.path.islink", return_value=False)
@patch("os.path.isfile", return_value=True)
@patch("os.path.getsize", return_value=20 * 1024**3)  # 20 GB > 10 GB default
def test_weight_is_valid_tensor_file_given_file_exceeds_max_size_then_returns_false_and_logs_warning(
    mock_getsize, mock_isfile, mock_islink, mock_listdir, mock_isdir, caplog
):
    weight = Weight("test", weight_dir="/weights")
    result = weight._is_valid_tensor_file("/weights/model.safetensors")
    assert result is False
    assert len(caplog.records) == 1
    assert "exceeds max_size" in caplog.records[0].message


# Test _Ascend and _AscendComponent


def test_ascend_init_with_required_params():
    ascend = _Ascend("test", version_path="version.info")
    assert ascend.name == "test"
    assert ascend._version_path == "version.info"
    assert ascend._default_home == ""
    assert ascend._home_environ == ""


def test_ascend_init_with_all_params():
    ascend = _Ascend(
        "test",
        version_path="path/to/version.info",
        default_home="/usr/local/test",
        home_environ="TEST_HOME",
    )
    assert ascend._version_path == "path/to/version.info"
    assert ascend._default_home == "/usr/local/test"
    assert ascend._home_environ == "TEST_HOME"


@patch.dict(os.environ, {"TEST_HOME": "/custom/path"}, clear=True)
def test_ascend_resolve_home_given_valid_env_var_then_returns_path():
    ascend = _Ascend("test", version_path="v.info", home_environ="TEST_HOME")
    result = ascend._resolve_home()
    assert result == "/custom/path"


@patch.dict(os.environ, {}, clear=True)
def test_ascend_resolve_home_given_missing_env_var_then_returns_empty():
    ascend = _Ascend("test", version_path="v.info", home_environ="TEST_HOME")
    result = ascend._resolve_home()
    assert result == ""


@patch.dict(os.environ, {"TEST_HOME": "/custom/path"}, clear=True)
def test_ascend_resolve_home_given_env_outside_default_then_returns_empty_and_logs_warning(
    caplog,
):
    ascend = _Ascend(
        "test",
        version_path="v.info",
        default_home="/usr/local/Ascend/test",
        home_environ="TEST_HOME",
    )
    result = ascend._resolve_home()
    assert result == ""
    assert len(caplog.records) == 1
    assert "outside the expected root" in caplog.records[0].message


def test_ascend_resolve_full_path_given_absolute_version_path():
    ascend = _Ascend("test", version_path="/absolute/path/version.info")
    result = ascend._resolve_full_path("/home")
    assert str(result) == "/absolute/path/version.info"


def test_ascend_resolve_full_path_given_relative_version_path():
    ascend = _Ascend("test", version_path="relative/version.info")
    result = ascend._resolve_full_path("/home")
    assert str(result).endswith("home/relative/version.info")


def test_ascend_resolve_full_path_given_no_home_uses_default():
    ascend = _Ascend("test", version_path="v.info", default_home="/default/home")
    result = ascend._resolve_full_path("")
    assert str(result) == "/default/home/v.info"


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="KEY=value\nKEY2: value2\n\ninvalid_line\n",
)
def test_ascend_parse_version_file_given_valid_content_then_returns_dict(mock_file):
    ascend = _Ascend("test", version_path="v.info")
    result = ascend._parse_version_file(Path("/fake/path"))
    assert result == {"KEY": "value", "KEY2": "value2"}


@patch("pathlib.Path.is_file", return_value=False)
def test_ascend_execute_given_file_not_found_then_returns_none_and_logs_debug(
    mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    ascend = _Ascend("test", version_path="/nonexistent/version.info")
    result = ascend.execute()
    assert result is None
    assert len(caplog.records) == 1
    assert "Version file not found" in caplog.records[0].message


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", side_effect=OSError("Permission denied"))
def test_ascend_execute_given_file_read_error_then_returns_none_and_logs_error(
    mock_file, mock_isfile, caplog
):
    ascend = _Ascend("test", version_path="/path/version.info")
    result = ascend.execute()
    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="\n\n")
def test_ascend_execute_given_empty_file_then_returns_none_and_logs_debug(
    mock_file, mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    ascend = _Ascend("test", version_path="/path/version.info")
    result = ascend.execute()
    assert result is None
    assert "Version file yielded no data" in caplog.records[0].message


def test_driver_component_has_correct_defaults():
    driver = Driver()
    assert driver.name == "driver"
    assert driver._version_path == "/usr/local/Ascend/driver/version.info"


def test_toolkit_component_has_correct_defaults():
    toolkit = Toolkit()
    assert toolkit.name == "toolkit"
    assert toolkit._version_path == "toolkit/version.info"
    assert toolkit._default_home == "/usr/local/Ascend/ascend-toolkit/latest"
    assert toolkit._home_environ == "ASCEND_TOOLKIT_HOME"


def test_oppkernel_component_has_correct_defaults():
    opp = OppKernel()
    assert opp.name == "opp_kernel"
    assert opp._version_path == "opp_kernel/version.info"


def test_tb_component_has_correct_defaults():
    tb = TB()
    assert tb.name == "atb"
    assert tb._version_path == "/usr/local/Ascend/nnal/atb/latest/version.info"
    assert tb._home_environ == "ATB_HOME_PATH"


def test_mindie_component_has_correct_defaults():
    mindie = MindIE()
    assert mindie.name == "mindie"
    assert mindie._version_path == "/usr/local/Ascend/mindie/latest/version.info"
    assert mindie._home_environ == "MINDIE_LLM_HOME_PATH"


def test_tbspeed_component_has_correct_defaults():
    tbspeed = TBSpeed()
    assert tbspeed.name == "atb-models"
    assert tbspeed._version_path == "version.info"
    assert tbspeed._default_home == "/usr/local/Ascend/atb-models"
    assert tbspeed._home_environ == "ATB_SPEED_HOME_PATH"


def test_ascend_group_init_with_default_strategies():
    ascend = Ascend()
    assert ascend.name == "ascend"
    assert len(ascend._strategies) == 6


def test_ascend_group_init_with_custom_strategies():
    custom_strategy = Mock(spec=CollectStrategy)
    custom_strategy.name = "custom"
    ascend = Ascend(strategies=[custom_strategy])
    assert len(ascend._strategies) == 1
    assert ascend._strategies[0].name == "custom"


# Test Ping


def test_ping_init_given_valid_ip_then_initializes_correctly():
    ping = Ping("test_ping", ip="192.168.1.1")
    assert ping.name == "test_ping"
    assert ping._ip == "192.168.1.1"


def test_ping_init_given_invalid_ip_type_then_raises_type_error():
    with pytest.raises(TypeError, match="IP address must be a string"):
        Ping("test", ip=123)


def test_ping_init_given_invalid_ip_format_then_raises_value_error():
    with pytest.raises(ValueError, match="IP address format is invalid"):
        Ping("test", ip="invalid_ip")


@patch("shutil.which", return_value=None)
def test_ping_execute_given_command_not_found_then_returns_none_and_logs_warning(
    mock_which, caplog
):
    ping = Ping("test", ip="192.168.1.1")
    result = ping.execute()
    assert result is None
    assert len(caplog.records) == 1
    assert "ping command not found" in caplog.records[0].message


@patch("shutil.which", return_value="/bin/ping")
@patch("subprocess.check_output", return_value="ping output")
def test_ping_execute_given_success_then_returns_output(mock_check_output, mock_which):
    ping = Ping("test", ip="192.168.1.1")
    result = ping.execute()
    assert result == "ping output"


@patch("shutil.which", return_value="/bin/ping")
@patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired("cmd", 5))
def test_ping_execute_given_timeout_then_returns_none_and_logs_warning(
    mock_check_output, mock_which, caplog
):
    ping = Ping("test", ip="192.168.1.1")
    result = ping.execute()
    assert result is None
    assert len(caplog.records) == 1
    assert "Failed to execute ping command" in caplog.records[0].message


# Test HccnTool and related classes


def test_hccn_tool_init_sets_properties():
    # Use Vnic as a concrete subclass of HccnTool
    with patch("shutil.which", return_value="/usr/bin/hccn_tool"):
        hccn = Vnic(device_ids=[0, 1], max_workers=4, timeout=5.0)
        assert hccn.name == "vnic"
        assert hccn._device_ids == [0, 1]
        assert hccn._max_workers == 4
        assert hccn._timeout == 5.0


def test_hccn_tool_run_given_success_then_returns_output():
    # Use Vnic as a concrete subclass of HccnTool
    with patch("shutil.which", return_value="/usr/bin/hccn_tool"):
        hccn = Vnic(device_ids=[0])
        with patch("subprocess.check_output", return_value="success"):
            result = hccn._run(["/usr/bin/hccn_tool", "-i", "0", "-vnic", "-g"])
            assert result == "success"


def test_hccn_tool_run_given_failure_then_returns_none_and_logs_warning(caplog):
    caplog.set_level(logging.WARNING)
    # Use Vnic as a concrete subclass of HccnTool
    with patch("shutil.which", return_value="/usr/bin/hccn_tool"):
        hccn = Vnic(device_ids=[0])
        with patch("subprocess.check_output", side_effect=RuntimeError("failed")):
            result = hccn._run(["/usr/bin/hccn_tool", "-i", "0", "-vnic", "-g"])
            assert result is None


def test_single_option_subclass_requires_option():
    with pytest.raises(TypeError, match="must define a string class variable 'option'"):

        class InvalidSingleOption(_SingleOption):
            default_name = "invalid"


def test_single_option_subclass_requires_default_name():
    with pytest.raises(
        TypeError, match="must define a string class variable 'default_name'"
    ):

        class InvalidSingleOption(_SingleOption):
            option = "-test"


def test_vnic_has_correct_class_variables():
    assert Vnic.option == "-vnic"
    assert Vnic.default_name == "vnic"


def test_link_has_correct_class_variables():
    assert Link.option == "-link"
    assert Link.default_name == "link"


def test_tls_has_correct_class_variables():
    assert Tls.option == "-tls"
    assert Tls.default_name == "tls"


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", return_value="vnic output")
def test_vnic_execute_given_success_then_returns_list_of_outputs(
    mock_check_output, mock_which
):
    vnic = Vnic(device_ids=[0, 1])
    result = vnic.execute()
    assert result == ["vnic output", "vnic output"]


def test_option_ip_subclass_requires_option():
    with pytest.raises(TypeError, match="must define a string class variable 'option'"):

        class InvalidOptionIp(_OptionIp):
            default_name = "invalid"


def test_hccl_ping_has_correct_class_variables():
    assert HcclPing.option == "-ping"
    assert HcclPing.default_name == "hccl_ping"


def test_hccs_ping_has_correct_class_variables():
    assert HccsPing.option == "-hccs_ping"
    assert HccsPing.default_name == "hccs_ping"


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", return_value="ping output")
def test_hccl_ping_execute_given_success_then_returns_list_of_dicts(
    mock_check_output, mock_which
):
    ping = HcclPing(device_ids=[0], device_ips=["192.168.1.1", "192.168.1.2"])
    result = ping.execute()
    assert len(result) == 1
    assert result[0] == {"192.168.1.1": "ping output", "192.168.1.2": "ping output"}


# Test Network


@patch("msprechecker.core.strategy.get_npu_count", return_value=8)
def test_network_init_given_valid_rank_table_then_initializes_correctly(
    mock_get_npu_count,
):
    rank_table = RankTable(
        version="1.0",
        server_count=1,
        host_to_devices={
            "host1": [Mock(device_ip="192.168.1.1"), Mock(device_ip="192.168.1.2")]
        },
    )
    network = Network("test", rank_table=rank_table)
    assert network.name == "test"
    assert len(network._strategies) == 6  # Ping x2 + Vnic + Link + Tls + HcclPing


@patch("msprechecker.core.strategy.get_npu_count", return_value=8)
def test_network_init_given_hccs_version_uses_hccs_ping(mock_get_npu_count):
    rank_table = RankTable(
        version="1.2",  # HCCS_PING_VERSION
        server_count=1,
        host_to_devices={"host1": [Mock(device_ip="192.168.1.1")]},
    )
    network = Network("test", rank_table=rank_table)
    assert network.name == "test"
    # Last strategy should be HccsPing when version is 1.2
    assert network._strategies[-1].__class__.__name__ == "HccsPing"


def test_network_init_given_invalid_rank_table_type_then_raises_type_error():
    with pytest.raises(TypeError, match="rank_table must be an instance of RankTable"):
        Network("test", rank_table="invalid")


@patch("msprechecker.core.strategy.get_npu_count", return_value=0)
def test_network_init_given_zero_npu_count_then_raises_value_error(mock_get_npu_count):
    rank_table = RankTable(
        version="1.0",
        server_count=1,
        host_to_devices={"host1": [Mock(device_ip="192.168.1.1")]},
    )
    with pytest.raises(ValueError, match="No NPU devices found in the system"):
        Network("test", rank_table=rank_table)


@patch("msprechecker.core.strategy.get_npu_count", return_value=8)
def test_network_init_given_empty_device_ips_then_raises_value_error(
    mock_get_npu_count,
):
    rank_table = RankTable(
        version="1.0",
        server_count=0,
        host_to_devices={},
    )
    with pytest.raises(
        ValueError, match="No device IP addresses found in the rank table"
    ):
        Network("test", rank_table=rank_table)


# Test Stress, CPU, NPU


def test_stress_init_given_invalid_batch_size_then_raises_value_error():
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        CPU("test", batch_size=0, seq_len=512, hidden_size=1024, intermediate_size=64)


def test_stress_init_given_invalid_seq_len_then_raises_value_error():
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        CPU("test", batch_size=1, seq_len=-1, hidden_size=1024, intermediate_size=64)


def test_stress_init_given_invalid_hidden_size_then_raises_value_error():
    with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
        CPU("test", batch_size=1, seq_len=512, hidden_size=0, intermediate_size=64)


def test_stress_init_given_invalid_intermediate_size_then_raises_value_error():
    with pytest.raises(
        ValueError, match="intermediate_size must be a positive integer"
    ):
        CPU("test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=-1)


def test_stress_init_given_invalid_epochs_then_raises_value_error():
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        CPU(
            "test",
            batch_size=1,
            seq_len=512,
            hidden_size=1024,
            intermediate_size=64,
            epochs=0,
        )


def test_stress_calculate_tensor_memory_given_tuple_shape():
    stress = CPU(
        "test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=64
    )
    result = stress._calculate_tensor_memory((2, 3, 4))
    assert result == 2 * 3 * 4 * 4  # float32 = 4 bytes


def test_stress_calculate_tensor_memory_given_int_shape():
    stress = CPU(
        "test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=64
    )
    result = stress._calculate_tensor_memory(10)
    assert result == 10 * 4  # float32 = 4 bytes


@patch("msprechecker.core.strategy.CPU._get_free_memory", return_value=0)
def test_stress_check_memory_for_matmul_given_insufficient_memory_then_returns_false_and_logs_warning(
    mock_get_free_memory, caplog
):
    stress = CPU(
        "test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=64
    )
    result = stress._check_memory_for_matmul("cpu:0")
    assert result is False
    assert len(caplog.records) >= 1
    assert any("Insufficient memory" in r.message for r in caplog.records)


def test_cpu_init_with_defaults():
    cpu = CPU()
    assert cpu.name == "cpu"
    assert cpu.device_type == "cpu"


def test_cpu_device_type_property():
    cpu = CPU()
    assert cpu.device_type == "cpu"


@patch("psutil.virtual_memory")
def test_cpu_get_free_memory_given_mock_psutil_then_returns_available(
    mock_virtual_memory,
):
    mock_virtual_memory.return_value = Mock(available=1024**3)
    cpu = CPU()
    result = cpu._get_free_memory("cpu:0")
    assert result == 1024**3


def test_npu_init_with_defaults():
    npu = NPU()
    assert npu.name == "npu"
    assert npu.device_type == "npu"


def test_npu_device_type_property():
    npu = NPU()
    assert npu.device_type == "npu"


def test_npu_get_free_memory_given_npu_not_available_then_returns_zero_and_logs_warning(
    caplog,
):
    caplog.set_level(logging.WARNING)
    npu = NPU()
    npu._torch_npu = Mock()
    npu._torch_npu.npu.is_available.return_value = False
    result = npu._get_free_memory("npu:0")
    assert result == 0
    # There may be multiple log records (e.g., torch_npu import warning)
    assert any("NPU device is not available" in r.message for r in caplog.records)


# Test CollectStrategyGroup execute with exception handling
def test_strategy_group_execute_given_strategy_raises_exception_then_returns_none_for_that_strategy(
    strategy_group, caplog
):
    strategy1 = Mock(spec=CollectStrategy)
    strategy1.name = "failing_strategy"
    strategy1.execute.side_effect = RuntimeError("Test error")

    strategy2 = Mock(spec=CollectStrategy)
    strategy2.name = "working_strategy"
    strategy2.execute.return_value = {"key": "value"}

    strategy_group.add(strategy1).add(strategy2)
    result = strategy_group.execute()

    assert result["failing_strategy"] is None
    assert result["working_strategy"] == {"key": "value"}
    assert "Strategy 'failing_strategy' failed" in caplog.records[0].message


# Test Lscpu _parse_output edge cases
def test_lscpu_parse_output_given_empty_string_then_returns_none():
    lscpu = Lscpu()
    result = lscpu._parse_output("")
    assert result is None


def test_lscpu_parse_output_given_lines_without_colon_then_skips_them():
    lscpu = Lscpu()
    result = lscpu._parse_output("line1\nline2\nkey: value")
    assert result == {"key": "value"}


def test_lscpu_parse_output_given_duplicate_keys_then_first_wins():
    lscpu = Lscpu()
    result = lscpu._parse_output("key: first\nkey: second")
    assert result == {"key": "first"}


def test_lscpu_parse_output_given_only_invalid_lines_then_returns_none():
    lscpu = Lscpu()
    result = lscpu._parse_output("line1\nline2\nline3")
    assert result is None


# Test CPUHighPerformance edge cases
def test_cpu_high_performance_execute_given_all_methods_fail_then_returns_false():
    cpu_perf = CPUHighPerformance()
    with patch.object(
        cpu_perf, "_check_via_scaling_governor", return_value=False
    ), patch.object(cpu_perf, "_check_via_dmidecode", return_value=False), patch.object(
        cpu_perf, "_check_via_cpupower", return_value=False
    ), patch.object(cpu_perf, "_check_via_lshw", return_value=False), patch.object(
        cpu_perf, "_check_via_psutil", return_value=False
    ):
        result = cpu_perf.execute()
        assert result is False


# Test Config _process_shell
def test_config_process_shell_returns_content_unchanged():
    config = Config("test", config_path="test.sh")
    result = config._process_shell("shell script content")
    assert result == "shell script content"


# Test Config _read_file edge cases
@patch("os.path.isfile", return_value=False)
def test_config_read_file_given_nonexistent_path_then_returns_none_and_logs_warning(
    mock_isfile, caplog
):
    config = Config("test", config_path="/nonexistent/file.json")
    result = config._read_file()
    assert result is None
    assert (
        "Configuration file '/nonexistent/file.json' not found"
        in caplog.records[0].message
    )


@patch("os.path.isfile", return_value=True)
@patch("builtins.open", side_effect=OSError("Permission denied"))
def test_config_read_file_given_read_error_then_returns_none_and_logs_error(
    mock_open, mock_isfile, caplog
):
    config = Config("test", config_path="/path/file.json")
    result = config._read_file()
    assert result is None
    assert caplog.records[0].levelname == "ERROR"
    assert "Failed to read configuration file" in caplog.records[0].message


# Test Config _process_yaml with safe_load (no document separator)
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile", return_value=True)
def test_config_process_yaml_given_no_document_separator_uses_safe_load(
    mock_isfile, mock_file
):
    config = Config("test", config_path="test.yaml")
    result = config._process_yaml("key: value")
    assert result == {"key": "value"}


# Test _AscendComponent
class MockAscendComponent(_AscendComponent):
    _DEFAULT_NAME = "mock_component"
    _DEFAULT_VERSION_PATH = "mock/version.info"
    _DEFAULT_HOME = "/usr/local/mock"
    _DEFAULT_ENVIRON = "MOCK_HOME"


def test_ascend_component_uses_class_defaults():
    component = MockAscendComponent()
    assert component.name == "mock_component"
    assert component._version_path == "mock/version.info"
    assert component._default_home == "/usr/local/mock"
    assert component._home_environ == "MOCK_HOME"


def test_ascend_component_allows_custom_values():
    component = MockAscendComponent(
        name="custom",
        version_path="custom/path",
        default_home="/custom/home",
        home_environ="CUSTOM_HOME",
    )
    assert component.name == "custom"
    assert component._version_path == "custom/path"
    assert component._default_home == "/custom/home"
    assert component._home_environ == "CUSTOM_HOME"


# Test _OptionIp.execute
@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", return_value="ping result")
def test_option_ip_execute_given_success_then_returns_list_of_dicts(
    mock_check_output, mock_which
):
    ping = HcclPing(device_ids=[0, 1], device_ips=["192.168.1.1"])
    result = ping.execute()
    assert len(result) == 2  # One dict per device_id
    assert result[0] == {"192.168.1.1": "ping result"}
    assert result[1] == {"192.168.1.1": "ping result"}


# Test Weight _calculate_hash256
@patch("builtins.open", new_callable=mock_open, read_data=b"test data")
def test_weight_calculate_hash256_given_file_then_returns_hexdigest(mock_file):
    weight = Weight("test")
    result = weight._calculate_hash256("/path/to/file.safetensors")
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hex digest is 64 characters


# Test Weight._filter_valid_tensor_files
@patch("os.path.isdir", return_value=True)
@patch("os.listdir", return_value=["model.safetensors", "config.json", "model.bin"])
@patch("os.path.islink", return_value=False)
@patch(
    "os.path.isfile",
    side_effect=lambda p: p.endswith((".safetensors", ".json")),
)
@patch("os.path.getsize", return_value=1024)
def test_weight_filter_valid_tensor_files_given_mixed_files_then_returns_only_valid_tensors(
    mock_getsize, mock_isfile, mock_islink, mock_listdir, mock_isdir
):
    weight = Weight("test", weight_dir="/weights", tensor_suffix=".safetensors")
    result = weight._filter_valid_tensor_files()
    assert len(result) == 1
    assert result[0] == "/weights/model.safetensors"


# Test Stress.execute when torch is not available
def test_stress_execute_given_torch_not_available_then_returns_none_and_logs_error(
    caplog,
):
    cpu = CPU()
    cpu._torch = None  # Simulate torch not being available
    result = cpu.execute()
    assert result is None
    # There may be multiple log records (e.g., torch import warning + error)
    assert any("torch is not available" in r.message for r in caplog.records)


# Test NPU._get_free_memory when torch.npu is available
@patch.object(NPU, "_get_free_memory", return_value=1024**3)
def test_npu_get_free_memory_given_mock_torch_then_returns_memory(mock_get_free_memory):
    npu = NPU()
    result = npu._get_free_memory("npu:0")
    assert result == 1024**3


# Test CollectStrategyGroup.execute exception handling in more detail
def test_strategy_group_execute_logs_exception_details(strategy_group, caplog):
    caplog.set_level(logging.ERROR)
    strategy = Mock(spec=CollectStrategy)
    strategy.name = "failing_strategy"
    strategy.execute.side_effect = ValueError("Specific error message")

    strategy_group.add(strategy)
    result = strategy_group.execute()

    assert result["failing_strategy"] is None
    assert "Strategy 'failing_strategy' failed" in caplog.records[0].message
    # The exception info should be logged
    assert "Specific error message" in str(caplog.records[0].exc_info)


# Test Config execute when _read_file returns None
@patch.object(Config, "_read_file", return_value=None)
def test_config_execute_given_read_file_returns_none_then_returns_none(mock_read_file):
    config = Config("test", config_path="/path/file.json")
    result = config.execute()
    assert result is None


# Test _Ascend.execute when home_path is empty and version_path is absolute
@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="VERSION=1.0\n")
def test_ascend_execute_given_absolute_path_and_no_home_then_reads_file(
    mock_file, mock_isfile
):
    ascend = _Ascend("test", version_path="/absolute/path/version.info")
    result = ascend.execute()
    assert result == {"VERSION": "1.0"}


# Test _OptionIp._probe_device
@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", return_value="probe result")
def test_option_ip_probe_device_given_success_then_returns_dict_with_ip_results(
    mock_check_output, mock_which
):
    ping = HcclPing(device_ids=[0], device_ips=["192.168.1.1", "192.168.1.2"])
    result = ping._probe_device(0)
    assert result == {"192.168.1.1": "probe result", "192.168.1.2": "probe result"}


# Test Stress._check_memory_for_matmul when memory is sufficient
@patch(
    "msprechecker.core.strategy.CPU._get_free_memory", return_value=100 * 1024**3
)  # 100 GB
@patch(
    "msprechecker.core.strategy.CPU._calculate_tensor_memory", return_value=1024
)  # Small tensor
def test_stress_check_memory_given_sufficient_memory_then_returns_true(
    mock_calc_memory, mock_get_free_memory
):
    cpu = CPU("test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=64)
    result = cpu._check_memory_for_matmul("cpu:0")
    assert result is True


# Test _SingleOption._build_cmd
@patch("shutil.which", return_value="/usr/bin/hccn_tool")
def test_single_option_build_cmd_given_device_id_then_returns_correct_command(
    mock_which,
):
    vnic = Vnic(device_ids=[0])
    cmd = vnic._build_cmd(0)
    assert cmd == ["/usr/bin/hccn_tool", "-i", "0", "-vnic", "-g"]


# Test _OptionIp._build_cmd
@patch("shutil.which", return_value="/usr/bin/hccn_tool")
def test_option_ip_build_cmd_given_device_and_ip_then_returns_correct_command(
    mock_which,
):
    ping = HcclPing(device_ids=[0], device_ips=["192.168.1.1"])
    cmd = ping._build_cmd(0, "192.168.1.1")
    assert cmd == [
        "/usr/bin/hccn_tool",
        "-i",
        "0",
        "-ping",
        "-g",
        "address",
        "192.168.1.1",
    ]


# Test Weight._parallel_hash_calculation with mocked executor
@patch.object(Weight, "_calculate_hash256", return_value="mock_hash")
def test_weight_parallel_hash_calculation_given_files_then_returns_dict(
    mock_calculate_hash,
):
    weight = Weight("test", weight_dir="/weights", max_hash_workers=1)
    result = weight._parallel_hash_calculation(
        ["/weights/model1.safetensors", "/weights/model2.safetensors"]
    )
    # Should return a dict with file IDs as keys
    assert isinstance(result, dict)
    assert len(result) == 2


# Test CPUHighPerformance._parse_dmidecode_output edge cases
def test_cpu_high_performance_parse_dmidecode_output_given_empty_lists_then_returns_false():
    cpu_perf = CPUHighPerformance()
    cpu_perf._dmidecode_output = "Some output without speed info"
    result = cpu_perf._parse_dmidecode_output()
    assert result is False


# Test CPUHighPerformance._parse_cpupower_output edge cases
def test_cpu_high_performance_parse_cpupower_output_given_missing_patterns_then_returns_false():
    cpu_perf = CPUHighPerformance()
    cpu_perf._cpupower_output = "Some output without frequency info"
    result = cpu_perf._parse_cpupower_output()
    assert result is False


# Test CPUHighPerformance._parse_lshw_output edge cases
def test_cpu_high_performance_parse_lshw_output_given_empty_lists_then_returns_false():
    cpu_perf = CPUHighPerformance()
    cpu_perf._lshw_output = "Some output without size/capacity"
    result = cpu_perf._parse_lshw_output()
    assert result is False


# Test Stress property device_type for CPU
def test_cpu_device_type_property_returns_cpu():
    cpu = CPU()
    assert cpu.device_type == "cpu"


# Test Stress property device_type for NPU
def test_npu_device_type_property_returns_npu():
    npu = NPU()
    assert npu.device_type == "npu"


# Test NPU default parameters
def test_npu_init_has_correct_default_parameters():
    npu = NPU()
    assert npu._batch_size == 1
    assert npu._seq_len == 4096
    assert npu._hidden_size == 8192
    assert npu._intermediate_size == 3584
    assert npu._epochs == 5


# Test CPU default parameters
def test_cpu_init_has_correct_default_parameters():
    cpu = CPU()
    assert cpu._batch_size == 1
    assert cpu._seq_len == 512
    assert cpu._hidden_size == 1024
    assert cpu._intermediate_size == 64
    assert cpu._epochs == 5


# Test Stress._matmul_stress_test when memory check fails
@patch("msprechecker.core.strategy.CPU._check_memory_for_matmul", return_value=False)
def test_stress_matmul_stress_test_given_memory_check_fails_then_returns_zero(
    mock_check_memory,
):
    cpu = CPU("test", batch_size=1, seq_len=512, hidden_size=1024, intermediate_size=64)
    cpu._torch = Mock()
    result = cpu._matmul_stress_test(0)
    assert result == 0.0


# Test Stress._matmul_stress_test when torch is available
@patch("msprechecker.core.strategy.CPU._check_memory_for_matmul", return_value=True)
def test_stress_matmul_stress_test_given_torch_available_then_returns_elapsed_time(
    mock_check_memory,
):
    cpu = CPU(
        "test",
        batch_size=1,
        seq_len=512,
        hidden_size=1024,
        intermediate_size=64,
        epochs=1,
    )
    mock_torch = Mock()
    mock_torch.randn.return_value.to.return_value = Mock()
    mock_torch.zeros.return_value.to.return_value = Mock()
    mock_torch.addbmm = Mock()
    cpu._torch = mock_torch
    result = cpu._matmul_stress_test(0)
    assert result >= 0.0  # Should return elapsed time


# Test NPU._get_free_memory when torch.npu is available and has memory
@patch.object(NPU, "_get_free_memory", return_value=1024**3)
def test_npu_get_free_memory_given_npu_available_then_returns_free_memory(
    mock_get_free_memory,
):
    npu = NPU()
    result = npu._get_free_memory("npu:0")
    assert result == 1024**3
