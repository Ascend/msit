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
from unittest.mock import Mock, mock_open, patch

import pytest

from msprechecker.core.strategy import CollectStrategy, CollectStrategyGroup


@pytest.fixture
def strategy_group():
    return CollectStrategyGroup("test")


def test_cant_instantiate_abstract_collect_strategy_given_base_class_when_init_then_raises_type_error():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class CollectStrategy with abstract methods execute",
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
        TypeError, match="must be instances of CollectStrategy or CollectStrategyGroup"
    ):
        CollectStrategyGroup("test", strategies=invalid_strategies)


def test_init_strategy_group_given_valid_strategies_then_initializes_correctly(mocker):
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
        match="must be an instance of CollectStrategy or CollectStrategyGroup",
    ):
        strategy_group.add(invalid_strategy)


def test_add_given_valid_strategy_and_group_then_adds_to_internal_list(
    strategy_group, mocker
):
    strategy1 = Mock(spec=CollectStrategy)
    strategy2 = Mock(spec=CollectStrategyGroup)
    strategy_group.add(strategy1).add(strategy2)
    assert strategy_group.name == "test"
    assert strategy_group._strategies == [strategy1, strategy2]


def test_execute_with_mixed_strategies_given_unique_names_then_returns_merged_result_dict(
    strategy_group, mocker
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


def test_execute_with_duplicate_strategy_names_then_overwrites_with_last_result(
    strategy_group, mocker
):
    strategy1 = Mock(spec=CollectStrategyGroup)
    strategy2 = Mock(spec=CollectStrategy)
    strategy1.name = "same_name"
    strategy1.execute.return_value = {"a": "b"}
    strategy2.name = "same_name"
    strategy2.execute.return_value = {"c": "d"}

    strategy_group.add(strategy1).add(strategy2)
    assert strategy_group.execute() == {strategy2.name: strategy2.execute.return_value}


# Test Env
import os

from msprechecker.core.strategy import Env


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
from msprechecker.core.strategy import Lscpu


@patch("shutil.which", return_value=None)
def test_execute_given_lscpu_command_not_found_when_execute_then_returns_empty_dict_and_logs_warning(
    mock_which, caplog
):
    assert Lscpu().execute() == {}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "lscpu command not found in system PATH"


@patch("shutil.which", return_value="random_path")
@patch("subprocess.check_output", side_effect=RuntimeError)
def test_execute_given_lscpu_command_fails_when_execute_then_returns_empty_dict_and_logs_warning(
    mock_check_output, mock_which, caplog
):
    assert Lscpu().execute() == {}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Failed to execute lscpu command: "


@patch("shutil.which", return_value="random_path")
@patch("subprocess.check_output", return_value="a: b\nc\nd: e")
def test_execute_given_valid_lscpu_output_when_parse_then_returns_correct_parsed_dictionary(
    mock_check_output, mock_which
):
    assert Lscpu().execute() == {"a": "b", "d": "e"}


# Test CPUHighPerformance
from msprechecker.core.strategy import CPUHighPerformance


@patch("psutil.cpu_freq", return_value=None)
def test_check_by_psutil_given_no_cpu_frequency_when_check_then_returns_false(
    mock_psutil, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_psutil()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert (
        caplog.records[0].message
        == "Unable to get CPU frequency information via psutil"
    )


@patch("psutil.cpu_freq", return_value=Mock(current=2000, max=3000))
def test_check_by_psutil_given_current_freq_below_max_when_check_then_returns_false(
    mock_which,
):
    assert not CPUHighPerformance()._check_by_psutil()


@patch("psutil.cpu_freq", return_value=Mock(current=3000, max=3000))
def test_check_by_psutil_given_current_freq_equals_max_when_check_then_returns_true(
    mock_which,
):
    assert not CPUHighPerformance()._check_by_psutil()


@patch("os.cpu_count", return_value=None)
def test_check_by_scaling_governor_given_cpu_count_is_none_when_check_then_returns_false_and_logs_debug_message(
    mock_cpu_count, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Unable to determine CPU count"


@patch("os.path.isfile", return_value=False)
@patch("os.cpu_count", return_value=1)
def test_check_by_scaling_governor_given_cpu_count_valid_but_scaling_file_not_found_when_check_then_returns_false_and_logs_debug_message(
    mock_cpu_count, mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Scaling governor file not found for CPU core 0"


@patch("builtins.open", new_callable=mock_open, read_data="normal")
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_by_scaling_governor_given_cpu_count_valid_and_file_contains_non_performance_value_when_check_then_returns_false_and_logs_debug_message(
    mock_cpu_count, mock_isfile, mock_file, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_scaling_governor()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert (
        caplog.records[0].message
        == "CPU core 0 scaling governor is not set to performance mode"
    )


@patch("builtins.open", side_effect=OSError)
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_by_scaling_governor_given_cpu_count_valid_and_file_open_fails_when_check_then_returns_false_without_logging(
    mock_cpu_count, mock_isfile, mock_cpu_freq
):
    assert not CPUHighPerformance()._check_by_scaling_governor()


@patch("builtins.open", new_callable=mock_open, read_data="performance")
@patch("os.path.isfile", return_value=True)
@patch("os.cpu_count", return_value=1)
def test_check_by_scaling_governor_given_governor_file_exists_and_content_is_performance_when_check_then_returns_true(
    mock_cpu_count, mock_isfile, mock_cpu_freq
):
    assert CPUHighPerformance()._check_by_scaling_governor() == True


@patch("shutil.which", return_value=None)
def test_check_by_cpupower_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_cpupower()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "cpupower command not found in system PATH"


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_by_cpupower_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_cpupower()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Failed to execute cpupower command: "


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_cpupower_given_current_freq_below_max_limit_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 1.00 GHz
"""
    assert not CPUHighPerformance()._check_by_cpupower()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_cpupower_given_output_missing_current_freq_line_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
"""
    assert not CPUHighPerformance()._check_by_cpupower()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_cpupower_given_current_freq_equals_max_limit_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 3.00 GHz
"""
    assert CPUHighPerformance()._check_by_cpupower() == True


@patch("shutil.which", return_value=None)
def test_check_by_dmidecode_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_dmidecode()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "dmidecode command not found in system PATH"


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_by_dmidecode_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_dmidecode()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Failed to execute dmidecode command: "


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_dmidecode_given_output_missing_current_speed_line_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
"""
    assert not CPUHighPerformance()._check_by_dmidecode()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_dmidecode_given_current_speed_is_zero_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
Current Speed: 0 MHz
"""
    assert not CPUHighPerformance()._check_by_dmidecode()


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_dmidecode_given_current_speed_equals_max_speed_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
Processor Information
Max Speed: 3000 MHz
Current Speed: 3000 MHz
"""
    assert CPUHighPerformance()._check_by_dmidecode() == True


@patch("shutil.which", return_value=None)
def test_check_by_lshw_given_command_not_found_when_check_then_returns_false(
    mock_which, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_lshw()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "lshw command not found in system PATH"


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_lshw_given_output_missing_capacity_key_when_check_then_returns_false(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
capacity: 1
"""
    assert not CPUHighPerformance()._check_by_lshw()


@patch("subprocess.check_output", side_effect=RuntimeError)
@patch("shutil.which", return_value="random_path")
def test_check_by_lshw_given_subprocess_execution_fails_when_check_then_returns_false(
    mock_which, mock_check_output, caplog
):
    caplog.set_level(logging.DEBUG)
    assert not CPUHighPerformance()._check_by_lshw()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "Failed to execute lshw command: "


@patch("subprocess.check_output")
@patch("shutil.which", return_value="random_path")
def test_check_by_lshw_given_output_contains_size_and_capacity_keys_when_check_then_returns_true(
    mock_which, mock_check_output
):
    mock_check_output.return_value = """
size: 1
capacity: 1
"""
    assert CPUHighPerformance()._check_by_lshw() == True


# Test VirtualMachine
from msprechecker.core.strategy import VirtualMachine


@patch("os.path.isfile", return_value=False)
def test_execute_given_cpuinfo_file_does_not_exist_when_check_then_returns_false(
    mock_isfile, caplog
):
    caplog.set_level(logging.DEBUG)
    assert VirtualMachine().execute() == False
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[0].message == "/proc/cpuinfo file not found"


@patch("builtins.open", side_effect=IOError)
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_but_open_fails_when_check_then_returns_false(
    mock_isfile, mock_file, caplog
):
    assert not VirtualMachine().execute()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Failed to read /proc/cpuinfo file: "


@patch("builtins.open", new_callable=mock_open, read_data="其他的玩意儿")
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_without_hypervisor_keyword_when_check_then_returns_false(
    mock_isfile, mock_file
):
    assert VirtualMachine().execute() == False


@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc\nhypervisor")
@patch("os.path.isfile", return_value=True)
def test_execute_given_cpuinfo_file_exists_containing_hypervisor_keyword_when_check_then_returns_true(
    mock_isfile, mock_file
):
    assert VirtualMachine().execute() == True


# Test TransparentHugepage
from msprechecker.core.strategy import TransparentHugepage


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
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "Failed to read transparent hugepage configuration: "
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
from msprechecker.core.strategy import Kernel


@patch("platform.uname")
def test_execute_given_mock_platform_uname_call_when_execute_then_returns_uname_asdict_result(
    mock_uname,
):
    test_data = dict(
        system="a", node="b", release="c", version="d", machine="e", processor="f"
    )
    mock_uname.return_value._asdict = Mock(return_value=test_data)
    assert Kernel().execute() == test_data


# Test PageSize
from msprechecker.core.strategy import PageSize


@patch("os.sysconf", side_effect=OSError)
def test_execute_given_sysconf_call_raises_oserror_when_execute_then_returns_none_and_logs_warning(
    mock_sysconf, caplog
):
    assert PageSize().execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Failed to get system page size: "


@patch("os.sysconf", return_value=2)
def test_execute_given_sysconf_call_returns_integer_value_when_execute_then_returns_that_value(
    mock_sysconf,
):
    assert PageSize().execute() == 2


# Test Sys
from msprechecker.core.strategy import Sys


def test_sys_initialization_with_default_strategies_then_has_correct_name_and_strategy_count():
    sys_strategy = Sys()
    assert sys_strategy.name == "sys"
    assert len(sys_strategy._strategies) == 6


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


from msguard import GlobalConfig

# Test Config
from msprechecker.core.strategy import Config


def test_given_empty_path_when_execute_then_logs_warning_and_returns_none(caplog):
    assert Config("test", "").execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Configuration path is empty or not provided"


def test_given_invalid_path_permissions_when_execute_with_global_flag_false_then_logs_permission_warning_and_returns_none(
    caplog,
):
    GlobalConfig.custom_return = False
    assert Config("test", "a").execute() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "Expected 'a' to be a regular file and not a soft link and a readable path and not writable to group or others and consistent with the current user and reasonable on its size"
    )
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_mocked_file_with_unsupported_format_when_execute_then_logs_unsupported_format_warning_and_returns_raw_content(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "random_data"
    assert Config("test", "a").execute() == "random_data"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "Unsupported configuration file format: 'a'"
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_mocked_json_file_with_parse_error_when_execute_then_logs_json_parse_error_and_returns_raw_content(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "random_data"
    assert Config("test", "a.json").execute() == "random_data"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "Failed to parse JSON configuration file 'a.json': Expecting value: line 1 column 1 (char 0)"
    )
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_mocked_yaml_file_with_parse_error_when_execute_then_logs_yaml_parse_error_and_returns_raw_content(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "- a\nb"
    assert Config("test", "a.yaml").execute() == "- a\nb"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        "Failed to parse YAML configuration file 'a.yaml': while scanning a simple key"
        in caplog.records[0].message
    )
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_mocked_yaml_file_with_document_separator_parse_error_when_execute_then_logs_generic_yaml_parse_error_and_returns_raw_content(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "--- - a\nb"
    assert Config("test", "a.yaml").execute() == "--- - a\nb"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Failed to parse YAML configuration file" in caplog.records[0].message
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_valid_mocked_json_file_when_execute_then_returns_parsed_dict(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = '{"a": "b"}'
    assert Config("test", "a.json").execute() == {"a": "b"}
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_valid_single_document_yaml_file_when_execute_then_returns_parsed_dict(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "a: b"
    assert Config("test", "a.yaml").execute() == {"a": "b"}
    GlobalConfig.custom_return = None


@patch("builtins.open", new_callable=mock_open)
def test_given_valid_multi_document_yaml_file_when_execute_then_returns_list_of_dicts(
    mock_file, caplog
):
    GlobalConfig.custom_return = True
    mock_file.return_value.read.return_value = "a: b\n---\nc : d"
    assert Config("test", "a.yaml").execute() == [{"a": "b"}, {"c": "d"}]
    GlobalConfig.custom_return = None


# test Weight
