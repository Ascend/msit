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

import json
import os
import stat
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

try:
    from importlib.metadata import PackageNotFoundError
except ImportError:
    from importlib_metadata import PackageNotFoundError

from msprechecker.util import (
    ConnMode,
    detect_framework,
    Framework,
    get_conn_mode,
    get_current_ip_and_addr,
    get_npu_count,
    get_npu_memory,
    get_npu_type,
    get_pkg_version,
    is_in_container,
    is_valid_ip,
    is_valid_ip_strict,
    MindIEProbe,
    NpuType,
    parse_rank_table,
    probe_hardware,
    RankTable,
    RankTableParseError,
    resolve_weight_dir,
    SGLangProbe,
    singleton,
    VLLMProbe,
    WeightDirNotFoundError,
)


# =============================================================================
# Tests for constant utilities
# =============================================================================


def test_log_levels_defined():
    from msprechecker.util import LOG_FORMAT, LOG_LEVELS

    assert "debug" in LOG_LEVELS
    assert "info" in LOG_LEVELS
    assert "warning" in LOG_LEVELS
    assert "error" in LOG_LEVELS
    assert "critical" in LOG_LEVELS
    assert LOG_FORMAT == "[%(levelname)s] [%(name)s] %(message)s"


# =============================================================================
# Tests for is_in_container
# =============================================================================


@patch("os.path.exists", return_value=True)
def test_is_in_container_given_dockerenv_file_exists_then_returns_true(mock_exists):
    result = is_in_container()
    assert result is True


@patch("os.path.exists", return_value=False)
@patch("os.path.join", return_value="/proc/1/sched")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="systemd (1, #threads: 1)\n",
)
def test_is_in_container_given_systemd_first_process_then_returns_false(
    mock_file, mock_join, mock_exists
):
    result = is_in_container()
    assert result is False


@patch("os.path.exists", return_value=False)
@patch("os.path.join", return_value="/proc/1/sched")
@patch("builtins.open", side_effect=Exception("read error"))
def test_is_in_container_given_sched_read_error_then_returns_true(
    mock_file, mock_join, mock_exists
):
    result = is_in_container()
    assert result is True


@patch("os.path.exists", return_value=False)
@patch("os.path.join", return_value="/proc/1/sched")
@patch("builtins.open", new_callable=mock_open, read_data="init (1, #threads: 1)\n")
def test_is_in_container_given_non_systemd_first_process_then_returns_true(
    mock_file, mock_join, mock_exists
):
    result = is_in_container()
    assert result is True


# =============================================================================
# Tests for singleton decorator
# =============================================================================


def test_singleton_decorator_creates_single_instance():
    @singleton
    class TestClass:
        def __init__(self):
            self.value = 0

    instance1 = TestClass()
    instance2 = TestClass()
    instance1.value = 42

    assert instance1 is instance2
    assert instance2.value == 42


def test_singleton_decorator_with_different_classes():
    @singleton
    class ClassA:
        pass

    @singleton
    class ClassB:
        pass

    a1 = ClassA()
    a2 = ClassA()
    b1 = ClassB()
    b2 = ClassB()

    assert a1 is a2
    assert b1 is b2
    assert a1 is not b1


# =============================================================================
# Tests for is_valid_ip
# =============================================================================


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("255.255.255.255", True),
        ("0.0.0.0", True),
        ("256.1.1.1", False),
        ("192.168.1", False),
        ("192.168.1.1.1", False),
        ("not_an_ip", False),
        ("", False),
        ("192.168.1.a", False),
    ],
)
def test_is_valid_ip_given_various_inputs_then_returns_expected(ip, expected):
    result = is_valid_ip(ip)
    assert result == expected


# =============================================================================
# Tests for get_pkg_version
# =============================================================================


@patch("msprechecker.util.version", return_value="1.2.3")
def test_get_pkg_version_given_valid_package_then_returns_version(mock_version):
    result = get_pkg_version("some_package")
    assert str(result) == "1.2.3"


@patch(
    "msprechecker.util.version",
    side_effect=PackageNotFoundError("Package not found"),
)
def test_get_pkg_version_given_invalid_package_then_returns_none(mock_version):
    result = get_pkg_version("nonexistent_package")
    assert result is None


# =============================================================================
# Tests for get_current_ip_and_addr
# =============================================================================


@patch("psutil.net_if_addrs")
def test_get_current_ip_and_addr_given_valid_interface_then_returns_interface_and_ip(
    mock_net_if_addrs,
):
    mock_addr = Mock()
    mock_addr.family = 2  # socket.AF_INET
    mock_addr.address = "192.168.1.100"

    mock_net_if_addrs.return_value = {"eth0": [mock_addr]}

    interface, ip = get_current_ip_and_addr()
    assert interface == "eth0"
    assert ip == "192.168.1.100"


@patch("psutil.net_if_addrs")
def test_get_current_ip_and_addr_given_docker_interface_then_skips_it(
    mock_net_if_addrs,
):
    mock_addr = Mock()
    mock_addr.family = 2
    mock_addr.address = "172.17.0.1"

    mock_net_if_addrs.return_value = {"docker0": [mock_addr]}

    interface, ip = get_current_ip_and_addr()
    assert interface == ""
    assert ip == ""


@patch("psutil.net_if_addrs")
def test_get_current_ip_and_addr_given_loopback_interface_then_skips_it(
    mock_net_if_addrs,
):
    mock_addr = Mock()
    mock_addr.family = 2
    mock_addr.address = "127.0.0.1"

    mock_net_if_addrs.return_value = {"lo": [mock_addr]}

    interface, ip = get_current_ip_and_addr()
    assert interface == ""
    assert ip == ""


@patch("psutil.net_if_addrs")
def test_get_current_ip_and_addr_given_no_interfaces_then_returns_empty(
    mock_net_if_addrs,
):
    mock_net_if_addrs.return_value = {}

    interface, ip = get_current_ip_and_addr()
    assert interface == ""
    assert ip == ""


# =============================================================================
# Tests for get_npu_count
# =============================================================================


@patch("os.stat")
def test_get_npu_count_given_no_devices_then_returns_zero(mock_stat):
    mock_stat.side_effect = OSError("No such file")
    result = get_npu_count()
    assert result == 0


@patch("os.stat")
def test_get_npu_count_given_three_devices_then_returns_three(mock_stat):
    def side_effect(path):
        if path in ["/dev/davinci0", "/dev/davinci1", "/dev/davinci2"]:
            mock_result = Mock()
            mock_result.st_mode = stat.S_IFCHR
            return mock_result
        raise OSError("No such file")

    mock_stat.side_effect = side_effect
    result = get_npu_count()
    assert result == 3


@patch("os.stat")
def test_get_npu_count_given_non_character_device_then_stops(mock_stat):
    def side_effect(path):
        if path == "/dev/davinci0":
            mock_result = Mock()
            mock_result.st_mode = stat.S_IFREG  # Regular file, not char device
            return mock_result
        raise OSError("No such file")

    mock_stat.side_effect = side_effect
    result = get_npu_count()
    assert result == 0


# =============================================================================
# Tests for get_npu_type
# =============================================================================


@patch("shutil.which", return_value=None)
def test_get_npu_type_given_lspci_not_found_then_returns_unknown(mock_which):
    result = get_npu_type()
    assert result == NpuType.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch("subprocess.check_output", side_effect=RuntimeError("lspci failed"))
def test_get_npu_type_given_lspci_fails_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device d100",
)
def test_get_npu_type_given_d100_device_then_returns_d100(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.d100


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device d500",
)
def test_get_npu_type_given_d500_device_then_returns_d500(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.d500


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device unknown",
)
def test_get_npu_type_given_unknown_device_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Device d100\n01:00.0 Accelerator: Huawei Device d500",
)
def test_get_npu_type_given_inconsistent_devices_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 VGA controller: Some other device",
)
def test_get_npu_type_given_no_accelerator_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.UNKNOWN


# =============================================================================
# Tests for get_npu_memory
# =============================================================================


@patch("shutil.which", return_value=None)
def test_get_npu_memory_given_npu_smi_not_found_then_returns_none(mock_which):
    result = get_npu_memory()
    assert result is None


@patch("shutil.which", return_value="/usr/bin/npu-smi")
@patch("subprocess.check_output", side_effect=RuntimeError("npu-smi failed"))
def test_get_npu_memory_given_npu_smi_fails_then_returns_none(
    mock_check_output, mock_which
):
    result = get_npu_memory()
    assert result is None


@patch("shutil.which", return_value="/usr/bin/npu-smi")
@patch(
    "subprocess.check_output",
    return_value="HBM Capacity: 65536",
)
def test_get_npu_memory_given_valid_output_then_returns_memory(
    mock_check_output, mock_which
):
    result = get_npu_memory()
    assert result == 65536


@patch("shutil.which", return_value="/usr/bin/npu-smi")
@patch(
    "subprocess.check_output",
    return_value="Some other output without HBM info",
)
def test_get_npu_memory_given_no_hbm_info_then_returns_none(
    mock_check_output, mock_which
):
    result = get_npu_memory()
    assert result is None


@patch("shutil.which", return_value="/usr/bin/npu-smi")
@patch(
    "subprocess.check_output",
    return_value="HBM Capacity : invalid MB",
)
def test_get_npu_memory_given_invalid_hbm_value_then_returns_none(
    mock_check_output, mock_which
):
    result = get_npu_memory()
    assert result is None


# =============================================================================
# Tests for get_conn_mode
# =============================================================================


@patch("shutil.which", return_value=None)
def test_get_conn_mode_given_hccn_tool_not_found_then_returns_unknown(mock_which):
    result = get_conn_mode()
    assert result == ConnMode.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", side_effect=RuntimeError("hccn_tool failed"))
def test_get_conn_mode_given_hccn_tool_fails_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch(
    "subprocess.check_output",
    return_value="System Description TLV\nRouting Switch",
)
def test_get_conn_mode_given_routing_output_then_returns_route(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.ROUTE


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch(
    "subprocess.check_output",
    return_value="System Description TLV\nAscendNPU Switch",
)
def test_get_conn_mode_given_ascend_output_then_returns_fiber(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.FIBER


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch(
    "subprocess.check_output",
    return_value="Some other output",
)
def test_get_conn_mode_given_unknown_output_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch("subprocess.check_output", return_value="")
def test_get_conn_mode_given_empty_output_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.UNKNOWN


@patch("shutil.which", return_value="/usr/bin/hccn_tool")
@patch(
    "subprocess.check_output",
    return_value="System Description TLV",
)
def test_get_conn_mode_given_no_next_line_then_returns_unknown(
    mock_check_output, mock_which
):
    result = get_conn_mode()
    assert result == ConnMode.UNKNOWN


# =============================================================================
# Tests for probe_hardware
# =============================================================================


@patch("msprechecker.util.get_npu_count", return_value=8)
@patch("msprechecker.util.get_npu_type", return_value=NpuType.d100)
@patch("msprechecker.util.get_npu_memory", return_value=65536)
@patch("msprechecker.util.get_conn_mode", return_value=ConnMode.ROUTE)
def test_probe_hardware_given_all_probes_succeed_then_returns_profile(
    mock_conn, mock_memory, mock_type, mock_count
):
    result = probe_hardware()
    assert result.npu_count == 8
    assert result.npu_type == NpuType.d100
    assert result.npu_memory_mb == 65536
    assert result.conn_mode == ConnMode.ROUTE


# =============================================================================
# Tests for FrameworkProbe implementations
# =============================================================================


@patch("os.path.isdir", return_value=True)
def test_mindie_probe_given_directory_exists_then_returns_mindie(mock_isdir):
    probe = MindIEProbe()
    result = probe.probe()
    assert result == Framework.MINDIE


@patch("os.path.isdir", return_value=False)
@patch.dict(os.environ, {"MINDIE_HOME": "/some/path"}, clear=True)
def test_mindie_probe_given_env_var_exists_then_returns_mindie(mock_isdir):
    probe = MindIEProbe()
    result = probe.probe()
    assert result == Framework.MINDIE


@patch("os.path.isdir", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_mindie_probe_given_no_mindie_then_returns_none(mock_isdir):
    probe = MindIEProbe()
    result = probe.probe()
    assert result is None


@patch("os.path.isdir", return_value=True)
def test_vllm_probe_given_workspace_exists_then_returns_vllm(mock_isdir):
    probe = VLLMProbe()
    result = probe.probe()
    assert result == Framework.VLLM


@patch("os.path.isdir", return_value=False)
def test_vllm_probe_given_import_available_then_returns_vllm(mock_isdir):
    probe = VLLMProbe()
    # Mock the import by temporarily adding to sys.modules
    mock_module = Mock()
    with patch.dict("sys.modules", {"vllm_ascend": mock_module}):
        result = probe.probe()
        assert result == Framework.VLLM


@patch("os.path.isdir", return_value=False)
def test_vllm_probe_given_no_vllm_then_returns_none(mock_isdir):
    probe = VLLMProbe()
    # Ensure vllm_ascend is not in sys.modules
    with patch.dict("sys.modules", {}, clear=True):
        result = probe.probe()
        assert result is None


def test_sglang_probe_given_import_available_then_returns_sglang():
    probe = SGLangProbe()
    mock_module = Mock()
    with patch.dict("sys.modules", {"sglang": mock_module}):
        result = probe.probe()
        assert result == Framework.SGLANG


def test_sglang_probe_given_no_sglang_then_returns_none():
    probe = SGLangProbe()
    with patch.dict("sys.modules", {}, clear=True):
        result = probe.probe()
        assert result is None


# =============================================================================
# Tests for detect_framework
# =============================================================================


def test_detect_framework_given_mindie_probe_matches_then_returns_mindie():
    mock_probe = Mock()
    mock_probe.probe.return_value = Framework.MINDIE
    result = detect_framework(probes=(mock_probe,))
    assert result == Framework.MINDIE


def test_detect_framework_given_no_probes_match_then_returns_unknown():
    mock_probe = Mock()
    mock_probe.probe.return_value = None
    result = detect_framework(probes=(mock_probe,))
    assert result == Framework.UNKNOWN


def test_detect_framework_given_empty_probes_then_returns_unknown():
    result = detect_framework(probes=())
    assert result == Framework.UNKNOWN


# =============================================================================
# Tests for is_valid_ip_strict
# =============================================================================


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("255.255.255.255", True),
        ("0.0.0.0", True),
        ("256.1.1.1", False),
        ("192.168.1", False),
        ("192.168.1.1.1", False),
        ("not_an_ip", False),
        ("", False),
        ("192.168.1.a", False),
        (" 192.168.1.1", False),  # Leading space should fail
        ("192.168.1.1 ", False),  # Trailing space should fail
    ],
)
def test_is_valid_ip_strict_given_various_inputs_then_returns_expected(ip, expected):
    result = is_valid_ip_strict(ip)
    assert result == expected


# =============================================================================
# Tests for parse_rank_table
# =============================================================================


@patch("builtins.open", new_callable=mock_open, read_data='{"server_list": []}')
@patch("pathlib.Path.resolve")
def test_parse_rank_table_given_mindie_framework_then_calls_mindie_parser(
    mock_resolve, mock_file
):
    mock_resolve.return_value = Path("/fake/path")
    with patch.object(Path, "is_file", return_value=True):
        # Test the framework dispatch - should not raise for empty server_list
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open, read_data='{"prefill_device_list": []}')
@patch("pathlib.Path.resolve")
def test_parse_rank_table_given_vllm_framework_then_calls_vllm_parser(
    mock_resolve, mock_file
):
    mock_resolve.return_value = Path("/fake/path")
    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


def test_parse_rank_table_given_unknown_framework_then_raises_value_error():
    with pytest.raises(ValueError, match="No rank table parser"):
        parse_rank_table(Path("/fake/path"), Framework.UNKNOWN)


def test_parse_rank_table_given_sglang_framework_then_raises_value_error():
    with pytest.raises(ValueError, match="No rank table parser"):
        parse_rank_table(Path("/fake/path"), Framework.SGLANG)


# =============================================================================
# Tests for resolve_weight_dir
# =============================================================================


def test_resolve_weight_dir_given_unknown_framework_then_raises_value_error():
    with pytest.raises(ValueError, match="Weight dir resolution not supported"):
        resolve_weight_dir(Framework.UNKNOWN)


def test_resolve_weight_dir_given_vllm_without_config_path_then_raises_value_error():
    with pytest.raises(ValueError, match="config_path .* is required"):
        resolve_weight_dir(Framework.VLLM, None)


def test_resolve_weight_dir_given_sglang_without_config_path_then_raises_value_error():
    with pytest.raises(ValueError, match="config_path .* is required"):
        resolve_weight_dir(Framework.SGLANG, None)


@patch("pathlib.Path.is_file", return_value=False)
def test_resolve_weight_dir_given_vllm_with_nonexistent_script_then_raises_error(
    mock_isfile,
):
    with pytest.raises(WeightDirNotFoundError):
        resolve_weight_dir(Framework.VLLM, Path("/nonexistent/script.sh"))


@patch("pathlib.Path.is_file", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='vllm serve "model/path"',
)
def test_resolve_weight_dir_given_vllm_with_valid_script_then_returns_path(
    mock_file, mock_isfile
):
    result = resolve_weight_dir(Framework.VLLM, Path("/path/to/script.sh"))
    assert result == "model/path"


@patch("pathlib.Path.is_file", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="python -m sglang --model=model/path",
)
def test_resolve_weight_dir_given_sglang_with_valid_script_then_returns_path(
    mock_file, mock_isfile
):
    result = resolve_weight_dir(Framework.SGLANG, Path("/path/to/script.sh"))
    assert result == "model/path"


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="no model pattern here")
def test_resolve_weight_dir_given_script_without_pattern_then_raises_error(
    mock_file, mock_isfile
):
    with pytest.raises(WeightDirNotFoundError, match="No model path pattern found"):
        resolve_weight_dir(Framework.VLLM, Path("/path/to/script.sh"))


# =============================================================================
# Tests for NpuType enum
# =============================================================================


def test_npu_type_enum_values():
    assert NpuType.d100.value == "300"
    assert NpuType.d500.value == "300I_DUO"
    assert NpuType.d801.value == "800I A1"
    assert NpuType.d802.value == "800I A2"
    assert NpuType.d803.value == "800I A3"
    assert NpuType.UNKNOWN.value == "unknown"


# =============================================================================
# Tests for ConnMode enum
# =============================================================================


def test_conn_mode_enum_values():
    assert ConnMode.ROUTE.value == "route"
    assert ConnMode.FIBER.value == "fiber"
    assert ConnMode.UNKNOWN.value == "unknown"


# =============================================================================
# Tests for Framework enum
# =============================================================================


def test_framework_enum_values():
    assert Framework.MINDIE.value == "mindie"
    assert Framework.VLLM.value == "vllm"
    assert Framework.SGLANG.value == "sglang"
    assert Framework.UNKNOWN.value == "unknown"


# =============================================================================
# Tests for RankTable dataclass
# =============================================================================


def test_rank_table_dataclass():
    from msprechecker.util import DeviceInfo

    devices = [DeviceInfo(device_ip="192.168.1.1", device_id=0, rank_id=0)]
    rank_table = RankTable(
        host_to_devices={"host1": devices},
        server_count=1,
        version="1.0",
    )
    assert rank_table.server_count == 1
    assert rank_table.version == "1.0"
    assert "host1" in rank_table.host_to_devices


# =============================================================================
# Tests for DeviceInfo dataclass
# =============================================================================


def test_device_info_dataclass():
    from msprechecker.util import DeviceInfo

    device = DeviceInfo(device_ip="192.168.1.1", device_id=0, rank_id=0)
    assert device.device_ip == "192.168.1.1"
    assert device.device_id == 0
    assert device.rank_id == 0


# =============================================================================
# Tests for HardwareProfile dataclass
# =============================================================================


def test_hardware_profile_dataclass():
    from msprechecker.util import HardwareProfile

    profile = HardwareProfile(
        npu_count=8,
        npu_type=NpuType.d100,
        npu_memory_mb=65536,
        conn_mode=ConnMode.ROUTE,
    )
    assert profile.npu_count == 8
    assert profile.npu_type == NpuType.d100
    assert profile.npu_memory_mb == 65536
    assert profile.conn_mode == ConnMode.ROUTE


def test_hardware_profile_with_none_values():
    from msprechecker.util import HardwareProfile

    profile = HardwareProfile(
        npu_count=0,
        npu_type=NpuType.UNKNOWN,
        npu_memory_mb=None,
        conn_mode=None,
    )
    assert profile.npu_count == 0
    assert profile.npu_memory_mb is None
    assert profile.conn_mode is None


# =============================================================================
# Tests for RankTableParseError
# =============================================================================


def test_rank_table_parse_error_is_value_error():
    with pytest.raises(ValueError):
        raise RankTableParseError("test error")


def test_rank_table_parse_error_message():
    error = RankTableParseError("custom message")
    assert str(error) == "custom message"


# =============================================================================
# Tests for WeightDirNotFoundError
# =============================================================================


def test_weight_dir_not_found_error_is_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        raise WeightDirNotFoundError("test error")


def test_weight_dir_not_found_error_message():
    error = WeightDirNotFoundError("custom message")
    assert str(error) == "custom message"


# =============================================================================
# Additional tests for MindIEProbe edge cases
# =============================================================================


@patch("os.path.isdir", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_mindie_probe_given_no_indicators_then_returns_none(mock_isdir):
    probe = MindIEProbe()
    result = probe.probe()
    assert result is None


# =============================================================================
# Additional tests for parse_rank_table with valid data
# =============================================================================


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_valid_data(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)
        assert result.server_count == 1


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_valid_data(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)
        assert result.server_count == 1


# =============================================================================
# Additional tests for _weight_dir_from_mindie_config
# =============================================================================


@patch("pathlib.Path.is_file", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "BackendConfig": {
                "ModelDeployConfig": {
                    "ModelConfig": [{"modelWeightPath": "/model/path"}]
                }
            }
        }
    ),
)
def test_weight_dir_from_mindie_config_given_valid_config_then_returns_path(
    mock_file, mock_isfile
):
    from msprechecker.util import _weight_dir_from_mindie_config

    result = _weight_dir_from_mindie_config(Path("/fake/config.json"))
    assert result == "/model/path"


@patch("pathlib.Path.is_file", return_value=False)
def test_weight_dir_from_mindie_config_given_nonexistent_config_then_raises_error(
    mock_isfile,
):
    from msprechecker.util import _weight_dir_from_mindie_config

    with pytest.raises(WeightDirNotFoundError):
        _weight_dir_from_mindie_config(Path("/nonexistent/config.json"))


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="invalid json")
def test_weight_dir_from_mindie_config_given_invalid_json_then_raises_error(
    mock_file, mock_isfile
):
    from msprechecker.util import _weight_dir_from_mindie_config

    with pytest.raises(RankTableParseError):
        _weight_dir_from_mindie_config(Path("/fake/config.json"))


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({}))
def test_weight_dir_from_mindie_config_given_missing_key_then_raises_error(
    mock_file, mock_isfile
):
    from msprechecker.util import _weight_dir_from_mindie_config

    with pytest.raises(RankTableParseError):
        _weight_dir_from_mindie_config(Path("/fake/config.json"))


# =============================================================================
# Additional tests for _weight_dir_from_script
# =============================================================================


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='vllm serve "/path/to model"')
def test_weight_dir_from_script_given_quoted_path_then_returns_path(
    mock_file, mock_isfile
):
    from msprechecker.util import _VLLM_MODEL_RE, _weight_dir_from_script

    result = _weight_dir_from_script(Path("/fake/script.sh"), _VLLM_MODEL_RE)
    assert result == "/path/to model"


@patch("pathlib.Path.is_file", return_value=True)
@patch("builtins.open", side_effect=OSError("read error"))
def test_weight_dir_from_script_given_read_error_then_raises_error(
    mock_file, mock_isfile
):
    from msprechecker.util import _VLLM_MODEL_RE, _weight_dir_from_script

    with pytest.raises(WeightDirNotFoundError, match="Failed to read script"):
        _weight_dir_from_script(Path("/fake/script.sh"), _VLLM_MODEL_RE)


# =============================================================================
# Additional tests for resolve_weight_dir with MindIE
# =============================================================================


@patch.dict(
    os.environ,
    {"MIES_INSTALL_PATH": "/usr/local/Ascend/mindie/latest/mindie-service"},
    clear=True,
)
@patch("pathlib.Path.is_file", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "BackendConfig": {
                "ModelDeployConfig": {
                    "ModelConfig": [{"modelWeightPath": "/model/path"}]
                }
            }
        }
    ),
)
def test_resolve_weight_dir_mindie_with_env_path_then_returns_path(
    mock_file, mock_isfile
):
    result = resolve_weight_dir(Framework.MINDIE)
    assert result == "/model/path"


@patch.dict(os.environ, {}, clear=True)
@patch("pathlib.Path.is_file", return_value=False)
def test_resolve_weight_dir_mindie_with_default_path_not_found_then_raises_error(
    mock_isfile,
):
    with pytest.raises(WeightDirNotFoundError):
        resolve_weight_dir(Framework.MINDIE)


# =============================================================================
# Additional tests for _parse_server_count
# =============================================================================


def test_parse_server_count_given_int_then_returns_int():
    from msprechecker.util import _parse_server_count

    result = _parse_server_count(5)
    assert result == 5


def test_parse_server_count_given_digit_string_then_returns_int():
    from msprechecker.util import _parse_server_count

    result = _parse_server_count("10")
    assert result == 10


def test_parse_server_count_given_invalid_then_returns_zero():
    from msprechecker.util import _parse_server_count

    result = _parse_server_count("invalid")
    assert result == 0


# =============================================================================
# Additional tests for parse_rank_table with edge cases
# =============================================================================


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_server_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "invalid_ip",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_device_ip(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "invalid_ip", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_device_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [
                    {"device_ip": "192.168.2.1", "device_id": "invalid", "rank_id": 0}
                ],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_rank_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [
                    {"device_ip": "192.168.2.1", "device_id": 0, "rank_id": "invalid"}
                ],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_invalid_server_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "invalid_ip",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_invalid_cluster_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": "invalid",
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_exceeds_host_limit(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more hosts than allowed
    data = {
        "server_list": [
            {"server_id": f"192.168.{i}.1", "device": []} for i in range(1001)
        ],
        "server_count": 1001,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(
        RankTableParseError, match="Host count exceeds limit"
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_exceeds_device_limit(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more devices than allowed per host
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [
                    {"device_ip": f"192.168.2.{i}", "device_id": i, "rank_id": i}
                    for i in range(33)
                ],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(
        RankTableParseError, match="Device count for host .* exceeds limit"
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_exceeds_total_limit(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more devices than total allowed
    data = {
        "prefill_device_list": [
            {
                "server_id": f"192.168.{i}.1",
                "device_ip": f"192.168.{i}.2",
                "device_id": i,
                "cluster_id": i + 1,
            }
            for i in range(32001)  # Exceeds _HOST_LIMIT * _DEVICE_LIMIT_PER_HOST
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(
        RankTableParseError, match="length exceeds limit"
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_exceeds_host_limit(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more hosts than allowed (using valid IPs)
    devices = []
    for i in range(1001):
        # Use valid IPs in the 10.0.x.x range
        octet2 = i // 256
        octet3 = i % 256
        devices.append(
            {
                "server_id": f"10.0.{octet2}.{octet3}",
                "device_ip": f"10.1.{octet2}.{octet3}",
                "device_id": 0,
                "cluster_id": i + 1,
            }
        )

    data = {
        "prefill_device_list": devices,
        "decode_device_list": [],
        "server_count": 1001,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(
        RankTableParseError, match="Host count exceeds limit"
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_missing_device_list(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


@patch("builtins.open", side_effect=OSError("Failed to open"))
@patch("pathlib.Path.resolve")
def test_parse_rank_table_json_load_error(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    with patch.object(Path, "is_file", return_value=True), pytest.raises(
        RankTableParseError
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


# =============================================================================
# Additional tests for _default_mindie_config_path
# =============================================================================


@patch.dict(os.environ, {"MIES_INSTALL_PATH": "/custom/path"}, clear=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch("pathlib.Path.resolve", return_value=Path("/custom/path/conf/config.json"))
def test_default_mindie_config_path_with_env(mock_resolve, mock_isfile):
    from msprechecker.util import _default_mindie_config_path

    result = _default_mindie_config_path()
    assert result == Path("/custom/path/conf/config.json")


@patch.dict(os.environ, {}, clear=True)
@patch("pathlib.Path.is_file", return_value=False)
def test_default_mindie_config_path_with_default_not_found(mock_isfile):
    from msprechecker.util import _default_mindie_config_path

    result = _default_mindie_config_path()
    assert result is None


# =============================================================================
# Additional tests for get_npu_type with different device patterns
# =============================================================================


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device d801",
)
def test_get_npu_type_given_d801_device_then_returns_d801(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.d801


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device d802",
)
def test_get_npu_type_given_d802_device_then_returns_d802(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.d802


@patch("shutil.which", return_value="/usr/bin/lspci")
@patch(
    "subprocess.check_output",
    return_value="00:00.0 Accelerator: Huawei Technologies Co., Ltd. Device d803",
)
def test_get_npu_type_given_d803_device_then_returns_d803(
    mock_check_output, mock_which
):
    result = get_npu_type()
    assert result == NpuType.d803
