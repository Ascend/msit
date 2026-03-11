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

from unittest.mock import patch

from msprechecker.core.checker import Failed, Passed, Severity, Skipped
from msprechecker.core.suite import (
    _cpu_high_performance,
    _cpu_stress,
    _driver_version,
    _find_abnormal,
    _hccl_ping,
    _jemalloc,
    _kernel_version,
    _link_status,
    _npu_connection_mode,
    _npu_stress,
    _ping,
    _tls_status,
    _transparent_hugepage,
    _virtual_machine,
    build_suite,
    GROUPS,
)
from msprechecker.util import ConnMode, Framework, RankTable


# =============================================================================
# Tests for GROUPS
# =============================================================================


def test_groups_system():
    assert GROUPS.SYSTEM.key == "system"
    assert GROUPS.SYSTEM.title == "Checking system environment"


def test_groups_ascend():
    assert GROUPS.ASCEND.key == "ascend"
    assert GROUPS.ASCEND.title == "Checking Ascend environment"


def test_groups_network():
    assert GROUPS.NETWORK.key == "network"
    assert GROUPS.NETWORK.title == "Checking network connectivity"


def test_groups_stress():
    assert GROUPS.STRESS.key == "stress"
    assert GROUPS.STRESS.title == "Running hardware stress tests"


# =============================================================================
# Tests for system checks
# =============================================================================


def test_cpu_high_performance_when_enabled():
    with patch("msprechecker.core.suite.CPUHighPerformance.execute", return_value=True):
        result = _cpu_high_performance()
        assert isinstance(result, Passed)
        assert result.result_text == "on"


def test_cpu_high_performance_when_disabled():
    with patch(
        "msprechecker.core.suite.CPUHighPerformance.execute", return_value=False
    ):
        result = _cpu_high_performance()
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO
        assert result.result_text == "off"


def test_virtual_machine_when_physical():
    with patch("msprechecker.core.suite.VirtualMachine.execute", return_value=False):
        result = _virtual_machine()
        assert isinstance(result, Passed)
        assert result.result_text == "physical"


def test_virtual_machine_when_virtual():
    with patch("msprechecker.core.suite.VirtualMachine.execute", return_value=True):
        result = _virtual_machine()
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO
        assert result.result_text == "virtual"


def test_transparent_hugepage_when_enabled():
    with patch(
        "msprechecker.core.suite.TransparentHugepage.execute",
        return_value="[always] madvise never",
    ):
        result = _transparent_hugepage()
        assert isinstance(result, Passed)
        assert result.result_text == "enabled"


def test_transparent_hugepage_when_disabled():
    with patch(
        "msprechecker.core.suite.TransparentHugepage.execute",
        return_value="always [madvise] never",
    ):
        result = _transparent_hugepage()
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO
        assert result.result_text == "disabled"


def test_transparent_hugepage_when_none():
    with patch(
        "msprechecker.core.suite.TransparentHugepage.execute", return_value=None
    ):
        result = _transparent_hugepage()
        assert isinstance(result, Failed)
        assert result.result_text == "disabled"


def test_kernel_version_when_valid():
    with patch(
        "msprechecker.core.suite.Kernel.execute", return_value={"release": "5.15.0"}
    ):
        result = _kernel_version()
        assert isinstance(result, Passed)
        assert "5.15.0" in result.result_text


def test_kernel_version_when_no_release():
    with patch("msprechecker.core.suite.Kernel.execute", return_value={}):
        result = _kernel_version()
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_kernel_version_when_old_kernel():
    with patch(
        "msprechecker.core.suite.Kernel.execute", return_value={"release": "5.4.0"}
    ):
        result = _kernel_version()
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO
        assert "5.4.0" in result.result_text


def test_kernel_version_when_invalid_version():
    with patch(
        "msprechecker.core.suite.Kernel.execute", return_value={"release": "invalid"}
    ):
        result = _kernel_version()
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_npu_connection_mode_when_route():
    with patch("msprechecker.core.suite.get_conn_mode", return_value=ConnMode.ROUTE):
        result = _npu_connection_mode()
        assert isinstance(result, Passed)
        assert result.result_text == "route"


def test_npu_connection_mode_when_fiber():
    with patch("msprechecker.core.suite.get_conn_mode", return_value=ConnMode.FIBER):
        result = _npu_connection_mode()
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING
        assert result.result_text == "fiber direct"


def test_npu_connection_mode_when_unknown():
    with patch("msprechecker.core.suite.get_conn_mode", return_value=ConnMode.UNKNOWN):
        result = _npu_connection_mode()
        assert isinstance(result, Passed)
        assert result.result_text == "unknown"


def test_jemalloc_for_non_vllm():
    result = _jemalloc(framework=Framework.MINDIE)
    assert isinstance(result, Skipped)
    assert "vllm" in result.reason.lower()


def test_jemalloc_when_installed():
    with patch("msprechecker.core.suite.JeMalloc.execute", return_value=True):
        result = _jemalloc(framework=Framework.VLLM)
        assert isinstance(result, Passed)
        assert result.result_text == "installed"


def test_jemalloc_when_not_installed():
    with patch("msprechecker.core.suite.JeMalloc.execute", return_value=False):
        result = _jemalloc(framework=Framework.VLLM)
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO
        assert result.result_text == "not installed"


# =============================================================================
# Tests for Ascend checks
# =============================================================================


def test_driver_version_when_not_found():
    with patch("msprechecker.core.suite.Driver.execute", return_value=None):
        result = _driver_version(framework=Framework.MINDIE, scene="test")
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_driver_version_when_valid():
    with patch(
        "msprechecker.core.suite.Driver.execute", return_value={"Version": "24.1.0"}
    ):
        result = _driver_version(framework=Framework.MINDIE, scene="test")
        assert isinstance(result, Passed)
        assert result.result_text == "24.1.0"


def test_driver_version_when_invalid_version_format():
    with patch(
        "msprechecker.core.suite.Driver.execute", return_value={"Version": "invalid"}
    ):
        result = _driver_version(framework=Framework.MINDIE, scene="test")
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_driver_version_when_too_old():
    with patch(
        "msprechecker.core.suite.Driver.execute", return_value={"Version": "23.0.0"}
    ):
        result = _driver_version(framework=Framework.MINDIE, scene="test")
        assert isinstance(result, Failed)
        assert result.severity == Severity.INFO


def test_driver_version_for_vllm_ep():
    with patch(
        "msprechecker.core.suite.Driver.execute", return_value={"Version": "24.0.0"}
    ):
        result = _driver_version(framework=Framework.VLLM, scene="pd_disaggregation_ep")
        assert isinstance(result, Failed)
        assert result.severity == Severity.ERROR


# =============================================================================
# Tests for network checks
# =============================================================================


def test_ping_when_all_success():
    def collect():
        return {"ping_192.168.1.1": "3 packets transmitted, 3 received, 0% packet loss"}

    result = _ping(collect)
    assert isinstance(result, Passed)


def test_ping_when_no_results():
    collect = dict
    result = _ping(collect)
    assert isinstance(result, Skipped)


def test_ping_when_some_fail():
    def collect():
        return {
            "ping_192.168.1.1": "3 packets transmitted, 3 received, 0% packet loss",
            "ping_192.168.1.2": None,
        }

    result = _ping(collect)
    assert isinstance(result, Failed)
    assert result.severity == Severity.ERROR


def test_ping_with_packet_loss():
    def collect():
        return {
            "ping_192.168.1.1": "3 packets transmitted, 1 received, 66% packet loss"
        }

    result = _ping(collect)
    assert isinstance(result, Failed)


def test_link_status_when_up():
    def collect():
        return {"link": ["up", "up", "up"]}

    result = _link_status(collect)
    assert isinstance(result, Passed)
    assert result.result_text == "up"


def test_link_status_when_not_present():
    collect = dict
    result = _link_status(collect)
    assert isinstance(result, Skipped)


def test_link_status_when_some_down():
    def collect():
        return {"link": ["up", None, "up"]}

    result = _link_status(collect)
    assert isinstance(result, Failed)
    assert result.severity == Severity.ERROR


def test_tls_status_when_up():
    def collect():
        return {"tls": ["up", "up"]}

    result = _tls_status(collect)
    assert isinstance(result, Passed)
    assert result.result_text == "up"


def test_tls_status_when_not_present():
    collect = dict
    result = _tls_status(collect)
    assert isinstance(result, Skipped)


def test_tls_status_when_down():
    def collect():
        return {"tls": [None, None]}

    result = _tls_status(collect)
    assert isinstance(result, Failed)
    assert result.severity == Severity.ERROR


def test_hccl_ping_when_success():
    def collect():
        return {"hccl_ping": [{"192.168.1.1": "ok"}, {"192.168.1.1": "ok"}]}

    result = _hccl_ping(collect)
    assert isinstance(result, Passed)


def test_hccl_ping_with_hccs():
    def collect():
        return {"hccs_ping": [{"192.168.1.1": "ok"}, {"192.168.1.1": "ok"}]}

    result = _hccl_ping(collect)
    assert isinstance(result, Passed)


def test_hccl_ping_when_not_present():
    collect = dict
    result = _hccl_ping(collect)
    assert isinstance(result, Skipped)


def test_hccl_ping_when_some_fail():
    def collect():
        return {"hccl_ping": [{"192.168.1.1": None}, {"192.168.1.1": "ok"}]}

    result = _hccl_ping(collect)
    assert isinstance(result, Failed)
    assert result.severity == Severity.ERROR


# =============================================================================
# Tests for stress checks helpers
# =============================================================================


def test_find_abnormal_with_empty_results():
    result = _find_abnormal({}, 20)
    assert result == []


def test_find_abnormal_with_no_valid_results():
    result = _find_abnormal({0: None, 1: 0}, 20)
    assert result == []


def test_find_abnormal_with_all_similar():
    result = _find_abnormal({0: 100.0, 1: 102.0, 2: 101.0}, 20)
    assert result == []


def test_find_abnormal_with_one_outlier():
    result = _find_abnormal({0: 100.0, 1: 100.0, 2: 150.0}, 20)
    assert 2 in result


def test_find_abnormal_with_zero_mean():
    result = _find_abnormal({0: 0.0, 1: 0.0}, 20)
    assert result == []


# =============================================================================
# Tests for stress checks
# =============================================================================


def test_cpu_stress_when_torch_not_available():
    with patch("msprechecker.core.suite.CPU.execute", return_value=None):
        result = _cpu_stress(threshold=20)
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_cpu_stress_when_no_abnormal():
    with patch(
        "msprechecker.core.suite.CPU.execute", return_value={0: 100.0, 1: 102.0}
    ):
        result = _cpu_stress(threshold=20)
        assert isinstance(result, Passed)


def test_cpu_stress_with_abnormal_cores():
    # Mean = 125, threshold 20% -> limit = 150, so 150 is not > 150
    # Use 160 to ensure it's above the threshold
    with patch(
        "msprechecker.core.suite.CPU.execute", return_value={0: 100.0, 1: 160.0}
    ):
        result = _cpu_stress(threshold=20)
        assert isinstance(result, Failed)
        assert result.severity == Severity.ERROR


def test_npu_stress_when_torch_not_available():
    with patch("msprechecker.core.suite.NPU.execute", return_value=None):
        result = _npu_stress(threshold=20)
        assert isinstance(result, Failed)
        assert result.severity == Severity.WARNING


def test_npu_stress_when_no_abnormal():
    with patch(
        "msprechecker.core.suite.NPU.execute", return_value={0: 100.0, 1: 102.0}
    ):
        result = _npu_stress(threshold=20)
        assert isinstance(result, Passed)


def test_npu_stress_with_abnormal_devices():
    # Mean = 125, threshold 20% -> limit = 150, so 150 is not > 150
    # Use 160 to ensure it's above the threshold
    with patch(
        "msprechecker.core.suite.NPU.execute", return_value={0: 100.0, 1: 160.0}
    ):
        result = _npu_stress(threshold=20)
        assert isinstance(result, Failed)
        assert result.severity == Severity.ERROR


# =============================================================================
# Tests for build_suite
# =============================================================================


def test_build_suite_basic():
    suite = build_suite(framework=Framework.MINDIE, scene="test")
    assert len(suite) >= 6  # system checks + ascend checks


def test_build_suite_with_network():
    with patch("msprechecker.core.suite.parse_rank_table") as mock_parse:
        mock_parse.return_value = RankTable(
            host_to_devices={"host1": []},
            server_count=1,
            version="1.0",
        )
        suite = build_suite(
            framework=Framework.MINDIE,
            scene="test",
            rank_table_path="/fake/path.json",
        )
        # Should include network checks
        network_checks = [c for c in suite if c.group == GROUPS.NETWORK]
        assert len(network_checks) == 4


def test_build_suite_with_hardware():
    suite = build_suite(
        framework=Framework.MINDIE,
        scene="test",
        hardware=True,
    )
    stress_checks = [c for c in suite if c.group == GROUPS.STRESS]
    assert len(stress_checks) == 2


def test_build_suite_with_custom_threshold():
    suite = build_suite(
        framework=Framework.MINDIE,
        scene="test",
        hardware=True,
        threshold=30,
    )
    # Just verify it doesn't raise an error
    assert len(suite) > 0


def test_build_suite_returns_check_objects():
    suite = build_suite(framework=Framework.MINDIE, scene="test")
    from msprechecker.core.checker import Check

    assert all(isinstance(c, Check) for c in suite)
