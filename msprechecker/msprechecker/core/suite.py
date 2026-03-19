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

"""
Suite assembly.

Design rules
~~~~~~~~~~~~
1. Every check function is a plain module-level function returning a
   ``CheckOutcome``.  No class inheritance, no magic attributes.

2. Runtime parameters (framework, scene, threshold …) are captured via
   ``functools.partial`` – never bare lambdas, so ``repr(check.fn)``
   is always meaningful in tracebacks and logs.

3. Network data collection is **deferred**: the shared collector runs
   inside each network check function (memoised to run once), so any
   collection failure is caught by the runner and recorded as a ``Failed``
   entry rather than crashing ``build_suite``.

4. ``build_suite`` returns a plain ``list[Check]``.  The runner needs no
   knowledge of how the list was assembled.
"""

from __future__ import annotations

import functools
import traceback as tb
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from packaging.version import InvalidVersion, Version

from ..utils import (
    ConnMode,
    detect_framework,
    Framework,
    get_conn_mode,
    get_npu_count,
    parse_rank_table,
)
from .checker import Check, CheckGroup, CheckOutcome, Failed, Passed, Severity, Skipped
from ..strategies import (
    CPU,
    CPUHighPerformance,
    Driver,
    JeMalloc,
    Kernel,
    Network,
    NPU,
    TransparentHugepage,
    VirtualMachine,
)

# ── section groups ────────────────────────────────────────────────────────────

GROUPS = SimpleNamespace(
    SYSTEM=CheckGroup("system", "Checking system environment"),
    ASCEND=CheckGroup("ascend", "Checking Ascend environment"),
    NETWORK=CheckGroup("network", "Checking network connectivity"),
    STRESS=CheckGroup("stress", "Running hardware stress tests"),
)


# ── system checks ─────────────────────────────────────────────────────────────


def _cpu_high_performance() -> CheckOutcome:
    if not CPUHighPerformance().execute():
        return Failed(
            msg="建议开启 CPU 高性能模式，可使 CPU 运行在最大频率，提升性能，但会增加能耗",
            severity=Severity.INFO,
            result_text="off",
        )
    return Passed("on")


def _virtual_machine() -> CheckOutcome:
    if VirtualMachine().execute():
        return Failed(
            msg="检测到虚拟机环境，其 CPU 核数和频率与物理机有差异，可能导致性能下降",
            severity=Severity.INFO,
            result_text="virtual",
        )
    return Passed("physical")


def _transparent_hugepage() -> CheckOutcome:
    result = TransparentHugepage().execute()
    if not (result and "[always]" in result):
        return Failed(
            msg="建议开启透明大页（Transparent Hugepage），可使吞吐率结果更加稳定",
            severity=Severity.INFO,
            result_text="disabled",
        )
    return Passed("enabled")


def _kernel_version() -> CheckOutcome:
    info = Kernel().execute() or {}
    release = info.get("release")

    if not release:
        return Failed(
            msg="无法检测到内核版本号",
            severity=Severity.WARNING,
            result_text="unknown",
        )

    try:
        version = Version(release)
    except InvalidVersion:
        return Failed(
            msg=f"无法解析内核版本号: {release}",
            severity=Severity.WARNING,
            result_text="unknown",
            traceback=tb.format_exc(),
        )

    min_version = Version("5.10")
    if version < min_version:
        return Failed(
            msg=(
                f"当前内核版本 {release} 低于推荐版本 5.10，"
                "升级后 CPU 指令下发速度更快，可减少 host bound"
            ),
            severity=Severity.INFO,
            result_text=f"{release} (requires >= {min_version})",
        )
    return Passed(release)


def _npu_connection_mode() -> CheckOutcome:
    conn_mode = get_conn_mode()
    if conn_mode == ConnMode.FIBER:
        return Failed(
            msg=(
                '检测到网线对端设备为昇腾 NPU，疑似"双机背靠背直连"架构。'
                "该架构下 HCCL 不支持全互联通信链路自动建立，模型通信可能受到影响，"
                "请确认当前部署环境"
            ),
            severity=Severity.WARNING,
            result_text="fiber direct",
        )
    return Passed(conn_mode.value)


def _jemalloc(*, framework: Framework) -> CheckOutcome:
    if framework != Framework.VLLM:
        return Skipped("jemalloc check only applies to VLLM")

    if not JeMalloc().execute():
        return Failed(
            msg=(
                "未检测到通过 apt/yum 安装的 jemalloc，建议安装以提升性能；"
                "若已安装至自定义路径，请忽略此项"
            ),
            severity=Severity.INFO,
            result_text="not installed",
        )
    return Passed("installed")


def _system_checks(framework: Framework) -> list[Check]:
    return [
        Check("CPU high performance mode", GROUPS.SYSTEM, _cpu_high_performance),
        Check("Virtual machine detection", GROUPS.SYSTEM, _virtual_machine),
        Check("Transparent hugepage", GROUPS.SYSTEM, _transparent_hugepage),
        Check("Kernel version", GROUPS.SYSTEM, _kernel_version),
        Check("NPU connection mode", GROUPS.SYSTEM, _npu_connection_mode),
        Check(
            "jemalloc",
            GROUPS.SYSTEM,
            functools.partial(_jemalloc, framework=framework),
        ),
    ]


# ── Ascend checks ─────────────────────────────────────────────────────────────


def _driver_version(*, framework: Framework, scene: str) -> CheckOutcome:
    min_default = Version("24.1")
    min_vllm_ep = Version("25.0")

    is_vllm_ep = framework == Framework.VLLM and "ep" in scene.lower()
    min_ver = min_vllm_ep if is_vllm_ep else min_default
    severity = Severity.ERROR if is_vllm_ep else Severity.INFO

    driver_info = Driver().execute()
    if driver_info is None:
        return Failed(
            msg="无法检测到 Ascend 驱动，请确认驱动已正确安装",
            severity=Severity.WARNING,
            result_text="not found",
        )

    version_str = driver_info.get("Version", "")
    try:
        version = Version(version_str)
    except InvalidVersion:
        return Failed(
            msg=f"驱动版本号格式无法解析: {version_str!r}，请检查驱动安装状态",
            severity=Severity.WARNING,
            result_text="unknown",
        )

    if version < min_ver:
        min_str = f"{min_ver.major}.{min_ver.minor}"
        msg = (
            f"当前驱动版本 {version_str} 低于 VLLM 大 EP 场景所需最低版本 {min_str}，"
            "不升级可能导致 dispatch_combine 等算子失败"
            if is_vllm_ep
            else f"当前驱动版本 {version_str} 低于推荐版本 {min_str}，"
                 "建议升级至最新驱动以获得更好的性能"
        )
        return Failed(
            msg=msg,
            severity=severity,
            result_text=f"{version_str} (requires >= {min_str})",
        )
    return Passed(version_str)


def _ascend_checks(framework: Framework, scene: str) -> list[Check]:
    return [
        Check(
            "Ascend driver version",
            GROUPS.ASCEND,
            functools.partial(_driver_version, framework=framework, scene=scene),
        ),
    ]


# ── network checks ────────────────────────────────────────────────────────────


def _make_network_collector(rank_table_path: str) -> Callable[[], dict]:
    """
    Return a memoised callable that runs network probes exactly once.

    Deferring collection to call time means any failure is caught by
    the runner and surfaces as a ``Failed`` record in the report.
    """
    _cache: dict = {}

    def collect() -> dict:
        if not _cache:
            framework = detect_framework()
            rank_table = parse_rank_table(Path(rank_table_path), framework)
            npu_count = get_npu_count()
            _cache["results"] = Network(
                rank_table=rank_table, npu_count=npu_count
            ).execute()
        return _cache["results"]

    return collect


def _ping(collect) -> CheckOutcome:
    results = collect()
    ping_results = {k: v for k, v in results.items() if k.startswith("ping_")}
    if not ping_results:
        return Skipped("no ping results in rank table output")
    failed_ips = [
        ip
        for ip, v in ping_results.items()
        if v is None or "0% packet loss" not in str(v)
    ]
    if failed_ips:
        return Failed(
            msg=f"Ping 测试失败的目标: {', '.join(failed_ips)}",
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


def _link_status(collect) -> CheckOutcome:
    results = collect()
    link = results.get("link")
    if link is None:
        return Skipped("link result not present")
    if not all(r is not None for r in link):
        return Failed(
            msg="链路状态检查失败", severity=Severity.ERROR, result_text="down"
        )
    return Passed("up")


def _tls_status(collect) -> CheckOutcome:
    results = collect()
    tls = results.get("tls")
    if tls is None:
        return Skipped("tls result not present")
    if not all(r is not None for r in tls):
        return Failed(
            msg="TLS 状态检查失败", severity=Severity.ERROR, result_text="down"
        )
    return Passed("up")


def _hccl_ping(collect) -> CheckOutcome:
    results = collect()
    hccl = results.get("hccl_ping") or results.get("hccs_ping")
    if hccl is None:
        return Skipped("hccl/hccs ping result not present")
    ok = all(all(v is not None for v in device.values()) for device in hccl)
    if not ok:
        return Failed(
            msg="HCCL ping 检查失败",
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


def _network_checks(rank_table_path: str) -> list[Check]:
    collect = _make_network_collector(rank_table_path)
    return [
        Check("Network ping test", GROUPS.NETWORK, functools.partial(_ping, collect)),
        Check(
            "Network link status",
            GROUPS.NETWORK,
            functools.partial(_link_status, collect),
        ),
        Check(
            "TLS link status", GROUPS.NETWORK, functools.partial(_tls_status, collect)
        ),
        Check("HCCL ping test", GROUPS.NETWORK, functools.partial(_hccl_ping, collect)),
    ]


# ── stress checks ─────────────────────────────────────────────────────────────


def _find_abnormal(
        results: dict[int, Optional[float]], threshold_pct: int
) -> list[int]:
    valid = {k: v for k, v in results.items() if v is not None and v > 0}
    if not valid:
        return []
    mean = sum(valid.values()) / len(valid)
    if mean == 0:
        return []
    ratio = threshold_pct / 100.0
    return [uid for uid, t in valid.items() if t > mean * (1 + ratio)]


def _cpu_stress(*, threshold: int) -> CheckOutcome:
    raw = CPU().execute()
    if raw is None:
        return Failed(
            msg="CPU 压测无法完成 (torch not available)",
            severity=Severity.WARNING,
            result_text="skipped",
        )
    abnormal = _find_abnormal(raw, threshold)
    if abnormal:
        return Failed(
            msg=(
                f"检测到异常核: {', '.join(map(str, abnormal))}。"
                "这些核心的计算时间明显高于其他核心，请检查系统负载或硬件问题"
            ),
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


def _npu_stress(*, threshold: int) -> CheckOutcome:
    raw = NPU().execute()
    if raw is None:
        return Failed(
            msg="NPU 压测无法完成，因为 torch 或者 torch_npu 不可用",
            severity=Severity.WARNING,
            result_text="skipped",
        )
    abnormal = _find_abnormal(raw, threshold)
    if abnormal:
        return Failed(
            msg=(
                f"检测到异常设备: {', '.join(map(str, abnormal))}。"
                "这些设备的计算时间明显高于其他设备，请检查系统负载或硬件问题"
            ),
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


def _stress_checks(threshold: int) -> list[Check]:
    return [
        Check(
            "CPU stress test",
            GROUPS.STRESS,
            functools.partial(_cpu_stress, threshold=threshold),
        ),
        Check(
            "NPU stress test",
            GROUPS.STRESS,
            functools.partial(_npu_stress, threshold=threshold),
        ),
    ]


# ── public API ────────────────────────────────────────────────────────────────


def build_suite(
        *,
        framework: Framework,
        scene: str,
        rank_table_path: str = "",
        hardware: bool = False,
        threshold: int = 20,
) -> list[Check]:
    """
    Assemble and return the full list of ``Check`` objects for a deployment scenario.

    Args:
        framework:        The framework to be checked (MindIE, VLLM, SGLANG, etc).
        scene:            The deployment scenario (e.g. pd_mix, pd_disaggregation_ep).
        rank_table_path:  Path to the rank table file; enables network checks.
        hardware:         Whether to run hardware stress tests.
        threshold:        Abnormality threshold percentage for stress tests (default 20).

    Returns:
        Ordered list of checks: system → Ascend → network (opt) → stress (opt).
    """
    checks: list[Check] = []
    checks += _system_checks(framework)
    checks += _ascend_checks(framework, scene)
    if rank_table_path:
        checks += _network_checks(rank_table_path)
    if hardware:
        checks += _stress_checks(threshold)
    return checks
