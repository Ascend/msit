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

import platform
from unittest.mock import Mock, patch

import pytest

from msprechecker.commands.banner import (
    AscendInfoSection,
    BannerPresenter,
    CpuInfoSection,
    InfoSection,
    NpuInfoSection,
    PlatformInfoSection,
    PythonInfoSection,
)


# =============================================================================
# Tests for InfoSection (abstract base)
# =============================================================================


def test_info_section_is_abstract():
    with pytest.raises(TypeError):
        InfoSection()


# =============================================================================
# Tests for PlatformInfoSection
# =============================================================================


def test_platform_info_section():
    section = PlatformInfoSection()
    info = section.get_info()
    assert "Platform:" in info
    assert platform.platform() in info


# =============================================================================
# Tests for PythonInfoSection
# =============================================================================


def test_python_info_section_with_installed_packages():
    with patch("msprechecker.commands.banner.get_pkg_version") as mock_get_version:
        mock_get_version.return_value = "1.0.0"
        section = PythonInfoSection(packages=["torch", "numpy"])
        info = section.get_info()
        assert "Python" in info
        assert platform.python_version() in info
        assert "torch-1.0.0" in info
        assert "numpy-1.0.0" in info


def test_python_info_section_with_not_installed_package():
    with patch("msprechecker.commands.banner.get_pkg_version") as mock_get_version:
        mock_get_version.return_value = None
        section = PythonInfoSection(packages=["nonexistent"])
        info = section.get_info()
        assert "nonexistent [not installed]" in info


def test_python_info_section_mixed_packages():
    with patch("msprechecker.commands.banner.get_pkg_version") as mock_get_version:

        def side_effect(pkg):
            return "1.0.0" if pkg == "torch" else None

        mock_get_version.side_effect = side_effect
        section = PythonInfoSection(packages=["torch", "missing"])
        info = section.get_info()
        assert "torch-1.0.0" in info
        assert "missing [not installed]" in info


# =============================================================================
# Tests for CpuInfoSection
# =============================================================================


def test_cpu_info_section_with_lscpu_data():
    with patch("msprechecker.commands.banner.Lscpu.execute") as mock_lscpu:
        mock_lscpu.return_value = {"Model name": "Intel(R) Core(TM) i7-9700K"}
        section = CpuInfoSection()
        info = section.get_info()
        assert "CPU:" in info
        assert "Intel(R) Core(TM) i7-9700K" in info
        assert f"({platform.machine()})" in info or "cores" in info


def test_cpu_info_section_with_none_result():
    with patch("msprechecker.commands.banner.Lscpu.execute") as mock_lscpu:
        mock_lscpu.return_value = None
        section = CpuInfoSection()
        info = section.get_info()
        assert "CPU:" in info
        assert "Unknown" in info


def test_cpu_info_section_with_empty_dict():
    with patch("msprechecker.commands.banner.Lscpu.execute") as mock_lscpu:
        mock_lscpu.return_value = {}
        section = CpuInfoSection()
        info = section.get_info()
        assert "CPU:" in info
        assert "Unknown" in info


# =============================================================================
# Tests for NpuInfoSection
# =============================================================================


def test_npu_info_section():
    with patch("msprechecker.commands.banner.get_npu_type") as mock_get_type, patch(
        "msprechecker.commands.banner.get_npu_count"
    ) as mock_get_count:
        mock_get_type.return_value = Mock(value="300")
        mock_get_count.return_value = 8
        section = NpuInfoSection()
        info = section.get_info()
        assert "NPU:" in info
        assert "300" in info
        assert "8 chips" in info


# =============================================================================
# Tests for AscendInfoSection
# =============================================================================


def test_ascend_info_section_with_data():
    with patch("msprechecker.commands.banner.Ascend.execute") as mock_ascend:
        mock_ascend.return_value = {
            "driver": {"Version": "24.1.0", "timestamp": "20240101"},
            "toolkit": {"Version": "7.0", "Commit": "abc123"},
        }
        section = AscendInfoSection()
        info = section.get_info()
        assert "Ascend:" in info
        assert "driver:" in info
        assert "toolkit:" in info
        assert "24.1.0" in info


def test_ascend_info_section_with_empty_data():
    with patch("msprechecker.commands.banner.Ascend.execute") as mock_ascend:
        mock_ascend.return_value = {}
        section = AscendInfoSection()
        info = section.get_info()
        assert "Ascend:" in info


def test_ascend_info_section_with_none():
    with patch("msprechecker.commands.banner.Ascend.execute") as mock_ascend:
        mock_ascend.return_value = None
        section = AscendInfoSection()
        info = section.get_info()
        assert "Ascend: not found" in info


def test_ascend_info_section_format_version_with_timestamp():
    section = AscendInfoSection()
    result = section._format_version({"Version": "24.1.0", "timestamp": "20240101"})
    assert "24.1.0" in result
    assert "20240101" in result


def test_ascend_info_section_format_version_with_commit():
    section = AscendInfoSection()
    result = section._format_version({"Version": "7.0", "Commit": "abc123"})
    assert "7.0" in result
    assert "abc123" in result


def test_ascend_info_section_format_version_empty():
    section = AscendInfoSection()
    result = section._format_version({})
    assert result == "not found"


# =============================================================================
# Tests for BannerPresenter
# =============================================================================


def test_banner_presenter_init_with_defaults():
    presenter = BannerPresenter()
    assert presenter.TITLE == "MindStudio Prechecker Tool"
    assert len(presenter.sections) == 5


def test_banner_presenter_init_with_custom_sections():
    custom_section = Mock(spec=InfoSection)
    custom_section.get_info.return_value = "Custom info"
    presenter = BannerPresenter(sections=[custom_section])
    assert len(presenter.sections) == 1


def test_banner_presenter_init_with_custom_packages():
    presenter = BannerPresenter(python_packages=["custom_pkg"])
    # Should not raise an error
    assert len(presenter.sections) == 5


def test_banner_presenter_add_section():
    presenter = BannerPresenter()
    custom_section = Mock(spec=InfoSection)
    presenter.add_section(custom_section)
    assert len(presenter.sections) == 6


def test_banner_presenter_render():
    from collections import namedtuple

    Size = namedtuple("Size", ["columns", "lines"])
    with patch("msprechecker.commands.banner.shutil.get_terminal_size") as mock_size:
        mock_size.return_value = Size(columns=80, lines=24)
        mock_section = Mock(spec=InfoSection)
        mock_section.get_info.return_value = "Test info"
        presenter = BannerPresenter(sections=[mock_section])
        output = presenter.render()
        assert "MindStudio Prechecker Tool" in output
        assert "Test info" in output
        assert "=" in output
        assert "-" in output


def test_banner_presenter_render_contains_all_sections():
    from collections import namedtuple

    Size = namedtuple("Size", ["columns", "lines"])
    with patch("msprechecker.commands.banner.shutil.get_terminal_size") as mock_size:
        mock_size.return_value = Size(columns=80, lines=24)
        section1 = Mock(spec=InfoSection)
        section1.get_info.return_value = "Section 1"
        section2 = Mock(spec=InfoSection)
        section2.get_info.return_value = "Section 2"
        presenter = BannerPresenter(sections=[section1, section2])
        output = presenter.render()
        assert "Section 1" in output
        assert "Section 2" in output


def test_banner_presenter_print_banner(capsys):
    from collections import namedtuple

    Size = namedtuple("Size", ["columns", "lines"])
    with patch("msprechecker.commands.banner.shutil.get_terminal_size") as mock_size:
        mock_size.return_value = Size(columns=80, lines=24)
        mock_section = Mock(spec=InfoSection)
        mock_section.get_info.return_value = "Test"
        presenter = BannerPresenter(sections=[mock_section])
        presenter.print_banner()
        captured = capsys.readouterr()
        assert "MindStudio Prechecker Tool" in captured.out
