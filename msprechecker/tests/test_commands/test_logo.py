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

# pylint: disable=duplicate-code

import re

import pytest

from msprechecker.commands.logo import _PLAIN_LOGO, _supports_color, print_logo, render_logo

_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")

_LOGO_WIDTH = 65
_EXPECTED_LINES = [
    "=" * _LOGO_WIDTH,
    ">>>>>   MindStudio   <<<<<".center(_LOGO_WIDTH),
    "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE".center(_LOGO_WIDTH),
    "=" * _LOGO_WIDTH,
]


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


@pytest.mark.parametrize("color", [False, True])
def test_render_logo_line_count(color):
    lines = render_logo(color=color).splitlines()
    assert len(lines) == 4


def test_render_logo_color_false_equals_plain_logo():
    assert render_logo(color=False) == _PLAIN_LOGO


def test_render_logo_color_false_no_ansi():
    assert "\033[" not in render_logo(color=False)


def test_render_logo_color_true_has_ansi():
    assert "\033[" in render_logo(color=True)


@pytest.mark.parametrize(
    ("line_index", "expected"),
    list(enumerate(_EXPECTED_LINES)),
)
def test_render_logo_plain_line_exact(line_index, expected):
    lines = render_logo(color=False).splitlines()
    assert lines[line_index] == expected


def test_plain_logo_no_leading_blank():
    assert not _PLAIN_LOGO.startswith("\n")


def test_color_strip_equals_plain():
    colored_lines = render_logo(color=True).splitlines()
    plain_lines = _PLAIN_LOGO.splitlines()
    for idx, plain in enumerate(plain_lines):
        assert _strip_ansi(colored_lines[idx]) == plain


@pytest.mark.parametrize(
    ("isatty", "term", "expected"),
    [
        (False, "xterm", False),
        (True, "dumb", False),
        (True, "unknown", False),
        (True, None, False),
        (True, "xterm", True),
        (True, "xterm-256color", True),
    ],
)
def test_supports_color(monkeypatch, isatty, term, expected):
    monkeypatch.setattr("sys.stderr.isatty", lambda: isatty)
    if term is None:
        monkeypatch.delenv("TERM", raising=False)
    else:
        monkeypatch.setenv("TERM", term)
    assert _supports_color() is expected


def test_render_logo_terminal_narrower_than_block_keeps_template_width():
    lines = render_logo(color=False, terminal_cols=60).splitlines()
    assert lines == _PLAIN_LOGO.splitlines()


@pytest.mark.parametrize("terminal_cols", [80, 120])
def test_render_logo_block_centered_in_terminal(terminal_cols):
    lines = render_logo(color=False, terminal_cols=terminal_cols).splitlines()
    assert all(len(line) == terminal_cols for line in lines)
    pad = (terminal_cols - _LOGO_WIDTH) // 2
    assert lines[0][pad : pad + _LOGO_WIDTH] == "=" * _LOGO_WIDTH
    assert lines[1][pad : pad + _LOGO_WIDTH] == ">>>>>   MindStudio   <<<<<".center(_LOGO_WIDTH)
    assert lines[2][pad : pad + _LOGO_WIDTH] == "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE".center(
        _LOGO_WIDTH
    )


def test_render_logo_color_with_terminal_cols_strip_matches_plain():
    terminal_cols = 120
    colored_lines = render_logo(color=True, terminal_cols=terminal_cols).splitlines()
    plain_lines = render_logo(color=False, terminal_cols=terminal_cols).splitlines()
    assert len(colored_lines) == len(plain_lines)
    for idx, colored_line in enumerate(colored_lines):
        assert _strip_ansi(colored_line) == plain_lines[idx]


def test_print_logo_trailing_blank(monkeypatch):
    writes = []

    def capture_write(data):
        writes.append(data)

    monkeypatch.setattr("sys.stderr.write", capture_write)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    monkeypatch.setattr(
        "msprechecker.commands.logo._terminal_cols",
        lambda: 120,
    )
    print_logo()
    assert len(writes) == 2
    assert writes[1] == "\n\n"
    logo_lines = writes[0].splitlines()
    assert len(logo_lines) == 4
    assert all(len(line) == 120 for line in logo_lines)
    pad = (120 - _LOGO_WIDTH) // 2
    assert logo_lines[0][pad : pad + _LOGO_WIDTH] == "=" * _LOGO_WIDTH
