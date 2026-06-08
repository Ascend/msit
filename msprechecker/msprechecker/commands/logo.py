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

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

_COLOR_BORDER: Final = "\033[38;5;240m"
_COLOR_TEXT: Final = "\033[1;97m"
_COLOR_BRAND: Final = "\033[48;5;21;38;5;46m"
_COLOR_RESET: Final = "\033[0m"

_LOGO_WIDTH: Final = 65
_DEFAULT_COLS: Final = 80
_BRAND_INNER: Final = ">>>>>   MindStudio   <<<<<"
_SLOGAN_TEXT: Final = "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE"
_NO_COLOR_TERMS: Final = frozenset({"dumb", "unknown"})

LogoBlock = tuple[str, str, str, str]


def _build_logo_block(block_width: int = _LOGO_WIDTH) -> LogoBlock:
    border = "=" * block_width
    return border, _BRAND_INNER.center(block_width), _SLOGAN_TEXT.center(block_width), border


def _center_block_in_terminal(
    lines: LogoBlock,
    terminal_cols: Optional[int],
) -> LogoBlock:
    block_width = len(lines[0])
    if terminal_cols is None or terminal_cols <= block_width:
        return lines
    return (
        lines[0].center(terminal_cols),
        lines[1].center(terminal_cols),
        lines[2].center(terminal_cols),
        lines[3].center(terminal_cols),
    )


def _colorize_logo_block(lines: LogoBlock) -> LogoBlock:
    top, brand_line, slogan_line, bottom = lines
    before, after = brand_line.split("MindStudio", 1)
    return (
        f"{_COLOR_BORDER}{top}{_COLOR_RESET}",
        (f"{_COLOR_TEXT}{before}{_COLOR_BRAND}MindStudio{_COLOR_RESET}{_COLOR_TEXT}{after}{_COLOR_RESET}"),
        f"{_COLOR_TEXT}{slogan_line}{_COLOR_RESET}",
        f"{_COLOR_BORDER}{bottom}{_COLOR_RESET}",
    )


_PLAIN_LOGO: Final = "\n".join(_build_logo_block(_LOGO_WIDTH))


def _terminal_cols() -> int:
    return shutil.get_terminal_size(fallback=(_DEFAULT_COLS, 24)).columns


def _supports_color() -> bool:
    if not sys.stderr.isatty():
        return False
    term = os.environ.get("TERM")
    if term is None:
        return False
    return term not in _NO_COLOR_TERMS


def render_logo(
    *,
    color: bool,
    width: int = _LOGO_WIDTH,
    terminal_cols: Optional[int] = None,
) -> str:
    """Return four-line logo text without trailing blank line."""
    lines = _center_block_in_terminal(_build_logo_block(width), terminal_cols)
    if color:
        lines = _colorize_logo_block(lines)
    return "\n".join(lines)


def print_logo() -> None:
    """Write brand logo to stderr, centered in the current terminal when wider than the template."""
    sys.stderr.write(
        render_logo(
            color=_supports_color(),
            width=_LOGO_WIDTH,
            terminal_cols=_terminal_cols(),
        )
    )
    sys.stderr.write("\n\n")
