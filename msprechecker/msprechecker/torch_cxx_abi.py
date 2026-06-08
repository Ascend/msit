# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
"""Detect PyTorch C++11 ABI without importing ``torch`` on every process start.

``TB`` (ATB) path selection needs ``cxx_abi_0`` vs ``cxx_abi_1``. Importing
``torch`` is slow; we cache the result under the active torch install identity
(``torch`` package path + ``__init__.py`` mtime).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

_CACHE_ENV = "MSPRECHECKER_CACHE_DIR"
_CACHE_FILENAME = "torch_cxx_abi.json"


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    d = base / "msprechecker"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _torch_install_key() -> Optional[tuple[str, int]]:
    """Return ``(resolved_torch_root, mtime_ns)`` if a ``torch`` install is discoverable."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    root = Path(spec.origin).resolve().parent
    init_py = root / "__init__.py"
    if not init_py.is_file():
        return None
    return str(root), init_py.stat().st_mtime_ns


def _detect_abi_via_torch_import() -> int:
    """Import ``torch`` and read :func:`torch.compiled_with_cxx11_abi`."""
    import torch

    return 1 if torch.compiled_with_cxx11_abi() else 0


def _read_cache(cache_path: Path) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as f:
            return cast("dict[str, Any]", json.load(f))
    except (OSError, json.JSONDecodeError):
        logger.debug("Ignoring unreadable torch ABI cache %s", cache_path)
        return None


def _write_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(cache_path)
    except OSError:
        logger.debug("Could not write torch ABI cache %s", cache_path, exc_info=True)


def get_torch_cxx11_abi() -> int:
    """Return ``1`` if PyTorch was built with C++11 ABI, else ``0``.

    Uses a JSON cache file so repeated CLI invocations avoid ``import torch``
    when the installation is unchanged.

    Returns:
        ``0`` when torch is missing or detection fails.

    Raises:
        Nothing; failures downgrade to ABI ``0``.
    """
    key = _torch_install_key()
    if key is None:
        return 0
    root, mtime_ns = key
    cache_path = _cache_dir() / _CACHE_FILENAME
    cached = _read_cache(cache_path)
    cached_mtime: Optional[int] = None
    cache_payload: Optional[dict[str, Any]] = None
    if isinstance(cached, dict) and cached.get("root") == root:
        cache_payload = cached
        with suppress(TypeError, ValueError):
            cached_mtime = int(cached.get("mtime_ns", -1))
    if cache_payload is not None and cached_mtime == mtime_ns:
        abi_raw = cache_payload.get("abi")
        if abi_raw is not None:
            try:
                return int(abi_raw)
            except (TypeError, ValueError):
                pass

    try:
        abi = _detect_abi_via_torch_import()
    except (ImportError, AttributeError, RuntimeError):
        logger.debug("torch ABI import detection failed; defaulting to 0", exc_info=True)
        return 0

    _write_cache(
        cache_path,
        {"root": root, "mtime_ns": mtime_ns, "abi": abi},
    )
    return abi
