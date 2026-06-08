# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import msprechecker.torch_cxx_abi as tca

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_get_torch_cxx11_abi_cache_hits_second_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return 1

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 1
    assert tca.get_torch_cxx11_abi() == 1
    assert calls["n"] == 1


def test_get_torch_cxx11_abi_cache_invalidates_on_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    key: list[tuple[str, int]] = [("/fake/torch", 1)]

    def fake_key() -> tuple[str, int]:
        return key[0]

    monkeypatch.setattr(tca, "_torch_install_key", fake_key)
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return calls["n"] % 2

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 1
    key[0] = ("/fake/torch", 2)
    assert tca.get_torch_cxx11_abi() == 0
    assert calls["n"] == 2


def test_get_torch_cxx11_abi_invalid_json_cache_falls_back_to_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    (tmp_path / "torch_cxx_abi.json").write_text("{not json", encoding="utf-8")
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return 1

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 1
    assert calls["n"] == 1


def test_get_torch_cxx11_abi_cache_missing_abi_falls_back_to_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    payload = {"root": "/fake/torch", "mtime_ns": 42}
    (tmp_path / "torch_cxx_abi.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return 0

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 0
    assert calls["n"] == 1


def test_get_torch_cxx11_abi_cache_invalid_abi_type_falls_back_to_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    payload = {"root": "/fake/torch", "mtime_ns": 42, "abi": "not-an-int"}
    (tmp_path / "torch_cxx_abi.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return 1

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 1
    assert calls["n"] == 1


def test_get_torch_cxx11_abi_detection_failure_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    cache_path = tmp_path / "torch_cxx_abi.json"
    calls = {"n": 0}

    def flaky_detect() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("transient torch import failure")
        return 1

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", flaky_detect)
    assert tca.get_torch_cxx11_abi() == 0
    assert not cache_path.is_file()
    assert tca.get_torch_cxx11_abi() == 1
    assert cache_path.is_file()
    assert calls["n"] == 2


def test_get_torch_cxx11_abi_cache_invalid_mtime_falls_back_to_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSPRECHECKER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tca, "_torch_install_key", lambda: ("/fake/torch", 42))
    payload = {"root": "/fake/torch", "mtime_ns": "not-a-number", "abi": 1}
    (tmp_path / "torch_cxx_abi.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls = {"n": 0}

    def fake_detect() -> int:
        calls["n"] += 1
        return 0

    monkeypatch.setattr(tca, "_detect_abi_via_torch_import", fake_detect)
    assert tca.get_torch_cxx11_abi() == 0
    assert calls["n"] == 1
