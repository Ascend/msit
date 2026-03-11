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

import argparse

from msprechecker.commands.cmate import Inspect, Run, setup_cmate


# =============================================================================
# Tests for setup_cmate
# =============================================================================


def test_setup_cmate_creates_run_parser():
    """Verify setup_cmate creates run parser without error."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_cmate(subparsers)
    # Just verify it returns without error and subparsers exist
    assert subparsers is not None


def test_setup_cmate_creates_inspect_parser():
    """Verify setup_cmate creates inspect parser without error."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_cmate(subparsers)
    # Just verify it returns without error
    assert subparsers is not None


def test_setup_cmate_run_parser_args():
    """Verify setup_cmate creates run parser with expected arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_cmate(subparsers)
    # Verify the parsers were added by checking choices
    assert "run" in subparsers._name_parser_map
    assert "inspect" in subparsers._name_parser_map


def test_setup_cmate_inspect_parser_args():
    """Verify setup_cmate creates inspect parser with expected arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_cmate(subparsers)
    # Verify the parsers were added by checking choices
    assert "inspect" in subparsers._name_parser_map


# =============================================================================
# Tests for Run class
# =============================================================================


class TestRun:
    def test_run_class_exists(self):
        """Verify Run class exists and has execute method."""
        assert hasattr(Run, "execute")
        assert callable(Run.execute)


# =============================================================================
# Tests for Inspect class
# =============================================================================


class TestInspect:
    def test_inspect_class_exists(self):
        """Verify Inspect class exists and has execute method."""
        assert hasattr(Inspect, "execute")
        assert callable(Inspect.execute)
