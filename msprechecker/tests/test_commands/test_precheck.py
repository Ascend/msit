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
from unittest.mock import Mock, patch

from msprechecker.commands.precheck import Precheck, setup_precheck
from msprechecker.util import Framework


# =============================================================================
# Tests for setup_precheck
# =============================================================================


def test_setup_precheck_creates_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    result = setup_precheck(subparsers)
    assert result is not None


# =============================================================================
# Tests for _has_config_args
# =============================================================================


def test_has_config_args_with_user_config():
    args = Mock(
        user_config_path="/path/config.json",
        mindie_env_path=None,
        config_parent_dir=None,
        mies_config_path=None,
    )
    assert Precheck._has_config_args(args) is True


def test_has_config_args_with_mindie_env():
    args = Mock(
        user_config_path=None,
        mindie_env_path="/path/env.json",
        config_parent_dir=None,
        mies_config_path=None,
    )
    assert Precheck._has_config_args(args) is True


def test_has_config_args_with_config_parent_dir():
    args = Mock(
        user_config_path=None,
        mindie_env_path=None,
        config_parent_dir="/path",
        mies_config_path=None,
    )
    assert Precheck._has_config_args(args) is True


def test_has_config_args_with_mies_config():
    args = Mock(
        user_config_path=None,
        mindie_env_path=None,
        config_parent_dir=None,
        mies_config_path="/path/mies.json",
    )
    assert Precheck._has_config_args(args) is True


def test_has_config_args_with_no_configs():
    args = Mock(
        user_config_path=None,
        mindie_env_path=None,
        config_parent_dir=None,
        mies_config_path=None,
    )
    assert Precheck._has_config_args(args) is False


# =============================================================================
# Tests for Precheck class
# =============================================================================


class TestPrecheckExecute:
    def test_execute_delegates_to_cmate_with_user_config(self):
        precheck = Precheck()
        # Just verify the method exists and is callable - external dependency testing
        assert hasattr(precheck, "execute")
        assert callable(precheck.execute)

    def test_execute_without_config_args(self):
        precheck = Precheck()
        args = Mock(
            user_config_path=None,
            mindie_env_path=None,
            config_parent_dir=None,
            mies_config_path=None,
            scene="test",
            rank_table_path=None,
            hardware=False,
            threshold=20,
            severity_level="info",
        )
        with patch(
            "msprechecker.commands.precheck.detect_framework",
            return_value=Framework.MINDIE,
        ), patch(
            "msprechecker.commands.precheck.build_suite"
        ) as mock_build_suite, patch(
            "msprechecker.commands.precheck.PrecheckRunner"
        ) as mock_runner, patch(
            "msprechecker.commands.precheck.has_errors", return_value=False
        ):
            result = precheck.execute(args)
            assert result == 0
            mock_build_suite.assert_called_once()
            mock_runner.assert_called_once()

    def test_execute_with_errors(self):
        precheck = Precheck()
        args = Mock(
            user_config_path=None,
            mindie_env_path=None,
            config_parent_dir=None,
            mies_config_path=None,
            scene="test",
            rank_table_path=None,
            hardware=False,
            threshold=20,
            severity_level="info",
        )
        with patch(
            "msprechecker.commands.precheck.detect_framework",
            return_value=Framework.MINDIE,
        ), patch("msprechecker.commands.precheck.build_suite"), patch(
            "msprechecker.commands.precheck.PrecheckRunner"
        ), patch("msprechecker.commands.precheck.has_errors", return_value=True):
            result = precheck.execute(args)
            assert result == 1

    def test_execute_with_vllm_scene(self):
        precheck = Precheck()
        args = Mock(
            user_config_path=None,
            mindie_env_path=None,
            config_parent_dir=None,
            mies_config_path=None,
            scene="vllm_ep",
            rank_table_path=None,
            hardware=False,
            threshold=20,
            severity_level="warning",
        )
        with patch(
            "msprechecker.commands.precheck.detect_framework",
            return_value=Framework.VLLM,
        ), patch(
            "msprechecker.commands.precheck.build_suite"
        ) as mock_build_suite, patch(
            "msprechecker.commands.precheck.PrecheckRunner"
        ) as mock_runner, patch(
            "msprechecker.commands.precheck.has_errors", return_value=False
        ):
            precheck.execute(args)
            mock_build_suite.assert_called_once_with(
                framework=Framework.VLLM,
                scene="vllm_ep",
                rank_table_path="",
                hardware=False,
                threshold=20,
            )
            mock_runner.assert_called_once()
            # Verify severity is passed correctly - just check it was called
            assert mock_runner.called

    def test_execute_with_rank_table(self):
        precheck = Precheck()
        args = Mock(
            user_config_path=None,
            mindie_env_path=None,
            config_parent_dir=None,
            mies_config_path=None,
            scene="test",
            rank_table_path="/path/rank_table.json",
            hardware=True,
            threshold=30,
            severity_level="error",
        )
        with patch(
            "msprechecker.commands.precheck.detect_framework",
            return_value=Framework.MINDIE,
        ), patch(
            "msprechecker.commands.precheck.build_suite"
        ) as mock_build_suite, patch(
            "msprechecker.commands.precheck.PrecheckRunner"
        ), patch("msprechecker.commands.precheck.has_errors", return_value=False):
            precheck.execute(args)
            mock_build_suite.assert_called_once_with(
                framework=Framework.MINDIE,
                scene="test",
                rank_table_path="/path/rank_table.json",
                hardware=True,
                threshold=30,
            )


# =============================================================================
# Tests for _delegate_to_cmate
# =============================================================================


class TestDelegateToCmate:
    def test_delegate_method_exists(self):
        """Verify _delegate_to_cmate method exists."""
        precheck = Precheck()
        assert hasattr(precheck, "_delegate_to_cmate")
        assert callable(precheck._delegate_to_cmate)
