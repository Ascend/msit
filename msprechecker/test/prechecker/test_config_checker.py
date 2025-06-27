# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from msprechecker.prechecker.config_checker import (
    MindieConfigChecker,
    RankTableChecker,
    ModelConfigChecker,
    UserConfigChecker,
    MindIEEnvChecker,
    CheckResult,
)
from msprechecker.prechecker.utils import MINDIE_SERVICE_DEFAULT_PATH


class TestMindieConfigChecker(unittest.TestCase):
    def setUp(self):
        self.checker = MindieConfigChecker()

    @patch.dict("os.environ", {}, clear=True)
    @patch("os.stat", side_effect=OSError)
    def test_collect_env_with_no_env_vars_and_no_default_config(self, _):
        self.assertIsNone(self.checker.collect_env())
        self.assertEqual(self.checker.config_path, os.path.join(MINDIE_SERVICE_DEFAULT_PATH, "conf", "config.json"))

    @patch.dict("os.environ", {"MIES_INSTALL_PATH": "/random/path"}, clear=True)
    def test_collect_env_with_mies_install_path_env_var(self):
        self.assertIsNone(self.checker.collect_env())
        self.assertEqual(self.checker.config_path, os.path.join("/random/path", "conf", "config.json"))

    def test_collect_env_with_directory_path_creates_default_config_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            random_path = os.path.join(temp_dir, "random_path")
            with open(random_path, "w") as f:
                json.dump({"ServerConfig": {"httpsEnabled": True}}, f)

            self.assertIsNone(self.checker.collect_env(random_path))
            self.assertEqual(self.checker.config_path, os.path.join(random_path, "conf", "config.json"))
    
    def test_collect_env_with_json_file_path_returns_config_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            random_path = os.path.join(temp_dir, "random_path.json")
            data = {"ServerConfig": {"httpsEnabled": True}}
            with open(random_path, "w") as f:
                json.dump(data, f)

            self.assertEqual(self.checker.collect_env(random_path), data)
            self.assertEqual(self.checker.config_path, random_path)

    def test_do_precheck_with_none_config_returns_none(self):
        self.assertIsNone(self.checker.do_precheck(None))

    def test_do_precheck_with_valid_config_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            random_path = os.path.join(temp_dir, "random_path.json")
            data = {"ServerConfig": {"httpsEnabled": True}}
            with open(random_path, "w") as f:
                json.dump(data, f)

            self.assertEqual(self.checker.collect_env(random_path), data)
            self.assertEqual(self.checker.config_path, random_path)
            self.assertIsNone(self.checker.do_precheck(data))


class TestRankTableChecker(unittest.TestCase):
    def setUp(self):
        self.checker = RankTableChecker()

    @patch.dict("os.environ", {}, clear=True)
    def test_collect_env_1(self):
        self.assertEqual(self.checker.collect_env(), {})
        self.assertIsNone(self.checker.config_path)

    @patch.dict("os.environ", {"RANKTABLEFILE": "/random/path"}, clear=True)
    def test_collect_env_2(self):
        self.assertEqual(self.checker.collect_env(), {})
        self.assertIsNone(self.checker.config_path)

    def test_collect_env_3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            random_path = os.path.join(temp_dir, "random_path")
            with open(random_path, "w") as f:
                json.dump({"Version": 1.0, "server_count": "2"}, f)

            self.assertEqual(self.checker.collect_env(random_path), {})
            self.assertEqual(self.checker.config_path, random_path)


class TestModelConfigChecker(unittest.TestCase):
    def setUp(self):
        self.checker = ModelConfigChecker()

    @patch('msprechecker.prechecker.config_checker.get_model_path_from_mindie_config')
    def test_collect_env_no_mindie_config(self, mock_get_model):
        mock_get_model.return_value = (None, None)
        result = self.checker.collect_env()
        self.assertIsNone(result)

    @patch('msprechecker.prechecker.config_checker.get_model_path_from_mindie_config')
    def test_collect_env_with_valid_config(self, mock_get_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested temp dir for model weights
            with tempfile.TemporaryDirectory(dir=temp_dir) as model_weight_path:
                config_path = os.path.join(model_weight_path, "config.json")
                config_data = {"torch_dtype": "float16", "transformers_version": "4.28.0"}
                with open(config_path, 'w') as f:
                    json.dump(config_data, f)
                
                mock_get_model.return_value = ("test_model", model_weight_path)
                result = self.checker.collect_env()
                
                self.assertIsNotNone(result)
                self.assertEqual(result["model_name"], "test_model")
                self.assertEqual(result["model_config"], config_data)
                self.assertEqual(self.checker.config_path, config_path)

    def test_do_precheck_no_config(self):
        with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
            self.checker.do_precheck(None)
            mock_show.assert_not_called()

    def test_do_precheck_invalid_torch_dtype(self):
        config = {
            "model_name": "test_model",
            "model_config": {"torch_dtype": "float32"}
        }
        
        with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
            self.checker.do_precheck(config)
            mock_show.assert_called_once()
            args, kwargs = mock_show.call_args
            self.assertEqual(args[0], "ModelConfig")
            self.assertEqual(args[1], "torch_dtype")
            self.assertEqual(args[2], CheckResult.ERROR)

    def test_do_precheck_valid_torch_dtype(self):
        config = {
            "model_name": "test_model",
            "model_config": {"torch_dtype": "float16"}
        }
        
        with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
            self.checker.do_precheck(config)
            mock_show.assert_not_called()

    @patch('msprechecker.prechecker.config_checker.show_check_result')
    def test_do_precheck_outdated_transformers(self, mock_show):
        config = {
            "model_name": "test_model",
            "model_config": {"transformers_version": "99.99.0"}
        }
        
        mock_transformers = MagicMock()
        mock_transformers.__version__ = "4.28.0"
        
        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            self.checker.do_precheck(config)
            mock_show.assert_called_once()
            args, _ = mock_show.call_args
            self.assertEqual(args[0], "ModelConfig")
            self.assertEqual(args[1], "transformers_version")
            self.assertEqual(args[2], CheckResult.ERROR)

    def test_do_precheck_missing_transformers(self):
        config = {
            "model_name": "test_model",
            "model_config": {"transformers_version": "4.28.0"}
        }
        
        with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
            with patch.dict('sys.modules', {"transformers": None}):
                self.checker.do_precheck(config)
                mock_show.assert_called_once()
                args, _ = mock_show.call_args
                self.assertEqual(args[0], "ModelConfig")
                self.assertEqual(args[1], "transformers_version")
                self.assertEqual(args[2], CheckResult.ERROR)

    def test_do_precheck_deepseek_model(self):
        config = {
            "model_name": "deepseek_test_model",
            "model_config": {"some_config": "value"}
        }
        
        with patch('msprechecker.prechecker.config_checker.is_deepseek_model') as mock_is_deepseek:
            mock_is_deepseek.return_value = True
            with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
                self.checker.do_precheck(config)
                mock_show.assert_not_called()

    def test_do_precheck_non_deepseek_model(self):
        config = {
            "model_name": "other_test_model",
            "model_config": {"some_config": "value"}
        }
        
        with patch('msprechecker.prechecker.config_checker.is_deepseek_model') as mock_is_deepseek:
            mock_is_deepseek.return_value = False
            with patch('msprechecker.prechecker.config_checker.show_check_result') as mock_show:
                self.checker.do_precheck(config)
                mock_show.assert_not_called()

    @patch('msprechecker.prechecker.config_checker.get_model_path_from_mindie_config')
    def test_collect_env_missing_config_file(self, mock_get_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory(dir=temp_dir) as model_weight_path:
                mock_get_model.return_value = ("test_model", model_weight_path)
                result = self.checker.collect_env()
                
                self.assertIsNotNone(result)
                self.assertEqual(result["model_name"], "test_model")
                self.assertEqual(result["model_config"], {})

    @patch('msprechecker.prechecker.config_checker.get_model_path_from_mindie_config')
    def test_collect_env_invalid_config_file(self, mock_get_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory(dir=temp_dir) as model_weight_path:
                config_path = os.path.join(model_weight_path, "config.json")
                with open(config_path, 'w') as f:
                    f.write("invalid json")
                
                mock_get_model.return_value = ("test_model", model_weight_path)
                result = self.checker.collect_env()
                
                self.assertIsNotNone(result)
                self.assertEqual(result["model_name"], "test_model")
                self.assertEqual(result["model_config"], {})


class TestUserConfigChecker(unittest.TestCase):
    def setUp(self):
        self.checker = UserConfigChecker()
        self.default_rules = {
            "deploy_config": {
                "p_instances_num": {
                    "expected": {
                        "type": ">=",
                        "value": 1
                    },
                    "reason": "p_instances_num A2场景应该大于等于1",
                    "severity": "high"
                },
                "d_instances_num": {
                    "expected": {
                        "type": ">=",
                        "value": 1
                    },
                    "reason": "d_instances_num A2场景应该大于等于1",
                    "severity": "high"
                },
                "single_p_instance_pod_num": {
                    "expected": {
                        "type": ">=",
                        "value": 2
                    },
                    "reason": "single_p_instance_pod_num A2场景应该大于等于2",
                    "severity": "high"
                },
                "single_d_instance_pod_num": {
                    "expected": {
                        "type": ">=",
                        "value": 4
                    },
                    "reason": "single_d_instance_pod_num A2场景应该大于等于4",
                    "severity": "high"
                }
            }
        }

    def test_user_config_with_valid_values(self):
        test_config = {
            "version": "v1.0",
            "deploy_config": {
                "p_instances_num": 1,
                "d_instances_num": 1,
                "single_p_instance_pod_num": 2,
                "single_d_instance_pod_num": 8
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "user_config.json")
            with open(config_path, 'w') as f:
                json.dump(test_config, f)

            with patch('msprechecker.prechecker.config_checker.get_default_rule') as mock_get_rule:
                mock_get_rule.return_value = self.default_rules
                with patch.object(self.checker, 'logger') as mock_logger:
                    # Test collect_env
                    collected = self.checker.collect_env(config_path)
                    self.assertEqual(collected, test_config)
                    
                    # Test do_precheck with valid config
                    self.checker.do_precheck(collected)
                    
                    # Verify all checks passed
                    mock_logger.info.assert_called_with(
                        "- config: All user config fields passed the checks "
                        "[Severity: (high,)]."
                    )

    def test_user_config_with_invalid_values(self):
        test_config = {
            "version": "v1.0",
            "deploy_config": {
                "p_instances_num": 0,  # Invalid
                "d_instances_num": 1,
                "single_p_instance_pod_num": 1,  # Invalid
                "single_d_instance_pod_num": 3   # Invalid
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "user_config.json")
            with open(config_path, 'w') as f:
                json.dump(test_config, f)

            with patch('msprechecker.prechecker.config_checker.get_default_rule') as mock_get_rule:
                mock_get_rule.return_value = self.default_rules
                with patch.object(self.checker, 'logger') as mock_logger:
                    # Test collect_env
                    collected = self.checker.collect_env(config_path)
                    
                    # Test do_precheck with invalid config
                    self.checker.do_precheck(collected)
                    
                    # Verify error messages were logged
                    output = '\n'.join([call[0][0] for call in mock_logger.info.call_args_list])
                    self.assertIn("User Config", output)
                    self.assertIn("p_instances_num", output)
                    self.assertIn("single_p_instance_pod_num", output)
                    self.assertIn("single_d_instance_pod_num", output)


class TestMindIEEnvChecker(unittest.TestCase):
    def setUp(self):
        self.checker = MindIEEnvChecker()
        self.default_rules = {
            "MINDIE_ENABLE_DP_DISTRIBUTED": {
                "expected": {
                    "type": "==",
                    "value": 1
                },
                "reason": "Should be enabled",
                "severity": "high"
            },
            "DP_PARTITION_UP_ENABLE": {
                "expected": {
                    "type": "==",
                    "value": 1
                },
                "reason": "Should be enabled",
                "severity": "high"
            },
            "HCCL_BUFFSIZE": {
                "expected": {
                    "type": ">=",
                    "value": 512
                },
                "reason": "Should be at least 512",
                "severity": "medium"
            },
            "ATB_CONTEXT_WORKSPACE_SIZE": {
                "expected": {
                    "type": "==",
                    "value": 0
                },
                "reason": "Should be 0",
                "severity": "high"
            }
        }

    def test_mindie_env_with_valid_values(self):
        test_config = {
            "mindie_server_decode_env": {
                "MINDIE_ENABLE_DP_DISTRIBUTED": 1,
                "DP_PARTITION_UP_ENABLE": 1,
                "DP_MOVE_UP_ENABLE": 1,
                "HCCL_BUFFSIZE": 512,
                "ATB_LAYER_INTERNAL_TENSOR_REUSE": 1,
                "ATB_CONVERT_NCHW_TO_ND": 1,
                "ATB_CONTEXT_WORKSPACE_SIZE": 0
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "mindie_env.json")
            with open(config_path, 'w') as f:
                json.dump(test_config, f)

            with patch('msprechecker.prechecker.config_checker.get_default_rule') as mock_get_rule:
                mock_get_rule.return_value = self.default_rules
                with patch.object(self.checker, 'logger') as mock_logger:
                    # Test collect_env
                    collected = self.checker.collect_env(mindie_env_config_path=config_path)
                    self.assertEqual(collected, test_config)
                    
                    # Test do_precheck with valid config
                    self.checker.do_precheck(collected['mindie_server_decode_env'])
                    
                    # Verify all checks passed
                    mock_logger.info.assert_called_with(
                        "- config: All mindie env fields passed the checks "
                        "[Severity: (high,)]."
                    )

    def test_mindie_env_with_invalid_values(self):
        test_config = {
            "mindie_server_decode_env": {
                "MINDIE_ENABLE_DP_DISTRIBUTED": 0,  # Invalid
                "DP_PARTITION_UP_ENABLE": 0,        # Invalid
                "DP_MOVE_UP_ENABLE": 1,
                "HCCL_BUFFSIZE": 256,               # Invalid
                "ATB_LAYER_INTERNAL_TENSOR_REUSE": 1,
                "ATB_CONVERT_NCHW_TO_ND": 1,
                "ATB_CONTEXT_WORKSPACE_SIZE": 1     # Invalid
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "mindie_env.json")
            with open(config_path, 'w') as f:
                json.dump(test_config, f)

            with patch('msprechecker.prechecker.config_checker.get_default_rule') as mock_get_rule:
                mock_get_rule.return_value = self.default_rules
                with patch.object(self.checker, 'logger') as mock_logger:
                    # Test collect_env
                    collected = self.checker.collect_env(mindie_env_config_path=config_path)
                    
                    # Test do_precheck with invalid config
                    self.checker.do_precheck(collected)
                    
                    # Verify error messages were logged
                    output = '\n'.join([call[0][0] for call in mock_logger.info.call_args_list])
                    self.assertIn("MindIE Env", output)
                    self.assertIn("MINDIE_ENABLE_DP_DISTRIBUTED", output)
                    self.assertIn("DP_PARTITION_UP_ENABLE", output)
                    self.assertIn("ATB_CONTEXT_WORKSPACE_SIZE", output)
