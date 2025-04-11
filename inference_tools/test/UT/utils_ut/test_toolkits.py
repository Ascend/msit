import re
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from msit.utils.exceptions import MsitException
from msit.utils.toolkits import (
    _INVALID_CHARS,
    CHECK_CSV_LEVEL_IGNORE,
    CHECK_CSV_LEVEL_REPLACE,
    CHECK_CSV_LEVEL_STRICT,
    DistBackend,
    filter_cmd,
    get_net_output_nodes_from_graph_def,
    get_rank,
    get_valid_name,
    is_input_yes,
    run_subprocess,
    safely_compute,
    sanitize_csv_value,
    seed_all,
    set_ld_preload,
    timestamp_sync,
)


class TestToolkitsFunctions(unittest.TestCase):
    def test_filter_cmd(self):
        cmd = ["ls", "-l", "|", "grep", "test", ">", "output.txt"]
        filtered_cmd = filter_cmd(cmd, _INVALID_CHARS)
        self.assertNotIn("|", filtered_cmd)
        self.assertNotIn(">", filtered_cmd)
        self.assertIn("ls", filtered_cmd)

    def test_get_valid_name(self):
        self.assertEqual(get_valid_name("/test/file.txt"), "test_file_txt")
        self.assertEqual(get_valid_name("path/to:file"), "path_to_file")

    @patch("msit.utils.log.logger.warning")
    def test_safely_compute(self, mock_warning):
        @safely_compute
        def divide(a, b):
            return a / b

        self.assertEqual(divide(10, 2), 5)
        self.assertIsNone(divide(10, 0))
        mock_warning.assert_called()

    @patch("msit.utils.log.logger.info")
    @patch("subprocess.Popen")
    def test_run_subprocess(self, mock_popen, mock_logger):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.communicate.return_value = (b"Output", b"")
        mock_process.returncode = 0
        mock_popen.return_value.__enter__.return_value = mock_process
        result = run_subprocess(["echo", "hello"], check_interval=0.001, capture_output=True)
        self.assertEqual(result, "hello\n")
        mock_logger.assert_called()

    @patch("msit.utils.log.logger.info")
    @patch("subprocess.Popen")
    def test_run_subprocess_with_none(self, mock_popen, mock_logger):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.communicate.return_value = (b"Output", b"")
        mock_process.returncode = 0
        mock_popen.return_value.__enter__.return_value = mock_process
        result = run_subprocess(["echo", "hello"], check_interval=0.001, capture_output=False)
        self.assertIsNone(result)
        mock_logger.assert_called()

    @patch("msit.utils.log.logger.error")
    @patch("msit.utils.toolkits.Popen")
    def test_run_subprocess_failure(self, mock_popen, mock_logger):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 1
        mock_popen.return_value.__enter__.return_value = mock_process
        with self.assertRaises(MsitException) as e:
            run_subprocess(["wrong_command"], check_interval=0.001, capture_output=True)
        self.assertIn("Failed to execute command:", str(e.exception))
        mock_logger.assert_called()

    @patch("msit.utils.toolkits.Popen")
    def test_run_subprocess_invalid_str_format(self, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 0
        mock_popen.return_value.__enter__.return_value = mock_process
        with self.assertRaises(MsitException) as e:
            run_subprocess("wrong_command", check_interval=0.001, capture_output=True)
        self.assertIn("[ERROR] invalid data type. `cmd` must be a list of strings.", str(e.exception))

    @patch("msit.utils.toolkits.Popen")
    def test_run_subprocess_invalid_minus_format(self, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 1
        mock_popen.return_value.__enter__.return_value = mock_process
        with self.assertRaises(MsitException) as e:
            run_subprocess(["ls"], check_interval=-1, capture_output=True)
        self.assertIn("[ERROR] invalid data type. `check_interval` must be a non-negative number.", str(e.exception))


class TestDistBackend(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("msit.utils.env.evars.get")
        self.mock_evars_get = self.patcher.start()
        self.mock_torch = MagicMock()
        DistBackend.torch = self.mock_torch

    def tearDown(self):
        self.patcher.stop()
        DistBackend.torch = None

    def test_get_visible_device_valid(self):
        self.mock_evars_get.return_value = "0,1"
        result = DistBackend._get_visible_device("CUDA_VISIBLE_DEVICES")
        self.assertEqual(result, 0)
        self.mock_evars_get.assert_called_once_with("CUDA_VISIBLE_DEVICES", "0")

    def test_get_visible_device_invalid(self):
        self.mock_evars_get.return_value = "invalid"
        with self.assertRaises(MsitException) as context:
            DistBackend._get_visible_device("ASCEND_VISIBLE_DEVICES")
        self.assertIn("Please check the value", str(context.exception))

    def test_is_device_available_npu_available(self):
        self.mock_evars_get.return_value = "0"
        self.mock_torch.npu.is_available.return_value = True
        result = DistBackend._is_device_available("npu", "ASCEND_VISIBLE_DEVICES")
        self.assertTrue(result)
        self.mock_evars_get.assert_called_with("ASCEND_VISIBLE_DEVICES", "0")

    def test_is_device_available_npu_unavailable(self):
        self.mock_evars_get.return_value = "0"
        self.mock_torch.npu.is_available.return_value = False
        result = DistBackend._is_device_available("npu", "ASCEND_VISIBLE_DEVICES")
        self.assertFalse(result)

    def test_is_device_available_cuda_available(self):
        self.mock_evars_get.return_value = "0"
        self.mock_torch.cuda.is_available.return_value = True
        result = DistBackend._is_device_available("cuda", "CUDA_VISIBLE_DEVICES")
        self.assertTrue(result)
        self.mock_evars_get.assert_called_with("CUDA_VISIBLE_DEVICES", "0")

    def test_is_device_available_cpu(self):
        self.assertTrue(DistBackend._is_device_available("cpu", ""))

    @patch.object(DistBackend, "_is_device_available")
    def test_get_global_device_priority(self, mock_is_available):
        mock_is_available.side_effect = lambda device, _: device == "npu"
        self.assertEqual(DistBackend._get_global_device(), "npu")
        mock_is_available.side_effect = lambda device, _: device == "cuda"
        self.assertEqual(DistBackend._get_global_device(), "cuda")
        mock_is_available.side_effect = lambda device, _: False
        self.assertEqual(DistBackend._get_global_device(), "cpu")

    def test_get_global_device_npu_available(self):
        self.mock_evars_get.return_value = "0"
        self.mock_torch.npu.is_available.return_value = True
        self.mock_torch.cuda.is_available.return_value = False
        self.assertEqual(DistBackend._get_global_device(), "npu")

    def test_get_global_device_fallback_to_cpu(self):
        self.mock_torch.npu.is_available.return_value = False
        self.mock_torch.cuda.is_available.return_value = False
        self.assertEqual(DistBackend._get_global_device(), "cpu")

    @patch.object(DistBackend, "_get_global_device")
    def test_get_method(self, mock_get_global):
        mock_get_global.return_value = "npu"
        self.assertEqual(DistBackend.get(), "hccl")
        mock_get_global.return_value = "cuda"
        self.assertEqual(DistBackend.get(), "nccl")
        mock_get_global.return_value = "cpu"
        self.assertEqual(DistBackend.get(), "gloo")
        mock_get_global.return_value = "unknown"
        self.assertEqual(DistBackend.get(), "cpu")


class TestTimestampSync(unittest.TestCase):
    @patch("msit.utils.dependencies.dependent.get")
    @patch("msit.utils.env.evars.get")
    def test_timestamp_sync(self, mock_evars_get, mock_dependent_get):
        mock_evars_get.side_effect = lambda key, default, typ=int: typ(default)
        mock_torch = MagicMock()
        mock_torch.distributed.is_initialized.return_value = False
        mock_dependent_get.return_value = mock_torch
        result = timestamp_sync(123456)
        self.assertEqual(result, 123456)

    @patch("msit.utils.toolkits.dependent.get")
    @patch("msit.utils.toolkits.evars.get")
    def test_single_process_returns_original(self, mock_evars_get, mock_dependent_get):
        mock_evars_get.return_value = 1
        mock_dependent_get.return_value = None

        result = timestamp_sync(12345)
        self.assertEqual(result, 12345)

    @patch("msit.utils.toolkits.dependent.get")
    @patch("msit.utils.toolkits.evars.get")
    @patch("msit.utils.toolkits.DistBackend.get")
    def test_distributed_sync_with_init(self, mock_backend_get, mock_evars_get, mock_dependent_get):
        mock_evars_get.side_effect = lambda key, default, *_: {"LOCAL_WORLD_SIZE": 4, "LOCAL_RANK": 2}.get(key, default)
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 54321
        mock_torch.tensor.return_value = mock_tensor

        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        mock_dist.ReduceOp.MAX = "MAX"
        mock_torch.distributed = mock_dist

        mock_dependent_get.return_value = mock_torch
        mock_backend_get.return_value = "nccl"

        result = timestamp_sync(12345)
        self.assertEqual(result, 54321)
        mock_torch.tensor.assert_called_once_with(12345)
        mock_dist.init_process_group.assert_called_once_with(backend="nccl", rank=2, world_size=4)
        mock_dist.all_reduce.assert_called_once_with(mock_tensor, op="MAX")
        mock_tensor.item.assert_called_once()

    @patch("msit.utils.toolkits.dependent.get")
    @patch("msit.utils.toolkits.evars.get")
    def test_already_initialized(self, mock_evars_get, mock_dependent_get):
        mock_evars_get.side_effect = lambda key, default, *_: {"LOCAL_WORLD_SIZE": 4, "LOCAL_RANK": 2}.get(key, default)
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 54321
        mock_torch.tensor.return_value = mock_tensor

        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.ReduceOp.MAX = "MAX"
        mock_torch.distributed = mock_dist

        mock_dependent_get.return_value = mock_torch
        result = timestamp_sync(12345)
        self.assertEqual(result, 54321)
        mock_dist.init_process_group.assert_not_called()

    @patch("msit.utils.toolkits.dependent.get")
    @patch("msit.utils.toolkits.evars.get")
    def test_no_torch_returns_original(self, mock_evars_get, mock_dependent_get):
        mock_evars_get.return_value = 4
        mock_dependent_get.return_value = None
        result = timestamp_sync(12345)
        self.assertEqual(result, 12345)


class TestGetRank(unittest.TestCase):
    @patch("msit.utils.dependencies.dependent.get")
    def test_torch_initialized_returns_rank(self, mock_dependent_get):
        mock_torch = MagicMock()
        mock_torch.distributed.is_initialized.return_value = True
        mock_torch.distributed.get_rank.return_value = 2
        mock_dependent_get.return_value = mock_torch
        result = get_rank()
        self.assertEqual(result, "2")

    @patch("msit.utils.dependencies.dependent.get")
    def test_torch_not_initialized_returns_empty(self, mock_dependent_get):
        mock_torch = MagicMock()
        mock_torch.distributed.is_initialized.return_value = False
        mock_dependent_get.return_value = mock_torch
        result = get_rank()
        self.assertEqual(result, "")

    @patch("msit.utils.dependencies.dependent.get")
    def test_no_torch_returns_empty(self, mock_dependent_get):
        mock_dependent_get.return_value = None
        result = get_rank()
        self.assertEqual(result, "")


class TestSeedAll(unittest.TestCase):
    @patch("msit.utils.toolkits.evars.set")
    @patch("msit.utils.toolkits.seed")
    @patch("numpy.random.seed")
    @patch("msit.utils.toolkits.dependent.get")
    @patch("msit.utils.toolkits.logger.info")
    def test_seed_all_with_full_deps(
        self, mock_logger, mock_dependent_get, mock_np_seed, mock_random_seed, mock_evars_set
    ):
        mock_torch = MagicMock()
        mock_torch.version.cuda = "10.2.100"
        mock_torch.cuda = MagicMock()
        mock_torch.backends = MagicMock()
        mock_torch_npu = MagicMock()

        mock_dependent_get.side_effect = lambda x: {"torch": mock_torch, "torch_npu": mock_torch_npu}.get(x)

        seed_all(666)
        expected_evar_calls = [
            call("LCCL_DETERMINISTIC", "1"),
            call("HCCL_DETERMINISTIC", "true"),
            call("PYTHONHASHSEED", "666"),
            call("ATB_MATMUL_SHUFFLE_K_ENABLE", "0"),
            call("ATB_LLM_LCOC_ENABLE", "0"),
            call("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ]
        mock_evars_set.assert_has_calls(expected_evar_calls, any_order=True)
        mock_random_seed.assert_called_once_with(666)
        mock_np_seed.assert_called_once_with(666)

        mock_torch.manual_seed.assert_called_once_with(666)
        mock_torch.use_deterministic_algorithms.assert_called_once_with(mode=True)
        mock_torch.cuda.manual_seed.assert_called_once_with(666)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(666)
        mock_torch.backends.cudnn.deterministic = True
        mock_torch.backends.cudnn.enable = False
        mock_torch.backends.cudnn.benchmark = False

        mock_torch_npu.npu.manual_seed.assert_called_once_with(666)
        mock_torch_npu.npu.manual_seed_all.assert_called_once_with(666)

        mock_logger.assert_called_once_with("Enable deterministic computation sucess! current seed is 666.")

    @patch("msit.utils.toolkits.evars.set")
    @patch("msit.utils.toolkits.dependent.get")
    def test_seed_all_without_cuda(self, mock_dependent_get, mock_evars_set):
        mock_torch = MagicMock()
        del mock_torch.cuda
        mock_torch.version.cuda = None
        mock_dependent_get.return_value = mock_torch
        seed_all(666)
        self.assertFalse(hasattr(mock_torch, "cuda"))
        cublas_calls = [call.args for call in mock_evars_set.mock_calls if call.args[0] == "CUBLAS_WORKSPACE_CONFIG"]
        self.assertEqual(len(cublas_calls), 0)


class MockNode:
    def __init__(self, name, inputs):
        self.name = name
        self.input = inputs


class TestGetNetOutputNodes(unittest.TestCase):
    def test_single_output_node(self):
        node_a = MockNode("A", [])
        node_b = MockNode("B", ["A"])
        node_c = MockNode("C", ["B"])
        graph_def = Mock()
        graph_def.node = [node_a, node_b, node_c]
        result = get_net_output_nodes_from_graph_def(graph_def)
        self.assertEqual(result, ["C"])

    def test_multiple_output_nodes(self):
        graph_def = MagicMock()
        node_a = MockNode("A", [])
        node_b = MockNode("B", ["A"])
        node_c = MockNode("C", ["A"])
        graph_def.node = [node_a, node_b, node_c]
        result = get_net_output_nodes_from_graph_def(graph_def)
        self.assertCountEqual(result, ["B", "C"])

    def test_empty_graph(self):
        graph_def = MagicMock()
        graph_def.node = []
        result = get_net_output_nodes_from_graph_def(graph_def)
        self.assertEqual(result, [])


class TestSanitizeCsvValue(unittest.TestCase):
    def test_sanitize_csv_value_ignore(self):
        value = "malicious;value"
        result = sanitize_csv_value(value, CHECK_CSV_LEVEL_IGNORE)
        self.assertEqual(result, value)

    def test_sanitize_csv_value_non_string(self):
        value = 123
        result = sanitize_csv_value(value, CHECK_CSV_LEVEL_STRICT)
        self.assertEqual(result, value)

    def test_sanitize_csv_value_safe_number(self):
        value = "3.14"
        result = sanitize_csv_value(value, CHECK_CSV_LEVEL_STRICT)
        self.assertEqual(result, value)

    @patch("msit.utils.toolkits._MALICIOUS_CSV_PATTERN", re.compile(r";"))
    def test_sanitize_csv_value_malicious_strict(self):
        value = "malicious;value"
        with self.assertRaises(MsitException) as e:
            sanitize_csv_value(value, CHECK_CSV_LEVEL_STRICT)
        self.assertIn(
            "Malicious value detected: malicious;value, please check the value written to the csv.", str(e.exception)
        )

    @patch("msit.utils.toolkits._MALICIOUS_CSV_PATTERN", re.compile(r";"))
    def test_sanitize_csv_value_malicious_replace(self):
        value = "malicious;value"
        result = sanitize_csv_value(value, CHECK_CSV_LEVEL_REPLACE)
        self.assertEqual(result, "<REPLACEMENT>")


class TestIsInputYes(unittest.TestCase):
    @patch("builtins.input", return_value="yes")
    def test_is_input_yes_positive(self, mock_input):
        self.assertTrue(is_input_yes("Prompt: "))

    @patch("builtins.input", return_value="no")
    def test_is_input_yes_negative(self, mock_input):
        self.assertFalse(is_input_yes("Prompt: "))

    @patch("builtins.input", return_value="  YES  ")
    def test_is_input_yes_whitespace(self, mock_input):
        self.assertTrue(is_input_yes("Prompt: "))

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("msit.utils.toolkits.logger.info")
    def test_is_input_yes_interrupted(self, mock_logger, mock_input):
        self.assertFalse(is_input_yes("Prompt: "))
        mock_logger.assert_called_with('Input interrupted. Defaulting to "no".')


class TestSetLdPreload(unittest.TestCase):
    @patch("msit.utils.toolkits.evars")
    @patch("msit.utils.toolkits.logger")
    def test_existing_ld_preload_updates_value(self, mock_logger: MagicMock, mock_evars: MagicMock):
        mock_evars.get.return_value = "existing_lib.so"
        set_ld_preload("new_lib.so")
        mock_evars.get.assert_called_once_with("LD_PRELOAD", required=False)
        mock_evars.set.assert_called_once_with("LD_PRELOAD", "new_lib.so:existing_lib.so")
        mock_logger.info.assert_called_once_with("Environment updated with .so library new_lib.so.")

    @patch("msit.utils.toolkits.evars")
    @patch("msit.utils.toolkits.logger")
    def test_no_existing_ld_preload_sets_new_value(self, mock_logger: MagicMock, mock_evars: MagicMock):
        mock_evars.get.return_value = None
        set_ld_preload("new_lib.so")
        mock_evars.get.assert_called_once_with("LD_PRELOAD", required=False)
        mock_evars.set.assert_called_once_with("LD_PRELOAD", "new_lib.so")
        mock_logger.info.assert_called_once_with("Environment updated with .so library new_lib.so.")

    @patch("msit.utils.toolkits.evars")
    @patch("msit.utils.toolkits.logger")
    def test_empty_ld_preload_sets_new_value(self, mock_logger: MagicMock, mock_evars: MagicMock):
        mock_evars.get.return_value = ""
        set_ld_preload("new_lib.so")
        mock_evars.set.assert_called_once_with("LD_PRELOAD", "new_lib.so")
