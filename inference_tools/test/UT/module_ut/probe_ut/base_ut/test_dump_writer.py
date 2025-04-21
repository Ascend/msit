import sys
import unittest
from unittest.mock import MagicMock, call, patch

from msit.common.dirs import DirPool
from msit.module.probe.base.dump_writer import _WITHOUT_CALL_STACK, WriterDump
from msit.utils.constants import CfgConst, DumpConst
from msit.utils.log import logger
from msit.utils.toolkits import get_valid_name


class TestWriterDump(unittest.TestCase):
    def setUp(self):
        class TestWriterDumpConcrete(WriterDump):
            def summ_dump_data(self):
                return "test_result"

        self.writer_cls = TestWriterDumpConcrete
        self.task = CfgConst.TASK_TENSOR
        self.mock_get_rank_dir = patch.object(DirPool, "get_rank_dir", return_value="/fake/rank_dir").start()
        self.mock_get_tensor_dir = patch.object(DirPool, "get_tensor_dir", return_value="/fake/tensor_dir").start()
        self.mock_get_model_dir = patch.object(DirPool, "get_model_dir", return_value="/fake/model_dir").start()
        self.mock_make_tensor_dir = patch.object(DirPool, "make_tensor_dir").start()
        self.mock_dirpool = patch("msit.common.dirs.DirPool").start()
        self.mock_datastat = patch("msit.common.stat.DataStat").start()
        self.addCleanup(patch.stopall)

    def test_init(self):
        writer = self.writer_cls(self.task)
        self.assertEqual(writer.task, self.task)
        self.assertEqual(writer.max_cache_size, 1_048_576)
        self.assertEqual(writer.cache_dump_json_size, 0)
        self.assertIn(CfgConst.TASK, writer.cache_dump_json)

    @patch("msit.module.probe.base.dump_writer.stack")
    def test_call_stack(self, mock_stack):
        mock_stack.return_value = [
            (None, "msit/core/module.py", 10, "func1", ["code1"], None),
            (None, "user_script.py", 20, "func2", ["code2"], None),
        ]
        writer = self.writer_cls(self.task)
        stack_info = writer._call_stack("test_node")
        self.assertIn("test_node", stack_info)
        self.assertNotIn("msit/core", stack_info["test_node"][0])

    @patch("msit.module.probe.base.dump_writer.stack", side_effect=Exception("mock error"))
    @patch.object(logger, "warning")
    def test_call_stack_exception_handling(self, mock_warn, mock_stack):
        writer = self.writer_cls(self.task)
        result = writer._call_stack("test_node")
        mock_warn.assert_called_once_with("The call stack of test_node failed to retrieve, mock error.")
        self.assertEqual(result, {"test_node": [_WITHOUT_CALL_STACK]})

    def test_remove_colon_with_colon(self):
        writer = self.writer_cls(self.task)
        self.assertEqual(writer._remove_colon("node:output"), "node")

    def test_remove_colon_without_colon(self):
        writer = self.writer_cls(self.task)
        self.assertEqual(writer._remove_colon("node_output"), "node_output")

    @patch.object(WriterDump, "_save_stack_json")
    def test_update_stack_behavior(self, mock_save):
        writer = self.writer_cls(self.task)
        mock_stack_data = {"node1": ["stack_line1"]}
        with patch.object(writer, "_call_stack", return_value=mock_stack_data):
            writer.update_stack("node1")
            self.assertEqual(writer.cache_stack_json, mock_stack_data)
            self.assertEqual(writer.cache_stack_json_size, sys.getsizeof(mock_stack_data))
            mock_save.assert_not_called()
            writer.cache_stack_json_size = writer.max_cache_size - 1
            writer.update_stack("node2")
            mock_save.assert_called_once()
            self.assertEqual(writer.cache_stack_json_size, 0)

    @patch.object(WriterDump, "_save_dump_json")
    def test_update_stat_flushes_cache(self, mock_save):
        writer = self.writer_cls(self.task)
        writer.max_cache_size = 100
        self.mock_datastat.collect_stats_for_numpy.return_value = {"mean": 0.5}

        with patch("sys.getsizeof", return_value=150):
            writer.cache_dump_json[DumpConst.DATA].setdefault(get_valid_name("node1"), {})
            writer.update_stat("node1", "input", "arg1", MagicMock())
            mock_save.assert_called_once()

    @patch.object(WriterDump, "update_stat")
    @patch.object(WriterDump, "_save_tensor_data")
    def test_through_inputs(self, mock_save_tensor, mock_update_stat):
        writer = self.writer_cls(self.task)
        inputs = [MagicMock(name="input1"), "input2"]
        input_map = {"input1": "data1", "input2": "data2"}
        writer.through_inputs(inputs, "node1", input_map)
        self.assertEqual(mock_update_stat.call_count, 2)
        self.assertEqual(mock_save_tensor.call_count, 2)

    @patch.object(WriterDump, "update_stat")
    @patch.object(WriterDump, "_save_tensor_data")
    @patch.object(logger, "info")
    def test_through_outputs_behavior(self, mock_info, mock_save_tensor, mock_update_stat):
        writer = self.writer_cls(self.task)
        mock_output_obj = MagicMock(name="output_obj")
        mock_output_obj.name = "output2"
        outputs = ["output1", mock_output_obj]
        output_map = {"output1": "data1", "output2": "data2"}
        writer.net_output_nodes = ["output2"]
        writer.through_outputs(outputs, "test_node", output_map)
        calls = [
            call("test_node", DumpConst.OUTPUT_ARGS, "output1", "data1"),
            call("test_node", DumpConst.OUTPUT_ARGS, "output2", "data2"),
        ]
        mock_update_stat.assert_has_calls(calls)
        mock_info.assert_called_once_with("net_output node index is: 0, node name: output2.")
        writer.task = CfgConst.TASK_TENSOR
        writer.through_outputs(outputs, "test_node", output_map)
        mock_save_tensor.assert_has_calls(
            [call("test_node", DumpConst.OUTPUT, 0, "data1"), call("test_node", DumpConst.OUTPUT, 1, "data2")]
        )

    @patch.object(WriterDump, "_save_dump_json")
    @patch.object(WriterDump, "_save_stack_json")
    def test_flush_remaining_cache(self, mock_save_stack, mock_save_dump):
        writer = self.writer_cls(self.task)
        writer.cache_dump_json_size = 500
        writer.cache_stack_json_size = 500
        writer._flush_remaining_cache()
        mock_save_dump.assert_called_once()
        mock_save_stack.assert_called_once()

    @patch.object(DirPool, "get_rank_dir", return_value="/mock/rank")
    @patch("msit.module.probe.base.dump_writer.save_json")
    def test_save_stack_json(self, mock_save, mock_dir):
        test_data = {"node1": ["stack_info"]}
        writer = self.writer_cls(self.task)
        writer.cache_stack_json = test_data
        writer._save_stack_json()
        mock_save.assert_called_once_with(test_data, "/mock/rank/stack.json", indent=4)

    @patch("msit.module.probe.base.dump_writer.save_json")
    def test_save_dump_json(self, mock_save):
        writer = self.writer_cls(self.task)
        writer.cache_dump_json = {"data": "test"}
        writer._save_dump_json()
        mock_save.assert_called_once_with({"data": "test"}, "/fake/rank_dir/dump.json", indent=4)
        self.mock_get_rank_dir.assert_called_once()

    @patch("msit.module.probe.base.dump_writer.save_npy")
    @patch("msit.module.probe.base.dump_writer.MsitPath.check")
    def test_save_tensor_data(self, mock_msitpath, mock_save_npy):
        writer = self.writer_cls(self.task)
        mock_msitpath.return_value = "/fake/tensor/path.npy"
        self.mock_dirpool.get_tensor_dir.return_value = "/fake/tensor_dir"
        writer._save_tensor_data("node1", "input", 0, "tensor_data")
        mock_save_npy.assert_called_once_with("tensor_data", "/fake/tensor/path.npy")

    @patch.object(WriterDump, "_flush_remaining_cache")
    @patch("msit.module.probe.base.dump_writer.save_json")
    def test_summ_dump_data_decorator(self, mock_save, mock_flush):
        writer = self.writer_cls(self.task)
        writer.net_output_nodes = ["output1"]
        self.mock_dirpool.get_model_dir.return_value = "/fake/model_dir"
        result = writer.summ_dump_data()
        self.assertEqual(result, "test_result")
        mock_flush.assert_called_once()
        mock_save.assert_called_once_with(["output1"], "/fake/model_dir/net_output_nodes.json")
