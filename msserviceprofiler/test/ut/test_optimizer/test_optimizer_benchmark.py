# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField, settings, \
    default_support_field, PerformanceIndex
from msserviceprofiler.modelevalstate.optimizer.optimizer import BenchMark, Simulator
from msserviceprofiler.modelevalstate.optimizer.optimizer import PSOOptimizer


class TestBenchMark:
    @pytest.fixture
    @patch("msserviceprofiler.modelevalstate.config.custom_command.shutil.which")
    def benchmark(self, mock_which):
        mock_which.return_value = 'benchmark'
        benchmark = BenchMark(MagicMock())
        benchmark.throughput_type = "common"
        benchmark.benchmark_config.output_path = Path("./result")
        return benchmark

    @patch("pathlib.Path.iterdir")
    @patch("msserviceprofiler.modelevalstate.config.custom_command.shutil.which")
    def test_get_performance_index_no_result_common(self, mock_iterdir, mock_which, benchmark):
        mock_iterdir.return_value = []
        mock_which.return_value = 'benchmark'
        with pytest.raises(AttributeError):
            benchmark.get_performance_index()

    @patch("pathlib.Path.iterdir")
    def test_get_performance_index_no_result_perf(self, mock_iterdir, benchmark):
        mock_file_common = MagicMock()
        mock_file_common.name = "result_common_20250517031241120.csv"
        mock_iterdir.return_value = [mock_file_common, ]
        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1995.7208 token/s"],
            "Returned": ["3000( 100.0% )"],
        })
        with patch("pandas.read_csv", side_effect=[df_common, ]):
            with pytest.raises(ValueError, match="Not Found first_token_time."):
                benchmark.get_performance_index()

    @patch("pathlib.Path.iterdir")
    def test_get_performance_index_success(self, mock_iterdir, benchmark):
        mock_file_common = MagicMock()
        mock_file_common.name = "result_common_20250517031241120.csv"
        mock_file_perf = MagicMock()
        mock_file_perf.name = "result_perf_20250517041302288.csv"
        mock_iterdir.return_value = [mock_file_common, mock_file_perf]

        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1995.7208 token/s"],
            "Returned": ["3000( 100.0% )"],
        })
        df_perf = pd.DataFrame({
            "FirstTokenTime": ["572.8072 ms"],
            "GeneratedTokenSpeed": ["7.5371 token/s"],
            "DecodeTime": ["127.9866 ms"],
        })

        with patch("pandas.read_csv", side_effect=[df_common, df_perf]):
            result = benchmark.get_performance_index()
            assert isinstance(result, PerformanceIndex)
            assert result.generate_speed == 1995.7208
            assert result.time_to_first_token == 0.5728072
            assert result.time_per_output_token == 0.1279866
            assert result.success_rate == 1

    @patch("pathlib.Path.iterdir")
    def test_get_performance_index_failure(self, mock_iterdir, benchmark):
        mock_file_common = MagicMock(name="result_common")
        mock_file_common.name = "result_common"
        mock_file_perf = MagicMock(name="result_perf")
        mock_file_perf.name = "result_perf"
        mock_iterdir.return_value = [mock_file_common, mock_file_perf]

        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1995.7208 token/s"],
            "Returned": ["3000( 56.0% )"],
        })
        df_perf = pd.DataFrame({
            "FirstTokenTime": ["572.8072 ms"],
            "GeneratedTokenSpeed": ["7.5371 token/s"],
            "DecodeTime": ["127.9866 ms"],
        })
        with patch("pandas.read_csv", side_effect=[df_common, df_perf]):
            result = benchmark.get_performance_index()
            assert isinstance(result, PerformanceIndex)
            assert result.generate_speed == 1995.7208
            assert result.time_to_first_token == 0.5728072
            assert result.time_per_output_token == 0.1279866
            assert result.success_rate == 0.56


class TestOptimizerBenchmark:
    @staticmethod
    def test_run():
        # 创建模拟对象
        mock_benchmark_config = MagicMock()
        mock_benchmark_config.work_path = os.getcwd()
        mock_benchmark_config.command = 'ls'

        mock_run_params = (
            OptimizerConfigField(name='env_var1', value=2.3, config_position='env'),
            OptimizerConfigField(name='env_var2', value=3, config_position='env'),
            *default_support_field
        )

        # 创建BenchMark实例并调用run方法
        benchmark = BenchMark(settings.benchmark)
        benchmark.benchmark_config = mock_benchmark_config
        benchmark.prepare = MagicMock()
        benchmark.run(mock_run_params)


        # 验证os.environ被正确设置
        assert os.environ['env_var1'] == '2.3'
        assert os.environ['env_var2'] == '3'
        time.sleep(3)
        assert Path(benchmark.run_log).resolve().stat().st_size > 0
        benchmark.stop()


class TestSimulate:
    @staticmethod
    def test_set_config_dict():
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    @staticmethod
    def test_set_config_list():
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    @staticmethod
    def test_set_config_new_key():
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4


def test_computer_fitness():
    # 创建一个PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field[:1])
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # 模拟load_history_data方法的行为
    optimizer.load_history_data = [
        {'generate_speed': 1, 'time_to_first_token': 2, 'time_per_output_token': 3, 'success_rate': 1.0,
         "max_batch_size": 3},
        {'generate_speed': 4, 'time_to_first_token': 5, 'time_per_output_token': 6, 'success_rate': 1.0,
         "max_batch_size": 4},
        {'generate_speed': 7, 'time_to_first_token': 8, 'time_per_output_token': 9, 'success_rate': 1.0,
         "max_batch_size": 5},
    ]
    # 调用computer_fitness方法
    positions, costs = optimizer.computer_fitness()

    # 检查结果
    assert np.array(positions).size == 3
    assert costs == [1.0, 1.0, 1.0]


def test_computer_fitness_with_key_error():
    # 创建一个PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # 模拟load_history_data方法的行为
    optimizer.load_history_data = [
        {'generate_speed': 1, 'time_to_first_token': 2, 'time_per_output_token': 3, 'success_rate': 1.0,
         "max_batch_size": 3},
        {'generate_speed': 4, 'time_to_first_token': 5, 'time_per_output_token': 6, 'success_rate': 1.0,
         "max_batch_size": 4},
        {'generate_speed': 7, 'time_to_first_token': 8, 'time_per_output_token': 9, 'success_rate': 1.0,
         "max_batch_size": 5},
    ]

    # 调用computer_fitness方法
    positions, costs = optimizer.computer_fitness()

    # 检查结果，缺少字段的数据应该被忽略
    assert len(positions) == 0
    assert len(costs) == 0


# 测试数据
TEST_PARAMS = np.array([[1.0, 2.0], [3.0, 4.0]])
TEST_PARAMS_FIELD = ('field1', 'field2')


# 测试用例
def test_op_func_success():
    # 创建PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    # 模拟scheduler.run方法
    optimizer.scheduler.run = MagicMock(return_value=PerformanceIndex(generate_speed=100,
                                                                      time_to_first_token=0.1,
                                                                      time_per_output_token=0.1,
                                                                      success_rate=1))

    # 调用op_func方法
    result = optimizer.op_func(TEST_PARAMS)

    # 验证scheduler.run和minimum_algorithm方法被正确调用
    optimizer.scheduler.run.assert_called()
    # 验证返回值
    assert result.size == 2


def test_op_func_exception():
    # 创建PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)

    # 模拟scheduler.run方法抛出异常
    optimizer.scheduler = MagicMock()
    optimizer.scheduler.run = MagicMock(side_effect=Exception("Test Exception"))

    # 模拟minimum_algorithm方法

    # 调用op_func方法
    result = optimizer.op_func(TEST_PARAMS)

    # 验证scheduler.run和minimum_algorithm方法被正确调用
    optimizer.scheduler.run.assert_called()

    # 验证返回值
    assert np.array_equal(result, np.array([float('inf'), float('inf')]))


class MockField:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value


class TestPSOOptimizer:
    @staticmethod
    def test_constructing_bounds_empty_target_field(optimizer):
        optimizer.target_field = []
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == ()
        assert max_bounds == ()

    @staticmethod
    def test_constructing_bounds_single_target_field(optimizer):
        optimizer.target_field = [MockField(min_value=0, max_value=10)]
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == (0,)
        assert max_bounds == (10,)

    @staticmethod
    def test_constructing_bounds_multiple_target_fields(optimizer):
        optimizer.target_field = [MockField(min_value=0, max_value=10), MockField(min_value=20, max_value=30)]
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == (0, 20)
        assert max_bounds == (10, 30)

    @pytest.fixture
    def optimizer(self):
        return PSOOptimizer(MagicMock(), target_field=default_support_field)