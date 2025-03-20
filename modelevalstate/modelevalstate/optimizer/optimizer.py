# -*- coding: utf-8 -*-
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
import argparse
import atexit
import json
import os
import shlex
import subprocess
import shutil
import time
import tempfile

import xmlrpc.client
from typing import List, Tuple, Optional
from pathlib import Path
from math import exp, inf
from xmlrpc.client import ServerProxy
import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyswarms.base.base_single import SwarmOptimizer
from loguru import logger
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface

from modelevalstate.inference.constant import IS_SLEEP_FLAG
from modelevalstate.common import get_train_sub_path
from modelevalstate.optimizer.config import default_support_field, PsoOptions, PerformanceIndex, OptimizerConfigField
from modelevalstate.optimizer.config import AnalyzeTool, BenchMarkConfig, MindieConfig, settings, RUN_TIME, \
    BenchMarkPolicy, DeployPolicy, map_param_with_value
from modelevalstate.optimizer.analyze import analyze
from modelevalstate.optimizer.analyze_deepseek import analyze as analyze_deepseek
from modelevalstate.optimizer.store import DataStorage
from modelevalstate.optimizer.global_best_custom import CustomGlobalBestPSO

_analyze_mapping = {
    AnalyzeTool.default.value: analyze,
    AnalyzeTool.deepseek.value: analyze_deepseek
}


def kill_children(children):
    for child in children:
        if not child.is_running():
            continue
        try:
            child.send_signal(9)
            child.wait(10)
        except Exception as e:
            logger.error(f"Failed in kill the {child.pid} process. detail: {e}")
            continue
        if child.is_running():
            logger.error(f"Failed to kill the {child.pid} process.")


def kill_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if not hasattr(proc, "info"):
            continue
        if process_name not in proc.info["name"]:
            continue
        children = psutil.Process(proc.pid).children(recursive=True)
        kill_children([proc])
        kill_children(children)


def remove_file(output_path: Path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        return
    if output_path.is_file():
        output_path.unlink()
        return
    for file in output_path.iterdir():
        if file.is_file():
            file.unlink()
        else:
            try:
                shutil.rmtree(file)
            except OSError:
                remove_file(file)


@atexit.register
def clearing_residual_process():
    kill_process(MindieConfig().process_name)


class BenchMark:
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common",
                 bak_path: Optional[Path] = None):
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.is_sleep_flag = True
        self.bak_path = bak_path

    def backup(self):
        shutil.copytree(Path(self.benchmark_config.output_path),
                        self.bak_path.joinpath(self.__class__.__name__).joinpath(
                            Path(self.benchmark_config.output_path).name))

    def get_performance_index(self):
        for file in Path(self.benchmark_config.output_path).iterdir():
            if "result_common" in file.name:
                df = pd.read_csv(file)
                try:
                    common_generate_speed = float(df["GenerateSpeed"][0].split()[0])
                except AttributeError:
                    logger.error(f"Failed in get GenerateSpeed. value: {df}")
                continue
            if "result_perf" in file.name:
                df = pd.read_csv(file)
                try:
                    first_token_time = float(df["FirstTokenTime"][0].split()[0])
                    perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
                    decode_time = float(df["DecodeTime"][0].split()[0])
                except AttributeError:
                    logger.error(f"Failed in get FirstTokenTime or GeneratedTokenSpeed . value: {df}")
        if "common_generate_speed" not in locals() and "perf_generate_token_speed" not in locals():
            raise ValueError("Not Found common_generate_speed or perf_generate_token_speed.")
        if "first_token_time" not in locals() or "decode_time" not in locals():
            raise ValueError("Not Found first_token_time.")
        if self.throughput_type == "common":
            average_decode_throughput = common_generate_speed
        else:
            average_decode_throughput = perf_generate_token_speed
        average_prefill_latency = first_token_time / 10 ** 6
        average_decode_latency = decode_time / 10 ** 6
        return PerformanceIndex(average_decode_throughput=average_decode_throughput,
                                average_prefill_latency=average_prefill_latency,
                                average_decode_latency=average_decode_latency)

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))

    def run(self, run_params: Tuple[OptimizerConfigField]):
        self.prepare()
        # 启动测试
        logger.info("Start the benchmark test.")
        for k in run_params:
            if k.config_position == "env":
                os.environ[k.name] = str(k.value)
        run_cmd = shlex.split(self.benchmark_config.command)
        if os.environ:
            os.environ[IS_SLEEP_FLAG] = str(self.is_sleep_flag)
            custom_env = os.environ
        else:
            custom_env = {IS_SLEEP_FLAG: str(self.is_sleep_flag)}
        logger.info(f"command: {' '.join(run_cmd)}")
        process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=custom_env,
                                   text=True)
        while True:
            errs = process.stderr.readline()
            logger.info(f"{errs}")
            if process.poll() is not None:
                logger.info(f"Process running complete. return code {process.returncode}")
                break
        time.sleep(1)
        # 获取测试结果
        performance_index = self.get_performance_index()
        logger.info("Finished the benchmark test.")
        if self.bak_path:
            self.backup()
        return performance_index


class CustomBenchMark(BenchMark):
    def __init__(self, benchmark_config: BenchMarkConfig, analyze_tool: AnalyzeTool = AnalyzeTool.default, **kwargs):
        super().__init__(benchmark_config=benchmark_config, **kwargs)
        self.is_sleep_flag = False
        self.analyze_tool = analyze_tool

    def extra_performance_index(self, *args, **kwargs):
        res = _analyze_mapping.get(self.analyze_tool)(*args, **kwargs)
        first_prefill_latency = decode_latency = success_rate = None
        if len(res) == 1:
            throughput = res
        elif len(res) == 2:
            throughput, first_prefill_latency = res
        elif len(res) == 3:
            throughput, first_prefill_latency, decode_latency = res
        elif len(res) == 4:
            throughput, first_prefill_latency, decode_latency, success_rate = res
        else:
            raise ValueError(f"Not Support. res: {res}")
        return PerformanceIndex(average_decode_throughput=throughput, average_prefill_latency=first_prefill_latency,
                                average_decode_latency=decode_latency, success_rate=success_rate)

    def backup(self):
        super().backup()
        shutil.copytree(Path(self.benchmark_config.custom_collect_output_path),
                        self.bak_path.joinpath(self.__class__.__name__).joinpath(
                            Path(self.benchmark_config.custom_collect_output_path).name))
        shutil.copytree(Path(self.benchmark_config.custom_analysis_output_path),
                        self.bak_path.joinpath(self.__class__.__name__).joinpath(
                            Path(self.benchmark_config.custom_analysis_output_path).name))

    def prepare(self):
        super().prepare()
        remove_file(Path(self.benchmark_config.custom_collect_output_path))
        remove_file(Path(self.benchmark_config.custom_analysis_output_path))

    def get_performance_index(self):
        logger.info("get_performance_index")
        collect_path = Path(self.benchmark_config.custom_collect_output_path)
        for file in collect_path.iterdir():
            if not file.is_dir():
                continue
            try:
                res = self.extra_performance_index(self.benchmark_config.custom_analysis_output_path, file,
                                                   self.benchmark_config.output_path)
            except Exception as e:
                logger.error(f"Failed in analyze. {e}")
                continue
            return res


class RPCCustomBenchMark(CustomBenchMark):
    def __init__(self, rpc_clients, benchmark_config: BenchMarkConfig, **kwargs):
        super().__init__(benchmark_config=benchmark_config, **kwargs)
        self.is_sleep_flag = False
        self.rpc_clients = rpc_clients

    def prepare(self):
        super().prepare()
        for rpc in self.rpc_clients:
            rpc.remove_file(self.benchmark_config.custom_collect_output_path)
            rpc.remove_file(self.benchmark_config.output_path)

    def sync_server_file(self):
        # 拷贝远程服务器文件到本机
        remote_file = []
        for rpc in self.rpc_clients:
            remote_file.extend(rpc.get_file(self.benchmark_config.custom_collect_output_path))
        for _file_name, file_binary in remote_file:
            _tmp_file = Path(_file_name)
            new_file = Path(self.benchmark_config.custom_collect_output_path).joinpath(
                f"{_tmp_file.stem}_rpc{_tmp_file.suffix}")
            with open(new_file, "wb") as f:
                f.write(file_binary.data)

    def get_performance_index(self):
        logger.info("get_performance_index")
        self.sync_server_file()
        collect_path = Path(self.benchmark_config.custom_collect_output_path)
        res = self.extra_performance_index(collect_path, self.benchmark_config.custom_analysis_output_path,
                                           self.benchmark_config.output_path)
        return res


class Simulator:
    def __init__(self, mindie_config: MindieConfig, bak_path: Optional[Path] = None):
        self.mindie_config = mindie_config
        logger.info(f"config path {self.mindie_config.config_path}", )
        if not self.mindie_config.config_path.exists():
            raise FileNotFoundError(self.mindie_config.config_path)
        with open(self.mindie_config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.info(f"config bak path {self.mindie_config.config_bak_path}", )
        if not self.mindie_config.config_bak_path.exists():
            with open(self.mindie_config.config_bak_path, "w") as f:
                json.dump(self.default_config, f, indent=4)
        self.mindie_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.process = None

    @staticmethod
    def get_new_config(origin_config, params: Tuple[OptimizerConfigField], upper_key: str = ""):
        if upper_key:
            _keys = [upper_key]
        else:
            _keys = []
        if isinstance(origin_config, dict):
            _new_config = {}
            for k, v in origin_config.items():
                _root_key = ".".join([*_keys, k])
                new_value = Simulator.get_new_config(v, params, _root_key)
                _new_config[k] = new_value
        elif isinstance(origin_config, list):
            _new_config = []
            for i, v in enumerate(origin_config):
                _root_key = ".".join([*_keys, str(i)])
                new_value = Simulator.get_new_config(v, params, f"{upper_key}.{i}")
                _new_config.append(new_value)
        else:
            _new_config = origin_config
            for _p in params:
                if upper_key == _p.config_position:
                    _new_config = _p.value
                    break
        return _new_config

    def backup(self):
        shutil.copy(self.mindie_config.config_path,
                    self.bak_path.joinpath(self.mindie_config.config_path.name))

    def update_config(self, params: Tuple[OptimizerConfigField]):
        # 将params值更新到新的config中
        new_config = self.get_new_config(self.default_config, params)
        # 将新的config写入到config文件中
        logger.info(f"new config {new_config}")
        with open(self.mindie_config.config_path, "w") as f:
            json.dump(new_config, f, indent=4)

    def check_env(self):
        remove_file(Path(self.mindie_config.log_path))
        _residual_process = []
        for proc in psutil.process_iter(["pid", "name"]):
            if not hasattr(proc, "info"):
                continue
            if self.mindie_config.process_name not in proc.info["name"]:
                continue
            _residual_process.append(proc)
        if _residual_process:
            logger.info("kill residual_process")
            kill_process(self.mindie_config.process_name)
        logger.info("check env")
        time.sleep(1)

    def check_success(self):
        with open(self.mindie_log, "r") as f:
            f.seek(self.mindie_log_offset)
            output = f.read()
            self.mindie_log_offset = f.tell()
        if output:
            logger.info(f"simulate out: \n{output}")
            if "Daemon start success!" in output:
                return True
        if self.process.poll() is not None:
            raise subprocess.SubprocessError(
                f"Failed in run mindie. return code: {self.process.returncode}. "
                f"Please check the service log or console output.")
        return False

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        cur_dir = os.getcwd()
        self.mindie_log = tempfile.NamedTemporaryFile(delete=False).name
        self.mindie_log_offset = 0
        if self.mindie_config.work_path:
            os.chdir(self.mindie_config.work_path)
        for k in run_params:
            if k.config_position == "env":
                os.environ[k.name] = str(k.value)
        logger.info(f"env {os.environ}")
        cmd = f"{self.mindie_config.command} {self.mindie_log}"
        run_cmd = shlex.split(cmd)
        logger.info(f"run cmd: {run_cmd}")

        self.process = subprocess.Popen(run_cmd, env=os.environ, text=True)
        os.chdir(cur_dir)

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'start run in simulator. run params: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        self.check_env()
        self.start_server(run_params)
        while True:
            time.sleep(1)
            if self.check_success():
                break
        logger.info(f"Successfully started the {self.process.pid} process.")

    def stop(self):
        logger.info("Stop mindie simulator process")
        if self.bak_path:
            self.backup()
        os.remove(self.mindie_log)
        self.mindie_log_offset = 0
        if self.process.poll() is not None:
            return
        children = psutil.Process(self.process.pid).children(recursive=True)
        self.process.kill()
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.send_signal(9)
        if self.process.poll() is not None:
            logger.info(f"The {self.process.pid} process has been shut down.")
        else:
            raise subprocess.SubprocessError(f"The {self.process.pid} process shutdown failed.")
        kill_children(children)
        kill_process(self.mindie_config.process_name)
        with open(self.mindie_config.config_path, "w") as f:
            json.dump(self.default_config, f)


class Scheduler:
    def __init__(self, simulator: Simulator, benchmark: BenchMark, data_storage: DataStorage,
                 bak_path: Optional[Path] = None):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path

    def back_up(self):
        if self.bak_path:
            _cur_bak_path = get_train_sub_path(self.bak_path)
            self.simulator.bak_path = _cur_bak_path
            self.benchmark.bak_path = _cur_bak_path

    def run(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        logger.info("Start run in scheduler.")
        self.back_up()
        _simulate_run_info = map_param_with_value(params, params_field)
        self.simulator.run(tuple(_simulate_run_info))
        performance_index = self.benchmark.run(tuple(_simulate_run_info))
        self.data_storage.save(performance_index, tuple(_simulate_run_info), self.benchmark.benchmark_config)
        self.simulator.stop()
        return performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, rpc_clients: List[ServerProxy], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpc_clients = rpc_clients

    def run(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        logger.info("Start run in scheduler.")
        self.back_up()
        _simulate_run_info = []
        _simulate_run_info = map_param_with_value(params, params_field)
        [rpc.run_simulator(params.tolist()) for rpc in self.rpc_clients]
        self.simulator.run(tuple(_simulate_run_info))
        [rpc.check_success() for rpc in self.rpc_clients]
        performance_index = self.benchmark.run(tuple(_simulate_run_info))
        self.data_storage.save(performance_index, tuple(_simulate_run_info), self.benchmark.benchmark_config)
        [rpc.stop_simulator() for rpc in self.rpc_clients]
        self.simulator.stop()
        return performance_index


class PSOOptimizer:
    def __init__(self, scheduler: Scheduler, n_particles: int = 10, iters=100, pso_options: PsoOptions = None,
                 target_field: Optional[Tuple] = None, prefill_lam: float = 0.5, decode_lam: float = 0.5,
                 success_rate_lam: float = 0.5, prefill_constrain: float = 0.05, decode_constrain: float = 0.05,
                 success_rate_constrain: float = 1, load_history_data: Optional[List] = None,
                 load_breakpoint: bool = False):
        self.scheduler = scheduler
        self.n_particles = n_particles
        self.iters = iters
        self.target_field = target_field if target_field else default_support_field
        if not pso_options:
            self.pso_options = PsoOptions()
        else:
            self.pso_options = pso_options
        self.prefill_lam = prefill_lam  # 优化算法中惩罚系数
        self.decode_lam = decode_lam
        self.success_rate_lam = success_rate_lam
        self.prefill_constrain = prefill_constrain
        self.decode_constrain = decode_constrain
        self.success_rate_constrain = success_rate_constrain
        self.load_history_data = load_history_data
        self.load_breakpoint = load_breakpoint
        self.init_pos = None
        self.history_cost, self.history_pos = None, None
        if self.load_history_data and self.load_breakpoint:
            self.history_pos, self.history_cost = self.computer_fitness()
        elif self.load_history_data:
            self.init_pos = self.custom_init_pos()

    @classmethod
    def visualization(cls, optimizer: SwarmOptimizer):
        plot_cost_history(cost_history=optimizer.cost_history)
        plt.savefig(settings.output.joinpath(f"cost_history_{RUN_TIME}.png"))
        plt.close()
        m = Mesher(func=fx.sphere)
        animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
        animation.save(settings.output.joinpath(f"pos_history_{RUN_TIME}.gif"), writer="pillow")
        pos_history_3d = m.compute_history_3d(optimizer.pos_history)
        d = Designer(limits=[(-1, 1), (-1, 1), (-0.1, 1)], label=['x-axis', 'y-axis', 'z-axis'])
        animation_3d = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0, 0, 0))
        animation_3d.save(str(settings.output.joinpath(f"pos_history_3d_{RUN_TIME}.gif").resolve()),
                          writer="pillow", )
        
        
    def custom_init_pos(self) -> Optional[np.ndarray]:
        _all_pos, _all_cost = self.computer_fitness()
        if not _all_pos or not _all_cost:
            return None
        _fitness = min(_all_cost)
        best_init_pos = np.array(_all_pos[_all_cost.index(_fitness)])
        logger.info(f"history best init pos: {best_init_pos}, fitness: {_fitness}")
        lb, ub = [i * (1 - settings.float_range_in_best_particle) for i in best_init_pos], [
            i * (1 + settings.float_range_in_best_particle) for i in best_init_pos]
        min_bounds = np.full((self.n_particles, len(self.target_field)), lb)
        max_bounds = np.full((self.n_particles, len(self.target_field)), ub)
        pos = np.random.uniform(
            low=min_bounds, high=max_bounds, size=(self.n_particles, len(self.target_field))
        )
        return pos

    def computer_fitness(self) -> Tuple:
        all_position = []
        all_cost = []
        for case_data in self.load_history_data:
            throughput = prefill_latency = decode_latency = success_rate = None
            if "average_decode_throughput" in case_data:
                throughput = case_data["average_decode_throughput"]
            if "average_prefill_latency" in case_data:
                prefill_latency = case_data["average_prefill_latency"]
            if "average_decode_latency" in case_data:
                decode_latency = case_data["average_decode_latency"]
            if "success_rate" in case_data:
                success_rate = case_data["success_rate"]
            performance_index = PerformanceIndex(average_decode_throughput=throughput,
                                                 average_prefill_latency=prefill_latency,
                                                 average_decode_latency=decode_latency,
                                                 success_rate=success_rate)
            try:
                _fitness = self.minimum_algorithm(performance_index)
                logger.info(f"fitness {_fitness}")
                _pos = [case_data.get(_field.name) for _field in self.target_field]
                all_cost.append(_fitness)
                all_position.append(_pos)
            except KeyError:
                continue
        if len(all_position) != len(all_cost):
            raise ValueError("Failed in computer_fitness.")
        return all_position, all_cost

    def minimum_algorithm(self, performance_index: PerformanceIndex) -> float:
        try:
            fitness = 1 / performance_index.average_decode_throughput
        except OverflowError:
            return inf
        if performance_index.average_prefill_latency is not None:
            _var = max(0.0, (
                    performance_index.average_prefill_latency - self.prefill_constrain) / self.prefill_constrain)
            try:
                fitness += self.prefill_lam * (exp(_var) - 1)
            except OverflowError:
                return inf
        if performance_index.average_decode_latency is not None:
            _decode_var = max(0.0, (
                    performance_index.average_decode_latency - self.decode_constrain) / self.decode_constrain)
            try:
                fitness += self.decode_lam * (exp(_decode_var) - 1)
            except OverflowError:
                return inf
        if performance_index.success_rate:
            _success_var = max(0.0, (
                    performance_index.success_rate - self.success_rate_constrain) / self.success_rate_constrain)
            try:
                fitness += self.success_rate_lam * (exp(_success_var) - 1)
            except OverflowError:
                return inf
        return fitness

    def op_func(self, x) -> np.ndarray:
        n_particles = x.shape[0]
        logger.info(f"Acquired n_particles: {n_particles}, value: {x}")
        throughput = []
        for i in range(n_particles):
            # 调用schedule， 获取采集的数据
            _res = self.scheduler.run(x[i], self.target_field)
            # 根据采集的数据，计算最优化值
            _fitness = self.minimum_algorithm(_res)
            logger.info(f"fitness {_fitness}")
            throughput.append(_fitness)
        return np.array(throughput)

    def constructing_bounds(self) -> Tuple[Tuple, Tuple]:
        """
        返回示例：((0, 10), (0, 10))
        """
        _min = []
        _max = []
        for _field in self.target_field:
            _min.append(_field.min)
            _max.append(_field.max)
        return (tuple(_min), tuple(_max))


    def run(self):
        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.target_field),
                                        options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                        init_pos=self.init_pos, breakpoint_pos=self.history_pos,
                                        breakpoint_cost=self.history_cost)
        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
        logger.info(
            f"best cost {cost}, best joint_vars: \
                {[self.target_field[i].format_func(k) for i, k in enumerate(joint_vars)]}")
        self.visualization(optimizer)


def main(args: argparse.Namespace):
    simulator = Simulator(settings.mindie)
    bak_path = None
    if args.back_up:
        bak_path = settings.output.joinpath("bak")
        if not bak_path.exists():
            bak_path.mkdir(parents=True)
    rpc_clients = []
    if args.deploy_policy == DeployPolicy.multiple.value:
        for server_address in settings.server:
            rpc = xmlrpc.client.ServerProxy(server_address, allow_none=True)
            logger.info(f"{server_address} support method {rpc.system.listMethods()}")
            rpc_clients.append(rpc)
    if args.benchmark_policy == BenchMarkPolicy.benchmark.value and \
        args.deploy_policy == DeployPolicy.single.value:
        benchmark = BenchMark(settings.benchmark, bak_path=bak_path)
    elif args.benchmark_policy == BenchMarkPolicy.custom_benchmark.value \
        and args.deploy_policy == DeployPolicy.multiple.value:
        benchmark = RPCCustomBenchMark(rpc_clients, settings.benchmark, \
                                       bak_path=bak_path, analyze_tool=args.analyze_tool)
    else:
        benchmark = CustomBenchMark(settings.benchmark, bak_path=bak_path, analyze_tool=args.analyze_tool)
    data_storage = DataStorage(settings.data_storage)
    if args.deploy_policy == DeployPolicy.multiple.value:
        scheduler = ScheduleWithMultiMachine(rpc_clients, simulator, benchmark, data_storage, bak_path=bak_path)
    else:
        scheduler = Scheduler(simulator, benchmark, data_storage, bak_path=bak_path)
    _load_history_data = None
    if args.load_history:
        _load_history_data = data_storage.load_history_position(settings.data_storage.store_dir)
    pso = PSOOptimizer(scheduler, n_particles=settings.n_particles, iters=settings.iters,
                       prefill_lam=settings.prefill_lam, target_field=settings.target_field,
                       decode_lam=settings.decode_lam, success_rate_lam=settings.success_rate_lam,
                       decode_constrain=settings.decode_constrain, prefill_constrain=settings.prefill_constrain,
                       success_rate_constrain=settings.success_rate_constrain, load_history_data=_load_history_data,
                       load_breakpoint=args.load_breakpoint)
    pso.run()


parser = argparse.ArgumentParser(prog='optimizer')
parser.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.custom_benchmark.value,
                    choices=[k.value for k in list(BenchMarkPolicy)],
                    help="Whether to use custom performance indicators or mindie performance indicators. \
                        Benchmark and custom_benchmark are supported.")
parser.add_argument("-lh", "--load_history", default=False, action="store_true",
                    help="Indicates whether to customize the initial location based on historical records.")
parser.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                    help="Continue from where the last optimization was aborted.")
parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                    choices=[k.value for k in list(DeployPolicy)],
                    help="Indicates whether the multi-node running policy is used.")
parser.add_argument("--back_up", default=True, action="store_true",
                    help="Whether to back up data.")
parser.add_argument("-a", "--analyze_tool", default=AnalyzeTool.default.value,
                    choices=[k.value for k in list(AnalyzeTool)], help="Tool of data to be analyze")

if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)