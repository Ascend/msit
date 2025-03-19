# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import atexit
import csv
import json
import os
import shlex
import subprocess
import shutil
import time
from copy import deepcopy
from typing import List, Tuple, Optional
from pathlib import Path
from math import exp, inf

import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps
from pyswarms.base.base_single import SwarmOptimizer
from loguru import logger
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface

from modelevalstate.inference.constant import IS_SLEEP_FLAG
from modelevalstate.optimizer.config import default_support_field, PsoOptions, PerformanceIndex, OptimizerConfigField
from modelevalstate.optimizer.config import DataStorageConfig, BenchMarkConfig, MindieConfig, settings, RUN_TIME, \
    BenchMarkPolicy
from modelevalstate.optimizer.analyze import analyze


def kill_children(children):
    for child in children:
        if not child.is_running():
            continue
        child.send_signal(9)
        child.wait(10)
        if child.is_running():
            logger.error(f"Failed to kill the {child.pid} process.")


def kill_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if not hasattr(proc, "info"):
            continue
        if process_name not in proc.info["name"]:
            continue
        children = psutil.Process(proc.pid).children(recursive=True)
        if proc.is_running():
            proc.send_signal(9)
            proc.wait(10)
            if proc.is_running():
                logger.error(f"Failed to kill the {proc.pid} process.")
        kill_children(children)


def remove_file(output_path: Path):
    if output_path.exists():
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


class DataStorage:
    def __init__(self, config: DataStorageConfig):
        self.config = config
        if not self.config.store_dir.exists():
            self.config.store_dir.mkdir(parents=True)
        self.save_file = self.config.store_dir.joinpath(f"data_storage_{RUN_TIME}.csv")

    @staticmethod
    def load_history_position(load_dir: Path) -> Optional[List]:
        if not load_dir.exists():
            raise FileNotFoundError(f"file: {load_dir}")
        if not load_dir.is_dir():
            raise ValueError(f"Expect a directory, not a file.")
        history_data = []
        for file in load_dir.iterdir():
            if not file.is_file():
                continue
            if file.name.startswith("data_storage") and file.suffix == ".csv":
                data = pd.read_csv(file).to_dict(orient="records")
                history_data.extend(data)
        if not history_data:
            return None
        return history_data

    def save(self, performance_index: PerformanceIndex, params: Tuple[OptimizerConfigField],
             bench_mark_config: BenchMarkConfig):
        logger.info("Save result with DataStorage.")
        _column = []
        _value = []
        for k, v in performance_index.model_dump().items():
            _column.append(k)
            _value.append(v)
        for _p in params:
            _column.append(_p.name)
            _value.append(_p.value)
        benchmark_param = shlex.split(bench_mark_config.command)[2:]
        for i in range(0, len(benchmark_param), 2):
            if (i + 1) < len(benchmark_param):
                _column.append(benchmark_param[i].strip("--"))
                _value.append(benchmark_param[i + 1])
            else:
                logger.warning(f"IndexError. index: {i + 1}, list: {benchmark_param}")
        if self.save_file.exists():
            with open(self.save_file, "a+") as f:
                data_writer = csv.writer(f)
                data_writer.writerow(_value)
        else:
            with open(self.save_file, "w") as f:
                data_writer = csv.writer(f)
                data_writer.writerow(_column)
                data_writer.writerow(_value)


class BenchMark:
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common"):
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.is_sleep_flag = True

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

    def run(self):
        self.prepare()
        # 启动测试
        logger.info("Start the benchmark test.")
        run_cmd = shlex.split(self.benchmark_config.command)
        run_cmd.extend(["--SavePath", self.benchmark_config.output_path])
        if os.environ:
            os.environ[IS_SLEEP_FLAG] = str(self.is_sleep_flag)
            custom_env = os.environ
        else:
            custom_env = {IS_SLEEP_FLAG: str(self.is_sleep_flag)}
        if self.benchmark_config.source_env:
            source_cmd = shlex.split(self.benchmark_config.source_env)
            process = subprocess.run([*source_cmd, ";", *run_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=custom_env, text=True)
        else:
            logger.info(f"command: {' '.join(run_cmd)}")
            process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=custom_env,
                                       text=True)
        while True:
            errs = process.stderr.readline()
            logger.info(f"{errs}")
            if "common metric result" in errs:
                logger.info("wait...")
                process.wait()
                break
            if process.poll() is not None:
                logger.info(f"Process running complete. return code {process.returncode}")
                break

        # 获取测试结果
        performance_index = self.get_performance_index()
        logger.info("Finished the benchmark test.")
        return performance_index


class CustomBenchMark(BenchMark):
    def __init__(self, benchmark_config: BenchMarkConfig, **kwargs):
        super().__init__(benchmark_config=benchmark_config, **kwargs)
        self.is_sleep_flag = False

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
                throughput, first_prefill_latency, decode_latency = analyze(
                    self.benchmark_config.custom_analysis_output_path, 
                    file
                )
            except Exception as e:
                logger.error(f"Failed in analyze. {e}")
                continue
            if not throughput or not first_prefill_latency:
                raise ValueError(
                    f"Not found value throughput: {throughput} or first_prefill_latency{first_prefill_latency}. ")
            return PerformanceIndex(average_decode_throughput=throughput, average_prefill_latency=first_prefill_latency,
                                    average_decode_latency=decode_latency)


class Simulator:
    def __init__(self, mindie_config: MindieConfig):
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
        time.sleep(10)

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'start run in simulator. run params: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        self.check_env()
        cur_dir = os.getcwd()
        if self.mindie_config.work_path:
            os.chdir(self.mindie_config.work_path)
        if self.mindie_config.source_env:
            cmd = f"{self.mindie_config.source_env};{self.mindie_config.command}"
            run_cmd = shlex.split(cmd)
            logger.info(f"run_cmd {run_cmd}")
            self.process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=os.environ, text=True)
        else:
            self.process = subprocess.Popen(shlex.split(self.mindie_config.command), stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=os.environ, text=True)
        while True:
            time.sleep(1)
            output = self.process.stdout.readline()
            logger.info(f"simulate stdout: {output}")
            if "Daemon start success!" in output:
                break
            if self.process.poll() is not None:
                logger.info(f"stderr:  {self.process.stderr.read()}")
                raise subprocess.SubprocessError(
                    f"Failed in run mindie. return code: {self.process.returncode}. "
                    f"Please check the service log or console output.")
        os.chdir(cur_dir)
        logger.info(f"Successfully started the {self.process.pid} process.")

    def stop(self):
        logger.info("Stop mindie simulator process")
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
    def __init__(self, simulator: Simulator, benchmark: BenchMark, data_storage: DataStorage):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage

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
        _simulate_run_info = []
        for i, v in enumerate(params_field):
            _field = deepcopy(v)
            _field.value = v.format_func(params[i])
            _simulate_run_info.append(_field)
        self.simulator.run(tuple(_simulate_run_info))
        performance_index = self.benchmark.run()
        self.data_storage.save(performance_index, tuple(_simulate_run_info), self.benchmark.benchmark_config)
        self.simulator.stop()

        return performance_index


class PSOOptimizer:
    def __init__(self, scheduler: Scheduler, n_particles: int = 10, iters=100, pso_options: PsoOptions = None,
                 target_field: Tuple = default_support_field, prefill_lam: float = 0.5, decode_lam: float = 0.5,
                 prefill_constrain: float = 0.05, decode_constrain: float = 0.05,
                 load_history_data: Optional[List] = None):
        self.scheduler = scheduler
        self.n_particles = n_particles
        self.iters = iters
        self.target_field = target_field
        if not pso_options:
            self.pso_options = PsoOptions()
        else:
            self.pso_options = pso_options
        self.prefill_lam = prefill_lam  # 优化算法中惩罚系数
        self.decode_lam = decode_lam
        self.prefill_constrain = prefill_constrain
        self.decode_constrain = decode_constrain
        self.load_history_data = load_history_data
        self.init_pos = None
        if self.load_history_data:
            self.init_pos = self.custom_init_pos()

    def custom_init_pos(self) -> Optional[np.ndarray]:
        best_init_pos = None
        _fitness = inf
        for case_data in self.load_history_data:
            try:
                performance_index = PerformanceIndex(average_decode_throughput=case_data["average_decode_throughput"],
                                                     average_prefill_latency=case_data["average_prefill_latency"],
                                                     average_decode_latency=case_data["average_decode_latency"])
                _new_fitness = self.minimum_algorithm(performance_index)
                if _new_fitness >= _fitness:
                    continue
                best_init_pos = [case_data.get(_field.name) for _field in self.target_field]
                _fitness = _new_fitness
            except KeyError:
                continue
        if best_init_pos:
            logger.info(f"history best init pos: {best_init_pos}, fitness: {_fitness}")
            lb, ub = [i * (1 - settings.float_range_in_best_particle) for i in best_init_pos], [
                i * (1 + settings.float_range_in_best_particle) for i in best_init_pos]
            min_bounds = np.repeat(
                np.array(lb)[np.newaxis, :], self.n_particles, axis=0
            )
            max_bounds = np.repeat(
                np.array(ub)[np.newaxis, :], self.n_particles, axis=0
            )
            pos = np.random.uniform(
                low=min_bounds, high=max_bounds, size=(self.n_particles, len(self.target_field))
            )
            return pos
        return best_init_pos

    def minimum_algorithm(self, performance_index: PerformanceIndex):
        _var = max(0.0, (performance_index.average_prefill_latency - self.prefill_constrain) / self.prefill_constrain)
        _decode_var = max(0.0,
                          (performance_index.average_decode_latency - self.decode_constrain) / self.decode_constrain)
        try:
            fitness = 1 / performance_index.average_decode_throughput + self.prefill_lam * (
                        exp(_var) - 1) + self.decode_lam * (exp(_decode_var) - 1)
        except OverflowError:
            fitness = inf
        logger.info(f"fitness {fitness}")
        return fitness

    def op_func(self, x):
        n_particles = x.shape[0]
        logger.info(f"Acquired n_particles: {n_particles}, value: {x}")
        throughput = []
        for i in range(n_particles):
            # 调用schedule， 获取采集的数据
            _res = self.scheduler.run(x[i], self.target_field)
            # 根据采集的数据，计算最优化值
            _fitness = self.minimum_algorithm(_res)
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

    @staticmethod
    def visualization(self, optimizer: SwarmOptimizer):
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

    def run(self):
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.target_field),
                                            options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                            init_pos=self.init_pos)
        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
        logger.info(
            f"best cost {cost}, best joint_vars: "
            f"{[self.target_field[i].format_func(k) for i, k in enumerate(joint_vars)]}"
        )
        self.visualization(optimizer)


def main(benchmark_policy: str, load_history: bool = False):
    simulator = Simulator(settings.mindie)
    if benchmark_policy == BenchMarkPolicy.benchmark:
        benchmark = BenchMark(settings.benchmark)
    else:
        benchmark = CustomBenchMark(settings.benchmark)
    data_storage = DataStorage(settings.data_storage)
    scheduler = Scheduler(simulator, benchmark, data_storage)
    _load_history_data = None
    if load_history:
        _load_history_data = data_storage.load_history_position(settings.data_storage.store_dir)
    pso = PSOOptimizer(
                        scheduler, 
                        n_particles=settings.n_particles, 
                        iters=settings.iters, 
                        prefill_lam=settings.prefill_lam,
                        decode_lam=settings.decode_lam, 
                        decode_constrain=settings.decode_constrain,
                        prefill_constrain=settings.prefill_constrain, 
                        load_history_data=_load_history_data
                      )
    pso.run()


parser = argparse.ArgumentParser(prog='optimizer')
parser.add_argument("benchmark_policy", default=BenchMarkPolicy.benchmark,
                    help="Whether to use custom performance indicators or mindie performance indicators. "
                         "Benchmark and custom_benchmark are supported.")
parser.add_argument("-lh", "--load_history", default=False, action="store_true",
                    help="Indicates whether to customize the initial location based on historical records.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.benchmark_policy, args.load_history)
