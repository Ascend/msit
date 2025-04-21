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
import stat
from copy import deepcopy

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
from modelevalstate.optimizer.analyze_profiler import analyze as analyze_profiler
from modelevalstate.optimizer.store import DataStorage
from modelevalstate.optimizer.global_best_custom import CustomGlobalBestPSO

_analyze_mapping = {
    AnalyzeTool.default.value: analyze,
    AnalyzeTool.deepseek.value: analyze_deepseek,
    AnalyzeTool.profiler.value: analyze_profiler
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
    if not output_path:
        return
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


def backup(target, bak, class_name=""):
    if not target:
        return
    if not isinstance(target, Path):
        target = Path(target)
    if not isinstance(target, Path):
        bak = Path(bak)
    if not target.exists():
        return
    if not bak.exists():
        return
    new_file = bak.joinpath(class_name).joinpath(target.name)
    if target.is_file():
        new_file.parent.mkdir(parents=True, exist_ok=True)
        if not new_file.exists():
            shutil.copy(target, new_file)
    else:
        if new_file.exists():
            for child in new_file.iterdir():
                backup(child, new_file, class_name)
        else:
            shutil.copytree(target, new_file)


def close_file_fp(file_fp):
    if not file_fp:
        return
    try:
        # 检查file_fp是否是一个文件对象
        if hasattr(file_fp, 'close'):
            file_fp.close()
        else:
            # 如果file_fp是一个文件描述符，调用os.close()
            os.close(file_fp)
    except (AttributeError, OSError):
        return


@atexit.register
def clearing_residual_process():
    kill_process(MindieConfig().process_name)


class BenchMark:
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common",
                 bak_path: Optional[Path] = None):
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.bak_path = bak_path
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None

    def backup(self, del_log=True):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def get_performance_index(self):
        output_path = Path(self.benchmark_config.output_path)
        common_generate_speed = None
        first_token_time = None
        perf_generate_token_speed = None
        decode_time = None
        for file in output_path.iterdir():
            if "result_common" in file.name:
                try:
                    df = pd.read_csv(file)
                    common_generate_speed = float(df["GenerateSpeed"][0].split()[0])
                except (KeyError, AttributeError) as e:
                    logger.error(f"Failed in get GenerateSpeed. error: {e}")
                continue
            if "result_perf" in file.name:
                try:
                    df = pd.read_csv(file)
                    first_token_time = float(df["FirstTokenTime"][0].split()[0])
                    perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
                    decode_time = float(df["DecodeTime"][0].split()[0])
                except (AttributeError, KeyError):
                    logger.error(f"Failed in get FirstTokenTime or GeneratedTokenSpeed. error: {e}")
        if common_generate_speed is None and perf_generate_token_speed is None:
            raise ValueError("Not Found common_generate_speed or perf_generate_token_speed.")
        if first_token_time is None or decode_time is None:
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

    def check_success(self, print_log=False):
        if self.run_log:
            run_log_path = Path(self.run_log)
            if run_log_path.exists() and print_log:
                try:
                    with open(run_log_path, "r", encoding="utf-8") as f:
                        f.seek(self.run_log_offset)
                        output = f.read()
                        self.run_log_offset = f.tell()
                        logger.info(f"benchmark out: \n{output}")
                except (UnicodeError, OSError) as e:
                    logger.error(f"Failed read benchmark log. error {e}")
        try:
            if self.process.poll() is None:
                return False
            elif self.process.poll() == 0:
                return True
            else:
                raise subprocess.SubprocessError(
                    f"Failed in run benchmark. return code: {self.process.returncode}. ")
        except AttributeError as e:
            logger.error(f"Failed to check process status, error {e}")
            return False

    def run(self, run_params: Tuple[OptimizerConfigField]):
        # 启动测试
        logger.info("Start the benchmark test.")
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate")
        self.run_log_offset = 0
        if self.benchmark_config.work_path:
            cwd = self.benchmark_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.config_position == "env":
                try:
                    os.environ[k.name] = str(k.value)
                except KeyError as e:
                    logger.error(f"Failed to set environment variable. error {e}")
        run_cmd = shlex.split(self.benchmark_config.command)
        try:
            self.process = subprocess.Popen(run_cmd, env=os.environ, stdout=self.run_log_fp, stderr=subprocess.STDOUT,
                                            text=True, cwd=cwd)
        except OSError as e:
            logger.error(f"Failed to run benchmark. error {e}")
            raise e
        logger.info(f"command: {' '.join(run_cmd)}, log file: {self.run_log}")

    def stop(self, del_log=True):
        self.backup(del_log)
        close_file_fp(self.run_log_fp)
        try:
            if self.process and self.process.poll() is None:
                self.process.kill()
        except AttributeError as e:
            logger.error(f"Failed to kill process. error {e}")
        if del_log:
            remove_file(Path(self.run_log))


class CustomBenchMark(BenchMark):
    def __init__(self, benchmark_config: BenchMarkConfig, analyze_tool: AnalyzeTool = AnalyzeTool.default, **kwargs):
        super().__init__(benchmark_config=benchmark_config, **kwargs)
        self.analyze_tool = analyze_tool

    def extra_performance_index(self, *args, **kwargs):
        logger.info("extra_performance_index")
        analyze_tool = _analyze_mapping.get(self.analyze_tool)
        if analyze_tool is None:
            raise ValueError(f"Analyze tool not found: {self.analyze_tool}")
        res = analyze_tool(*args, **kwargs)
        first_prefill_latency = decode_latency = success_rate = None
        if isinstance(res, tuple):
            if len(res) == 1:
                throughput = res[0]
            elif len(res) == 2:
                throughput, first_prefill_latency = res
            elif len(res) == 3:
                throughput, first_prefill_latency, decode_latency = res
            elif len(res) == 4:
                throughput, first_prefill_latency, decode_latency, success_rate = res
            else:
                raise ValueError(f"Not Support. res: {res}")
        else:
            throughput = res
        return PerformanceIndex(average_decode_throughput=throughput, average_prefill_latency=first_prefill_latency,
                                average_decode_latency=decode_latency, success_rate=success_rate)

    def backup(self, del_log=True):
        super().backup(del_log)
        backup(self.benchmark_config.custom_collect_output_path, self.bak_path, self.__class__.__name__)
        backup(self.benchmark_config.custom_analysis_output_path, self.bak_path, self.__class__.__name__)

    def prepare(self):
        super().prepare()
        remove_file(Path(self.benchmark_config.custom_collect_output_path))
        remove_file(Path(self.benchmark_config.custom_analysis_output_path))

    def get_performance_index(self):
        logger.info("get_performance_index")
        collect_path = Path(self.benchmark_config.custom_collect_output_path)
        if not collect_path.exists():
            raise FileNotFoundError(f"Collect path not found: {collect_path}")
        if self.analyze_tool == AnalyzeTool.deepseek.value:
            res = self.extra_performance_index(collect_path, self.benchmark_config.custom_analysis_output_path,
                                               self.benchmark_config.output_path)
            return res
        if not collect_path.is_dir():
            raise NotADirectoryError(f"Collect path is not a directory: {collect_path}")
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


class ProfilerBenchmark(CustomBenchMark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler_cmd = ["python", "-m", "ms_service_profiler.parse",
                             f"--input-path={self.benchmark_config.profile_input_path}",
                             f"--output-path={self.benchmark_config.profile_output_path}"]
        self.profiler_log = None
        self.profiler_log_fp = None
        self.profiler_log_offset = 0
        self.profiler_process = None

    def backup(self, del_log=True):
        super().backup(del_log)
        backup(self.benchmark_config.profile_input_path, self.bak_path, self.__class__.__name__)
        backup(self.benchmark_config.profile_output_path, self.bak_path, self.__class__.__name__)
        if not del_log and self.profiler_log:
            backup(self.profiler_log, self.bak_path, self.__class__.__name__)

    def prepare(self):
        super().prepare()
        remove_file(Path(self.benchmark_config.profile_input_path))
        remove_file(Path(self.benchmark_config.profile_output_path))

    def check_profiler(self, print_log=False):
        if print_log:
            try:
                with open(self.profiler_log, "r") as f:
                    f.seek(self.profiler_log_offset)
                    output = f.read()
                    self.profiler_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read benchmark log. error {e}")
            if output:
                logger.info(f"benchmark out: \n{output}")
        if self.profiler_process.poll() is None:
            return False
        elif self.profiler_process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run benchmark. return code: {self.process.returncode}. ")

    def start_profiler(self):
        self.profiler_log_fp, self.profiler_log = tempfile.mkstemp(prefix="modelevalstate")
        self.profiler_log_offset = 0
        if not os.path.exists(self.benchmark_config.work_path):
            raise FileNotFoundError(f"Work path not found: {self.benchmark_config.work_path}")
        logger.info(f"command: {' '.join(self.profiler_cmd)}, log file: {self.profiler_log}")
        self.profiler_process = subprocess.Popen(self.profiler_cmd, env=os.environ, stdout=self.profiler_log_fp,
                                                 stderr=subprocess.STDOUT,
                                                 text=True, cwd=self.benchmark_config.work_path)

    def get_performance_index(self):
        logger.info("get_performance_index")
        try:
            self.start_profiler()
            logger.info("wait profiler")
            while True:
                if self.check_profiler(print_log=True):
                    break
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed in start profiler. relation log: {self.profiler_log}")
            raise e
        collect_path = Path(self.benchmark_config.custom_collect_output_path)
        if self.analyze_tool == AnalyzeTool.profiler.value:
            res = self.extra_performance_index(self.benchmark_config.profile_output_path, collect_path)
            return res
        else:
            return super().get_performance_index()

    def stop(self, del_log=True):
        super().stop(del_log)
        close_file_fp(self.profiler_log_fp)
        if del_log:
            remove_file(Path(self.profiler_log))
        try:
            if self.profiler_process and self.profiler_process.poll() is None:
                self.profiler_process.kill()
        except AttributeError as e:
            logger.error(f"Failed to kill process. error {e}")


class RPCCustomBenchMark(CustomBenchMark):
    def __init__(self, rpc_clients, benchmark_config: BenchMarkConfig, **kwargs):
        super().__init__(benchmark_config=benchmark_config, **kwargs)
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
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = stat.S_IWUSR | stat.S_IRUSR
            try:
                with os.fdopen(os.open(new_file, flags, modes), 'wb') as fout:
                    fout.write(file_binary.data)
            except FileExistsError:
                logger.error(f"File already exists: {new_file}")
            except PermissionError:
                logger.error(f"Permission denied: {new_file}")
            except AttributeError:
                logger.error(f"Invalid file binary: {file_binary}")

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
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.mindie_config.config_bak_path, flags, modes), 'w') as fout:
                json.dump(self.default_config, fout, indent=4)
        self.mindie_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.mindie_log_fp = None
        self.process = None

    @staticmethod
    def get_new_config(origin_config, params: Tuple[OptimizerConfigField], upper_key: str = "") -> Any:
        if upper_key:
            _keys = [upper_key]
        else:
            _keys = []
        if isinstance(origin_config, dict):
            _dict_config = {}
            for k, v in origin_config.items():
                _root_key = ".".join([*_keys, k])
                new_value = Simulator.get_new_config(v, params, _root_key)
                _dict_config[k] = new_value
            return _dict_config

        elif isinstance(origin_config, list):
            _list_config = []
            for i, v in enumerate(origin_config):
                _root_key = ".".join([*_keys, str(i)])
                new_value = Simulator.get_new_config(v, params, f"{upper_key}.{i}")
                _list_config.append(new_value)
            return _list_config
        else:
            for _p in params:
                if upper_key == _p.config_position:
                    logger.info(f"Update Config key: {upper_key}")
                    return _p.value
            return origin_config

    @staticmethod
    def set_config(origin_config, key: str, value: Any):
        next_level = None
        if "." in key:
            _f_index = key.index(".")
            _cur_key, next_level = key[:_f_index], key[_f_index + 1:]
        else:
            _cur_key = key
        if next_level:
            if isinstance(origin_config, dict):
                Simulator.set_config(origin_config[_cur_key], next_level, value)
            elif isinstance(origin_config, list):
                Simulator.set_config(origin_config[int(_cur_key)], next_level, value)
            else:
                raise ValueError(f"Not Support type {type(origin_config)}")
        else:
            origin_config[_cur_key] = value

    def backup(self, del_log=True):
        backup(self.mindie_config.config_path, self.bak_path, self.__class__.__name__)
        if not del_log and self.mindie_log:
            backup(self.mindie_log, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Tuple[OptimizerConfigField]):
        # 将params值更新到新的config中
        new_config = deepcopy(self.default_config)
        for p in params:
            if not p.config_position.startswith("BackendConfig"):
                continue
            Simulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        if self.mindie_config.config_path.exists():
            self.mindie_config.config_path.unlink()
        with os.fdopen(os.open(self.mindie_config.config_path, flags, modes), "w") as fout:
            json.dump(new_config, fout, indent=4)

    def check_env(self):
        logger.info("check env")
        _residual_process = []
        _all_process_name = self.mindie_config.process_name.split(",")
        for proc in psutil.process_iter(["pid", "name"]):
            if not hasattr(proc, "info"):
                continue
            _proc_flag = []
            for p in _all_process_name:
                if p not in proc.info["name"]:
                    _proc_flag.append(True)
                else:
                    _proc_flag.append(False)
            if all(_proc_flag):
                continue
            _residual_process.append(proc)
        if _residual_process:
            logger.info("kill residual_process")
            for _p_name in _all_process_name:
                try:
                    kill_process(_p_name)
                except Exception as e:
                    logger.error(f"Failed to kill process. {e}")
        time.sleep(1)

    def check_success(self, print_log=False):
        with open(self.mindie_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.info(f"Failed in read mindie log. error: {e}")
        if output:
            if print_log:
                logger.info(f"simulate out: \n{output}")
            if "Daemon start success!" in output:
                return True
        if self.process.poll() is not None:
            raise subprocess.SubprocessError(
                f"Failed in run mindie. return code: {self.process.returncode}. "
                f"Please check the service log or console output.")
        return False

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        self.mindie_log_fp, self.mindie_log = tempfile.mkstemp(prefix="modelevalstate")
        self.mindie_log_offset = 0
        if self.mindie_config.work_path:
            cwd = self.mindie_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.config_position == "env":
                os.environ[k.name] = str(k.value)
        logger.debug(f"env {os.environ}")
        run_cmd = shlex.split(self.mindie_config.command)
        logger.info(f"run cmd: {run_cmd}, log path: {self.mindie_log}")
        self.process = subprocess.Popen(run_cmd, stdout=self.mindie_log_fp, stderr=subprocess.STDOUT, env=os.environ,
                                        text=True, cwd=cwd)

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'start run in simulator. run params: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        try:
            self.check_env()
        except Exception as e:
            logger.error(f"Failed to check env. {e}")
        self.start_server(run_params)

    def stop(self, del_log=True):
        logger.info("Stop mindie simulator process")
        if self.bak_path:
            self.backup(del_log)
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.mindie_log)
        self.mindie_log_offset = 0
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"mindie already. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.info(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
            kill_process(self.mindie_config.process_name)
            remove_file(self.mindie_config.config_path)
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.mindie_config.config_path, flags, modes), "w") as fout:
                json.dump(self.default_config, fout)
        except Exception as e:
            logger.error(f"Failed to stop mindie simulator process. {e}")


class Scheduler:
    def __init__(self, simulator: Simulator, benchmark: BenchMark, data_storage: DataStorage,
                 bak_path: Optional[Path] = None, retry_number: int = 3):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.simulate_run_info = None

    def back_up(self):
        if self.bak_path:
            _cur_bak_path = get_train_sub_path(self.bak_path)
            self.simulator.bak_path = _cur_bak_path
            self.benchmark.bak_path = _cur_bak_path

    def wait_simulate(self):
        logger.info("wait run mindie")
        while True:
            time.sleep(1)
            if self.simulator.check_success():
                break
        logger.info(f"Successfully started the {self.simulator.process.pid} process.")

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()

    def monitoring_status(self):
        logger.info("monitor status")
        while True:
            if self.simulator.process.poll() is not None:
                self.simulator.stop(del_log=False)
                self.benchmark.stop(del_log=False)
                raise subprocess.SubprocessError(f"Failed in run mindie. "
                                                 f"return code: {self.simulator.process.returncode}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_target_server(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 检查mindie状态，检查benchmark状态
        """
        for _ in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
            except Exception as e:
                logger.error(f"Failed in Mindie Running. error: {e}， mindie log {self.simulator.mindie_log}")
                self.stop_target_server(del_log=False)
                continue
            time.sleep(1)
            try:
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e}, benchmark log {self.benchmark.run_log}")
                self.stop_target_server(del_log=False)
                continue
            time.sleep(1)
            try:
                self.monitoring_status()
            except Exception as e:
                self.stop_target_server(del_log=False)
                logger.error(f"Failed in monitoring status. error: {e}, mindie log {self.simulator.mindie_log}, "
                             f"benchmark log {self.benchmark.run_log}")
                continue
            return
        raise ValueError(
            f"Failed in run_target_server, params: {self.simulate_run_info}")

    def stop_target_server(self, del_log=True):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

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
        self.simulate_run_info = map_param_with_value(params, params_field)
        try:
            self.run_target_server(params, params_field)
            time.sleep(1)
            performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}")
            self.data_storage.save(PerformanceIndex(), tuple(self.simulate_run_info), self.benchmark.benchmark_config,
                                   error=e)
            self.stop_target_server(del_log=False)
            raise e
        self.data_storage.save(performance_index, tuple(self.simulate_run_info), self.benchmark.benchmark_config,
                               error=None)
        self.stop_target_server()
        return performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, rpc_clients: List[ServerProxy], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpc_clients = rpc_clients

    def back_up(self):
        if self.bak_path:
            _cur_bak_path = get_train_sub_path(self.bak_path)
            self.simulator.bak_path = _cur_bak_path
            self.benchmark.bak_path = _cur_bak_path
            for rpc in self.rpc_clients:
                if rpc.simulator:
                    rpc.simulator.bak_path = _cur_bak_path

    def monitoring_status(self):
        logger.info("Start monitoring")
        while True:
            all_poll = [self.simulator.process.poll()]
            for rpc in self.rpc_clients:
                all_poll.append(rpc.process_poll())
            if any([_i is not None for _i in all_poll]):
                self.stop_target_server(del_log=False)
                raise subprocess.SubprocessError(
                    f"Failed in run mindie. all status: {all_poll}, machine info: master, {self.rpc_clients}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        _simulate_run_info = map_param_with_value(params, params_field)
        [rpc.run_simulator(params.tolist()) for rpc in self.rpc_clients]
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()
        [rpc.check_success() for rpc in self.rpc_clients]

    def stop_target_server(self, del_log=True):
        super(ScheduleWithMultiMachine, self).stop_target_server(del_log)
        [rpc.stop_simulator(del_log) for rpc in self.rpc_clients]


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
                if not all(_pos):
                    continue
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
            try:
                _res = self.scheduler.run(x[i], self.target_field)
                # 根据采集的数据，计算最优化值
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(f"Failed. error: {e}, please check.")
                logger.exception("What?!")
                _fitness = inf
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
        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.target_field),
                                        options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                        init_pos=self.init_pos, breakpoint_pos=self.history_pos,
                                        breakpoint_cost=self.history_cost)
        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
        logger.info(
            f"best cost {cost}, best joint_vars: "
            f"{[self.target_field[i].format_func(k) for i, k in enumerate(joint_vars)]}")
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
    # 单机benchmark
    if args.benchmark_policy == BenchMarkPolicy.benchmark.value and args.deploy_policy == DeployPolicy.single.value:
        benchmark = BenchMark(settings.benchmark, bak_path=bak_path)
    # 多机 传统benchmark
    elif args.benchmark_policy == BenchMarkPolicy.custom_benchmark.value and \
            args.deploy_policy == DeployPolicy.multiple.value:
        benchmark = RPCCustomBenchMark(rpc_clients, settings.benchmark, bak_path=bak_path,
                                       analyze_tool=args.analyze_tool)
    # profiler benchmark, profiler只能采集主节点，因此多机情况下，也是运行单个机器的实例，处理数据。
    elif args.benchmark_policy == BenchMarkPolicy.profiler_benchmark.value:
        benchmark = ProfilerBenchmark(settings.benchmark, bak_path=bak_path, analyze_tool=args.analyze_tool)
    else:
        # 默认 自定义单机
        benchmark = CustomBenchMark(settings.benchmark, bak_path=bak_path, analyze_tool=args.analyze_tool)
    # 存储结果，只在主节点存储结果
    data_storage = DataStorage(settings.data_storage)
    # 初始化调度模块，支持单机和多机。
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
parser.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.profiler_benchmark.value,
                    choices=[k.value for k in list(BenchMarkPolicy)],
                    help="Whether to use custom performance indicators or mindie performance indicators. "
                         "Benchmark and custom_benchmark are supported.")
parser.add_argument("-lh", "--load_history", default=False, action="store_true",
                    help="Indicates whether to customize the initial location based on historical records.")
parser.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                    help="Continue from where the last optimization was aborted.")
parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                    choices=[k.value for k in list(DeployPolicy)],
                    help="Indicates whether the multi-node running policy is used.")
parser.add_argument("--back_up", default=False, action="store_false",
                    help="Whether to back up data.")
parser.add_argument("-a", "--analyze_tool", default=AnalyzeTool.profiler.value,
                    choices=[k.value for k in list(AnalyzeTool)], help="Tool of data to be analyze")

if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
