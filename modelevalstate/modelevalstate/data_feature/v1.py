# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import csv
import math
import threading
import json
from collections import namedtuple
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from statistics import mean
from warnings import warn

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from modelevalstate.inference.file_reader import FileHanlder
from modelevalstate.inference.common import get_bins_and_label

HARDWARE_FIELD = ("cpu_count", "cpu_mem", "soc_name", "npu_mem")
HardWare = namedtuple("HardWare", HARDWARE_FIELD, defaults=[0, 0, "Ascend910B3", 0])

ENV_FIELD = (
    "atb_llm_razor_attention_enable", "atb_llm_razor_attention_rope", "bind_cpu", "mies_use_mb_swapper",
    "mies_pecompute_threshold",
    "mies_tokenizer_sliding_window_size", "atb_llm_lcoc_enable", "lccl_deterministic",
    "hccl_deterministic", "atb_matmul_shuffle_k_enable")
EnvField = namedtuple("EnvField", ENV_FIELD)

MINDIE_FIELD = (
    "cache_block_size", "mindie__max_seq_len", "world_size", "cpu_mem_size", "npu_mem_size", "max_prefill_tokens",
    "max_prefill_batch_size", "max_batch_size")
MindieConfig = namedtuple("MindieConfig", MINDIE_FIELD)

MODEL_CONFIG_FIELD = (
    "architectures", "hidden_act", "initializer_range", "intermediate_size", "max_position_embeddings", "model_type",
    "num_attention_heads", "num_hidden_layers", "tie_word_embeddings", "torch_dtype", "use_cache", "vocab_size",
    "quantize", "quantization_config")

ModelConfig = namedtuple("ModelConfig", MODEL_CONFIG_FIELD)

BATCH_FIELD = (
    "batch_stage", "batch_size", "total_need_blocks", "total_prefill_token", "max_seq_len", "model_execute_time")
BATCH_FILE_FIELD = ("ibis_batchid", *BATCH_FIELD, "req_info")
BatchField = namedtuple("BatchField", BATCH_FIELD)
BatchFileField = namedtuple("BatchFileField", BATCH_FILE_FIELD)

REQUEST_FIELD = ("input_length", "need_blocks", "output_length")
REQUEST_FILE_FIELD = ("ibis_reqid", "execute_id", *REQUEST_FIELD)
RequestField = namedtuple("RequestField", REQUEST_FIELD)
RequestFileField = namedtuple("RequestFileField", REQUEST_FILE_FIELD)

MODEL_OP_FIELD = (
    "op_name", "call_count", "input_count", "input_dtype", "input_shape", "output_count", "output_dtype",
    "output_shape", "host_setup_time", "host_execute_time", "kernel_execute_time", "aic_cube_fops", "aiv_vector_fops")
ModelOpField = namedtuple("ModelOpField", MODEL_OP_FIELD)
BATCH_SIZE = "batch_size"
MAX_SEQ_LEN = "max_seq_len"

MODEL_STRUCT_FIELD = (
    "total_param_num", "total_param_size", "embed_tokens_param_size_rate", "self_attn_param_size_rate",
    "mlp_param_size_rate", "input_layernorm_param_size_rate", "post_attention_layernorm_param_size_rate",
    "norm_param_size_rate",
    "lm_head_param_size_rate")
ModelStruct = namedtuple("ModelStruct", MODEL_STRUCT_FIELD, defaults=[0 for i in range(len(MODEL_STRUCT_FIELD))])

QUESTION_FIELD = ("question", "answer")
QuestionField = namedtuple("QuestionField", QUESTION_FIELD)


class HistInfo:
    input_length = get_bins_and_label("input_length", interval=80)
    need_blocks = get_bins_and_label("need_blocks", interval=1)
    need_slots = get_bins_and_label("need_slots", interval=128)
    output_length = get_bins_and_label("output_length", interval=10)


@dataclass
class ModelFilePaths:
    base_path: Path = Path("data/model")
    hardware_path: Optional[Path] = None
    env_path: Optional[Path] = None
    mindie_config_path: Optional[Path] = None
    config_path: Optional[Path] = None
    batch_path: Optional[Path] = None
    request_path: Optional[Path] = None
    model_struct_path: Optional[Path] = None
    model_decode_op_path: Optional[Path] = None
    model_prefill_op_path: Optional[Path] = None

    def __post_init__(self):
        if not self.base_path.exists():
            raise FileNotFoundError(self.base_path)
        if self.hardware_path is None:
            self.hardware_path = self.base_path.joinpath("hardware.json")
        if self.env_path is None:
            self.env_path = self.base_path.joinpath("env.json")
        if self.mindie_config_path is None:
            self.mindie_config_path = self.base_path.joinpath("mindie_config.json")
        if self.config_path is None:
            self.config_path = self.base_path.joinpath("model_config.json")
        if self.batch_path is None:
            self.batch_path = self.base_path.joinpath("batch_need.csv")
        if self.request_path is None:
            self.request_path = self.base_path.joinpath("request_need.csv")
        if self.model_struct_path is None:
            self.model_struct_path = self.base_path.joinpath("model_struct.csv")
        if self.model_decode_op_path is None:
            self.model_decode_op_path = self.base_path.joinpath("model_decode_op.csv")
        if self.model_prefill_op_path is None:
            self.model_prefill_op_path = self.base_path.joinpath("model_prefill_op.csv")
        for path in [self.hardware_path, self.env_path, self.mindie_config_path, self.config_path, self.batch_path,
                     self.request_path,
                     self.model_struct_path, self.model_decode_op_path, self.model_prefill_op_path]:
            if not path.exists():
                raise FileNotFoundError(path)


class ConvertModelFileToCsv:
    """
    转换model数据为一行数据
    """

    def __init__(self, model_file_paths: ModelFilePaths, output: Path = Path("feature.csv"),
                 req_warm_up=0):
        self.model_file_paths = model_file_paths
        self.req_warm_up = req_warm_up
        self.out = output
        self.hardware: Optional[HardWare] = None
        self.env_info: Optional[EnvField] = None
        self.mindie_info: Optional[MindieConfig] = None
        self.model_config_info: Optional[ModelConfig] = None
        self.model_struct_info: Optional[ModelStruct] = None
        self.prefill_op_data: Optional[Dict[int, List[ModelOpField]]] = None
        self.decode_op_data: Optional[Dict[int, List[ModelOpField]]] = None
        self.only_req_num: Dict = {}  # 请求最大的decode
        self.all_request_info = {}    # 所有request信息

    @staticmethod
    def load_hardware_data(hardware_path: Path) -> HardWare:
        with open(hardware_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data
        return HardWare(**{k: v for k, v in data.items() if k in HARDWARE_FIELD})

    @staticmethod
    def load_env_data(env_path: Path) -> EnvField:
        with open(env_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data
        return EnvField(**{k: v for k, v in data.items() if k in ENV_FIELD})

    @staticmethod
    def load_mindie_config(mindie_config_path: Path) -> MindieConfig:
        with open(mindie_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data
        if "max_seq_len" in data:
            data["mindie__max_seq_len"] = data["max_seq_len"]
        return MindieConfig(**{k: v for k, v in data.items() if k in MINDIE_FIELD})

    @staticmethod
    def load_model_config(config_path: Path) -> ModelConfig:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data
        return ModelConfig(**{k: v for k, v in data.items() if k in MODEL_CONFIG_FIELD})

    @staticmethod
    def load_model_struct(model_struct_path: Path) -> ModelStruct:
        with open(model_struct_path, "r", encoding="utf-8", newline="") as f:
            model_struct_reader = csv.reader(f)
            model_struct = None
            for i, row in enumerate(model_struct_reader):
                if i == 0:
                    try:
                        assert tuple(row) == MODEL_STRUCT_FIELD
                    except AssertionError as e:
                        raise AssertionError(f"get fields: {row}, expected fields: {MODEL_STRUCT_FIELD}") from e
                    continue
                model_struct = ModelStruct(*row)
        assert model_struct
        return model_struct 

    @staticmethod
    def load_op_data(op_path: Path) -> Dict[int, List]:
        return FileHanlder.load_op_data(op_path)

    @staticmethod
    def load_all_req_info(request_need_csv: Path, req_queue: Optional[Queue] = None):
        res = {}
        with open(request_need_csv, newline='') as request_files:
            request_reader = csv.reader(request_files)
            for i, row in enumerate(request_reader):
                if i == 0:
                    try:
                        assert tuple(row) == REQUEST_FILE_FIELD
                    except AssertionError as e:
                        raise AssertionError(f"get fields: {row}, expected fields: {REQUEST_FILE_FIELD}") from e
                    continue
                cur_row_info = RequestFileField(*row)
                res[(int(cur_row_info.ibis_reqid), int(cur_row_info.execute_id))] = RequestField(*cur_row_info[2:])
        if req_queue:
            req_queue.put(res)
        return res

    @staticmethod
    def get_num_of_prefill_and_decode(cur_batch_info: BatchField, tmp_prefill_req_id: List[int], cur_batch_size: int, 
                                      all_prefill_req: List[int], all_relation_req: List[int]):
        _prefill_num, _decode_num = 0, 0

        if cur_batch_info.batch_stage == "prefill":
            _prefill_num = cur_batch_size
            if set(tmp_prefill_req_id) & set(all_prefill_req):
                ValueError(
                    f"Found duplicated prefill request. {set(tmp_prefill_req_id) & set(all_prefill_req)}")
            else:
                all_prefill_req.extend(tmp_prefill_req_id)
        elif cur_batch_info.batch_stage == "decode":
            _decode_num = cur_batch_size
        else:
            for j in range(0, len(all_relation_req), 2):
                if all_relation_req[j + 1] == 0:
                    _prefill_num += 1
                else:
                    _decode_num += 1
        return _prefill_num, _decode_num

    def load_static_data(self):
        self.hardware = self.load_hardware_data(self.model_file_paths.hardware_path)
        self.env_info = self.load_env_data(self.model_file_paths.env_path)
        self.mindie_info = self.load_mindie_config(self.model_file_paths.mindie_config_path)
        self.model_config_info = self.load_model_config(self.model_file_paths.config_path)
        self.model_struct_info = self.load_model_struct(self.model_file_paths.model_struct_path)
        self.prefill_op_data = self.load_op_data(self.model_file_paths.model_prefill_op_path)
        self.decode_op_data = self.load_op_data(self.model_file_paths.model_decode_op_path)

    def get_op_field(self, batch_stage: str, batch_size: int, max_seq_len: int = 0) -> Tuple[ModelOpField]:
        return FileHanlder.get_op_field(batch_stage, batch_size, max_seq_len, self.prefill_op_data, self.decode_op_data)

    def load_batch_info(self) -> Tuple[BatchField, List[RequestField], List[ModelOpField], ModelStruct, ModelConfig,
                                       MindieConfig, EnvField, HardWare]:
        # 读取整个model获取的所有特性信息，将其转换为csv
        req_queue = Queue()
        t = threading.Thread(target=self.load_all_req_info, args=(self.model_file_paths.request_path, req_queue))
        t.start()
        self.load_static_data()
        t.join()
        all_request_info = req_queue.get()
        self.all_request_info = all_request_info
        # 读取固定的json数据
        target_csv = []
        # 读取batch信息
        # 将batch 信息和request信息对应起来
        _prefill_sum = 0
        _decode_sum = 0
        _all_prefill_req = []
        with open(self.model_file_paths.batch_path, newline='') as batch_files:
            batch_reader = csv.reader(batch_files)
            for i, row in enumerate(batch_reader):
                if i == 0:
                    try:
                        assert tuple(row) == BATCH_FILE_FIELD
                    except AssertionError as e:
                        raise AssertionError(f"get fields: {row}, expected fields: {BATCH_FILE_FIELD}") from e
                    continue
                cur_row_info = BatchFileField(*row)
                cur_batch_info = BatchField(*cur_row_info[1:-1])
                _cur_batch_info_batch_size = int(cur_batch_info.batch_size)
                relation_req = []
                _all_relation_req = eval(cur_row_info.req_info)
                _tmp_prefill_req_id = []
                _tmp_decode_in_prefill = 0
                for j in range(0, len(_all_relation_req), 2):
                    if tuple(_all_relation_req[j: j + 2]) in all_request_info:
                        if int(_all_relation_req[j]) < self.req_warm_up:
                            # 去掉预热
                            break
                        if cur_batch_info.batch_stage == "prefill" and _all_relation_req[j + 1] != 0:
                            _cur_batch_info_batch_size -= 1
                            _tmp_decode_in_prefill += 1
                            continue
                        _tmp_prefill_req_id.append(_all_relation_req[j])
                        _cur_req_info = all_request_info[tuple(_all_relation_req[j: j + 2])]
                        relation_req.append(tuple(_cur_req_info))
                    else:
                        raise ValueError("Not Found request info.")
                # 预热的数据不分析
                if not relation_req:
                    continue
                target_csv.append([tuple(cur_batch_info), tuple(relation_req), 
                                   tuple(self.get_op_field(cur_batch_info.batch_stage, _cur_batch_info_batch_size,
                                                           int(float(cur_batch_info.max_seq_len)))),
                                   tuple(self.model_struct_info), 
                                   tuple(self.model_config_info), tuple(self.mindie_info), tuple(self.env_info),
                                   tuple(self.hardware)])
                _tmp_prefill_num, _tmp_decode_num = self.get_num_of_prefill_and_decode(cur_batch_info, 
                                                                                       _tmp_prefill_req_id,
                                                                                       _cur_batch_info_batch_size,
                                                                                       _all_prefill_req,
                                                                                       _all_relation_req)
                _prefill_sum += _tmp_prefill_num
                _decode_sum += _tmp_decode_num + _tmp_decode_in_prefill
        
        self.only_req_num = {}
        for _req_id, _req_execute_num in all_request_info.keys():
            if _req_id < self.req_warm_up:
                continue
            if _req_id in self.only_req_num:
                self.only_req_num[_req_id] = max(self.only_req_num[_req_id], int(_req_execute_num))
            else:
                self.only_req_num[_req_id] = int(_req_execute_num)
        
        assert len(self.only_req_num.keys()) == _prefill_sum
        assert sum(self.only_req_num.values()) == _decode_sum

        return tuple(target_csv)

    def convert_to_csv(self):
        res = self.load_batch_info()
        with open(self.out, "w", newline="") as f:
            batch_writer = csv.writer(f)
            batch_writer.writerow(
                [BATCH_FIELD, REQUEST_FIELD, MODEL_OP_FIELD, MODEL_STRUCT_FIELD, MODEL_CONFIG_FIELD, MINDIE_FIELD,
                 ENV_FIELD, HARDWARE_FIELD])

            batch_writer.writerows(res)
    
    def plot_feature(self, save_path: Optional[Path]):
        p = sns.displot(list(self.only_req_num.values()), kde=True)
        _mean = int(mean(self.only_req_num.values()))
        p.savefig(save_path.joinpath(f"request_decode_num_hist_mean_{_mean}.png"))
        plt.close()

        req_input_lens = []
        req_decode_nums = []
        for req_id, max_execute_id in self.only_req_num.items():
            req_decode_nums.append(max_execute_id)
            req_input_lens.append(self.all_request_info[(req_id, max_execute_id)].input_length)
        sns.scatterplot(x=req_input_lens, y=req_decode_nums, )
        plt.xlabel("input length")
        plt.ylabel("decode num")
        plt.savefig(save_path.joinpath(f"request_input_len_and_decode_num_scatter_{_mean}.png"))
        plt.close()
    
    def save_only_req_num(self, save_path: Optional[Path]):
        with open(save_path.joinpath("req_id_and_decode_num.json"), "w") as f:
            json.dump(self.only_req_num, f)


class FileReader:
    def __init__(self, file_paths: List[Path], num_lines: int = math.inf, start_lines: int = 0,
                 start_file_index: int = 0,
                 columns: Optional[List[str]] = None):
        self.file_paths = file_paths
        self.num_lines = num_lines
        self.current_file_index = start_file_index
        self.current_line_index = start_lines
        for _file in file_paths:
            if not _file.exists():
                raise FileNotFoundError(_file)
        self.columns = columns

    def read_rows_number(self, lines: List[pd.DataFrame]) -> int:
        if not lines:
            return self.num_lines
        rows_number = 0
        for _df in lines:
            rows_number += _df.shape[0]
        return self.num_lines - rows_number

    def read_lines(self) -> pd.DataFrame:
        lines = []
        while len(lines) < self.num_lines:
            try:
                if self.current_file_index >= len(self.file_paths):
                    # 读取完所有文件结束
                    break
                file_path = self.file_paths[self.current_file_index]
                if self.num_lines == math.inf:
                    df = pd.read_csv(file_path, skiprows=self.current_line_index)
                    lines.append(df)
                    # 继续读取下一个文件
                    self.current_file_index += 1
                    self.current_line_index = 0
                else:
                    _expect_nrows = self.read_rows_number(lines)
                    df = pd.read_csv(file_path, nrows=_expect_nrows, skiprows=self.current_line_index)
                    if self.columns:
                        df.columns = self.columns
                    else:
                        self.columns = df.columns.tolist()
                    lines.append(df)
                    lines_rows = sum([k.shape[0] for k in lines])
                    if lines_rows == self.num_lines:
                        # 读取到所需行结束
                        self.current_line_index += _expect_nrows
                        break
                    else:
                        # 未读取到所需行数据
                        self.current_line_index = 0
                        self.current_file_index += 1
            except Exception as e:
                warn(f"读取文件 {self.file_paths[self.current_file_index]} 时发生错误: {e}. 请核对。暂时跳过读取该文件的数据。")
                self.current_file_index += 1
                self.current_line_index = 0
        if not lines and self.current_file_index >= len(self.file_paths):
            raise StopIteration
        elif not lines:
            raise ValueError(f"lines is empty. lines: {lines}")
        return pd.concat(lines)


class ConvertRequestFileToCsv:
"""提取prompt信息 request信息等转换为预测decode num的基本数据"""

    def __init__(self, request_path: Path, question_path: Path, output: Path = Path("decode_num.csv"),
                 req_warm_up: int = 0):
        self.request_path = request_path
        self.question_path = question_path
        self.req_warm_up = req_warm_up
        self.out = output

    @staticmethod
    def load_question_data(question_path: Path) -> List[QuestionField]:
        target_data = []
        with open(question_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                q = QuestionField(*[data.get(k, "") for k in QUESTION_FIELD])
                target_data.append(q)
        return target_data

    def convert_to_csv(self):
        # 加载request信息
        all_req_info = ConvertModelFileToCsv.load_all_req_info(self.request_path)
        # 加载question问题
        question = self.load_question_data(self.question_path)
        # 关联req和question
        # 处理req信息，只留req id信息
        req_info = {}
        for k, v in all_req_info.items():
            _req_id, _execute_id = k
            if _req_id in req_info:
                req_info[_req_id] = max(req_info[_req_id], int(_execute_id))
            else:
                req_info[_req_id] = int(_execute_id)
        _target_info = []
        for _req_id, _max_execute_id in req_info.items():
            if _req_id < self.req_warm_up:
                continue
            _target_info.append({**all_req_info[(_req_id, _max_execute_id)]._asdict(),
                                **question[_req_id - self.req_warm_up]._asdict()})
        df = pd.DataFrame(_target_info)
        # 保存特征到csv中
        df.to_csv(self.out, index=False)


if __name__ == "__main__":
    base_path = Path(r"D:\PyProject\ModelEvalState\data\v1.0.0\llama3-8b-2")
    analysis_feature = base_path.joinpath("analysis_feature")
    analysis_feature.mkdir(parents=True, exist_ok=True)
    model_file_paths = ModelFilePaths(base_path=base_path)
    convert_model_file_to_csv = ConvertModelFileToCsv(model_file_paths=model_file_paths,
                                                          output=base_path.joinpath("feature.csv"),
                                                          req_warm_up=20)
    convert_model_file_to_csv.convert_to_csv()
    convert_model_file_to_csv.plot_feature(analysis_feature)
    convert_model_file_to_csv.save_only_req_num(save_path=analysis_feature)

    crf = ConvertRequestFileToCsv(base_path.joinpath("request_need.csv"), base_path.joinpath("medium1_dataset3.jsonl"),
                                  req_warm_up=20, output=base_path.joinpath("decode_num.csv"))
    crf.convert_to_csv()
    print("finished")
