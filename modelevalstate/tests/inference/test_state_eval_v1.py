# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import sys
sys.path.append(r"D:\PyProject\state_eval\inference")
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, List, Callable, Dict
from pandas import DataFrame

from modelevalstate.data_feature.v1 import FileReader
from modelevalstate.inference.state_eval_v1 import XGBStateEvaluate
from modelevalstate.inference.dataset import CustomLabelEncoder, InputData, preset_category_data, SimpleDataProcessor
from modelevalstate.inference.data_format_v1 import ModelOpField, ModelStruct, ModelConfig, MindieConfig, EnvField, HardWare, RequestField, BatchField
from modelevalstate.train.pretrain import NodeInfo, PretrainModel
from modelevalstate.analysis import AnalysisState


def predict_with_model(lines_data: DataFrame,
                       save_path: Optional[Path] = None,
                       xgb_model_path: Optional[Path] = None,
                       ohe_path: Optional[Path] = None,
                       train_field="model_execute_time",
                       dataset_type: SimpleDataProcessor = SimpleDataProcessor):
    # 转换格式为接口需要格式
    origin_data: List[NodeInfo] = []
    predict_data: List[NodeInfo] = []
    # custom_encoder = CustomOneHotEncoder(preset_category_data,
    #                                              save_dir=ohe_path)
    # custom_encoder.fit(load=True)
    custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=ohe_path)
    custom_encoder.fit()
    # data_processor = DataProcessor(custom_encoder)
    # data_processor = SimpleDataProcessor(custom_encoder)
    data_processor = dataset_type(custom_encoder)
    xgb_state_eval = XGBStateEvaluate(
        xgb_model_path=Path(xgb_model_path),
        dataprocessor=data_processor)
    for ind, row in lines_data.iterrows():
        # 获取真实结果
        batch_data = eval(row[0])
        batch_field = BatchField(*batch_data[:-1])
        _cur_node = NodeInfo(batch_field.batch_stage, batch_field.batch_size)
        setattr(_cur_node, train_field, float(batch_data[-1]))
        origin_data.append(_cur_node)
        input_data = InputData(
            batch_field=batch_field,
            request_field=tuple([RequestField(*[int(float(i)) for i in _req]) for _req in eval(row[1])]),
            model_op_field=tuple([ModelOpField(*_op) for _op in eval(row[2])]),
            model_struct_field=ModelStruct(*eval(row[3])),
            model_config_field=ModelConfig(*eval(row[4])),
            mindie_field=MindieConfig(*eval(row[5])),
            env_field=EnvField(*eval(row[6])),
            hardware_field=HardWare(*eval(row[7]))
        )
        # 使用模型进行预测
        _up, _ud = xgb_state_eval.predict(input_data)
        _cur_node = deepcopy(_cur_node)
        if _up != -1:
            setattr(_cur_node, train_field, _up)
        else:
            setattr(_cur_node, train_field, _ud)
        predict_data.append(_cur_node)
    # 绘制结果图
    all_Up, all_Ud = PretrainModel.get_up_ud(tuple(predict_data), train_field)
    origin_Up, origin_Ud = PretrainModel.get_up_ud(tuple(origin_data), train_field)
    AnalysisState.plot_input_velocity_with_predict(origin_Up, all_Up, "batch_prefill",
                                                   f"origin predict Up {train_field} std velocity", "batch_prefill",
                                                   "velocity", save_path=save_path)
    AnalysisState.plot_input_velocity_with_predict(origin_Ud, all_Ud, "batch_decode",
                                                   f"origin predict Ud {train_field} std velocity", "batch_decode",
                                                   "velocity", save_path=save_path)
    all_Up, all_Ud = PretrainModel.get_decode_and_prefill_time(tuple(predict_data), train_field)
    origin_Up, origin_Ud = PretrainModel.get_decode_and_prefill_time(tuple(origin_data), train_field)
    AnalysisState.plot_input_velocity_with_predict(origin_Up, all_Up, "batch_prefill",
                                                   f"origin predict Up {train_field} std time", "batch_prefill",
                                                   "time", save_path=save_path)
    AnalysisState.plot_input_velocity_with_predict(origin_Ud, all_Ud, "batch_decode",
                                                   f"origin predict Ud {train_field} std time", "batch_decode",
                                                   "time", save_path=save_path)


def run_case(process_num: int, save_result_path: Path, fl: FileReader, call_func: Callable, kwargs: Dict):
    count = 1
    with Pool(process_num) as p:
        while True:
            try:
                # 读取数据
                lines = fl.read_lines()
                # 增量拟合
                save_path = save_result_path.joinpath(str(count))
                save_path.mkdir(exist_ok=True, parents=True)
                if process_num == 1:
                    call_func(lines, save_path, **kwargs)
                else:
                    p.apply_async(call_func, args=(lines, save_path), kwds=kwargs)
                count += 1
            except StopIteration:
                break
        p.close()
        p.join()


def test_state_eval():
    file_paths = [Path(r"D:\PyProject\ModelEvalState\data\v1\llama3-8b1226-12\feature.csv")]
    base_path = Path(r"D:\PyProject\state_eval\tmp\pd_content\train\116")
    xgb_model_path = base_path.joinpath("bak/base/xgb_model.ubj")
    ohe_path = base_path.joinpath("ohe")
    train_field = "model_execute_time"
    save_result_path = base_path.joinpath("test_state_eval")
    save_result_path.mkdir(exist_ok=True, parents=True)

    fl = FileReader(file_paths, num_lines=1000)
    # process_num = cpu_count() - 2
    process_num = 1
    run_case(process_num, save_result_path, fl, predict_with_model, {
                        "xgb_model_path": xgb_model_path,
                        "ohe_path": ohe_path,
                        "train_field": train_field,
                        "dataset_type": SimpleDataProcessor
                    })


if __name__ == '__main__':
    test_state_eval()
