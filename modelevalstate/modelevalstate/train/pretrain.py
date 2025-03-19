# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
1. 训练基础模型
2. 用基础模型拟合概率
3. 用新增加数据继续训练模型
"""
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from math import ceil
from typing import Tuple, Optional, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from modelevalstate.train.common import computer_speed_with_second, get_train_sub_path, update_gloabal_coefficient
from modelevalstate.train.xgb_state_model import StateXgbModel
from modelevalstate.data_feature.dataset import (
    MyDataSet, 
    CustomOneHotEncoder, 
    CustomLabelEncoder, 
    preset_category_data, 
    DecodeDataSet
)
from modelevalstate.inference.constant import OpAlgorithm
from modelevalstate.inference.common import (
    HistInfo, 
    model_op_size, 
    OP_EXPECTED_FIELD_MAPPING, 
    OP_SACLE_FIELD_MAPPING
)


@dataclass
class NodeInfo:
    stage: str # 当前模型状态的类型 P/D
    batch_size: int # 当前模型状态处理的请求个数


class PreTrainModel:
    def __init__(self, state_param: Optional[StateParam] = None, dataset: Optional[MyDataSet] = None,
                 model: Optional[StateXgbModel] = None, plt_data: bool = False):
        self.state_param = state_param
        self.dataset = dataset
        self.model = model
        self.plt_data = plt_data
        self.computer_up_expectations = []
        self.computer_ud_expectations = []
        self.real_up_expectations = []
        self.real_ud_expectations = []
        self.rmse = []
        self.r2 = []
        self.mape = []

    @staticmethod
    def get_decode_and_prefill_time(target: Tuple[NodeInfo], field: str):
        # 获取不同类型batch的运行时间
        _inner_prefill_time = {}
        _inner_decode_time = {}
        for line in target:
            _cur_state = State()
            if line.stage == _PREFILL:
                _cur_state.batch_prefill = line.batch_size
                update_global_coefficient(_inner_prefill_time, _cur_state, getattr(line, field))
            elif line.stage == _DECODE:
                _cur_state.batch_decode = line.batch_size
                update_global_coefficient(_inner_decode_time, _cur_state, getattr(line, field))
        return _inner_prefill_time, _inner_decode_time

    @staticmethod
    def get_up_ud(target: Tuple[NodeInfo], field: str):
        # 获取不同类型的batch运行的单位时延迟
        _inner_up = {}
        _inner_ud = {}
        for line in target:
            _cur_state = State()
            if line.stage == _PREFILL:
                _cur_state.batch_prefill = line.batch_size
                update_global_coefficient(_inner_up, _cur_state, computer_speed_with_second(line, field))
            elif line.stage == _DECODE:
                _cur_state.batch_decode = line.batch_size
                update_global_coefficient(_inner_ud, _cur_state, computer_speed_with_second(line, field))
        return _inner_up, _inner_ud

    def train(self, lines_data: Optional[DataFrame] = None,
              middle_save_path: Optional[Path] = None):
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=middle_save_path)
        self.dataset.custom_encoder.save()
        rmse = self.model.train(self.dataset, middle_save_path=middle_save_path)
        self.rmse.append(rmse)

    def partial_train(self, lines_data: Optional[DataFrame] = None,
                      middle_save_path: Optional[Path] = None):
        self.dataset.custom_encoder.fit(load=True)
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=middle_save_path)
        rmse = self.model.train(self.dataset, train_type="partial_fit", middle_save_path=middle_save_path)
        self.rmse.append(rmse)

    @staticmethod
    def get_stage_after_preprocess(row: pd.Series, encoder: Union[CustomOneHotEncoder, CustomLabelEncoder]):
        # 根据预处理后的数据识别该行数据是decode还是prefill

        if isinstance(encoder, CustomOneHotEncoder):
            batch_stage_encoder = \
            [encoder.one_hot_encoders[i] for i, v in enumerate(encoder.one_hots) if v.name == "batch_stage"][0]
            _batch_index = [i for i in row.index if "batch_stage" in i]
            stage = batch_stage_encoder.inverse_transform([[int(row[i]) for i in _batch_index]])

        else:
            batch_stage_encoder = \
            [encoder.category_encoders[i] for i, v in enumerate(encoder.category_info) if v.name == "batch_stage"][0]
            stage = batch_stage_encoder.inverse_transform([int(row.batch_stage)])[0]

        return stage

    def get_nodes_with_model_predict(self, features: DataFrame):
        # 使用模型进行预测
        target_data = []
        for _, row in features.iterrows():
            _predict = self.model.predict((row,))[0]
            stage = self.get_stage_after_preprocess(row, self.dataset.custom_encoder)
            _cur_node = NodeInfo(stage, row.batch_size)
            setattr(_cur_node, self.state_param.predict_field, _predict)
            target_data.append(_cur_node)
        return tuple(target_data)

    @staticmethod
    def get_nodes_with_origin_data(features: DataFrame, labels: DataFrame, predict_field: str,
                                   encoder: Union[CustomOneHotEncoder, CustomLabelEncoder]):
        # 获取原来的node信息
        target_data = []
        for ind, row in features.iterrows():
            stage = PretrainModel.get_stage_after_preprocess(row, encoder)
            _cur_node = NodeInfo(stage, row.batch_size)
            setattr(_cur_node, predict_field, labels.iloc[ind, 0])
            target_data.append(_cur_node)
        return tuple(target_data)

    def predict_and_plot(self, features: DataFrame, labels: DataFrame, predict_field: str, save_path: Optional[Path]):
        origin_data = self.get_nodes_with_origin_data(features, labels, predict_field, self.dataset.custom_encoder)
        data = self.get_nodes_with_model_predict(features)
        r2 = r2_score([getattr(k, predict_field) for k in origin_data], [getattr(k, predict_field) for k in data])
        self.r2.append(r2)
        mape = mean_absolute_percentage_error([getattr(k, predict_field) for k in origin_data],
                                              [getattr(k, predict_field) for k in data])
        self.mape.append(mape)
        _all_up, _all_ud = self.get_up_ud(data, predict_field)
        origin_up, origin_ud = self.get_up_ud(tuple(origin_data), predict_field)

        if self.state_param.plot_velocity_std:
            self.plot_velocity_std(origin_up, _all_up, origin_ud, _all_ud, save_path=save_path)
        if self.state_param.plot_input_time_with_predict:
            # 绘制时间
            _all_prefill_time, _all_decode_time = self.get_decode_and_prefill_time(data, predict_field)
            origin_prefill_time, origin_decode_time = self.get_decode_and_prefill_time(tuple(origin_data),
                                                                                       predict_field)
            AnalysisState.plot_input_velocity_with_predict(origin_prefill_time, _all_prefill_time, "batch_prefill",
                                                           f"origin and predict prefill time {predict_field} std",
                                                           "batch_prefill",
                                                           "time us", save_path=save_path)
            AnalysisState.plot_input_velocity_with_predict(origin_decode_time, _all_decode_time, "batch_decode",
                                                           f"origin and predict decode time {predict_field} std",
                                                           "batch_decode",
                                                           "time us", save_path=save_path)
        return _all_up, _all_ud

    def predict(self, lines_data: DataFrame,
                save_path: Optional[Path] = None):
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=save_path)
        return self.predict_and_plot(self.dataset.features, self.dataset.labels, self.state_param.predict_field,
                                     save_path=save_path)

    def plot_velocity_std(self, origin_up, all_Up, origin_ud, all_ud, save_path: Optional[Path] = None):
        # 对比Up,Ud的分布
        AnalysisState.plot_input_velocity_with_predict(origin_up, all_Up, "batch_prefill",
                                                       f"origin and predict Up {self.state_param.predict_field} std",
                                                       "batch_prefill",
                                                       "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity_with_predict(origin_ud, all_ud, "batch_decode",
                                                       f"origin and predict Ud {self.state_param.predict_field} std",
                                                       "batch_decode",
                                                       "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity(origin_up, "batch_prefill", f"Up {self.state_param.predict_field} std",
                                          "batch_prefill",
                                          "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity(origin_ud, "batch_decode", f"Ud {self.state_param.predict_field} std",
                                          "batch_decode",
                                          "velocity", save_path=save_path)

    def bak_model(self, increment_stage: str = "base"):
        _bak_dir = self.state_param.bak_dir.joinpath(increment_stage)
        _bak_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.state_param.xgb_model_save_model_path,
                    _bak_dir.joinpath(self.state_param.xgb_model_save_model_path.name))

    def save_readme(self):
        with open(self.state_param.bak_dir.joinpath("../README.md"), 'w', encoding="utf-8") as f:
            f.write("模型训练参数\n")
            f.write(f"model: {type(self.model)}\n")
            f.write(f"dataset: {type(self.dataset)}\n")
            f.write(f"encoder info: {type(self.dataset.custom_encoder)} \n")
            f.write(f"rmse: {self.rmse}us(微妙), r2(确定系数): {self.r2}, mape(平均绝对百分比误差): {self.mape} \n")
            f.write(f"param: \n")
            for k, v in asdict(self.state_param).items():
                f.write(f"  {k}:{v}\n")
            hist_info = {k: v for k, v in HistInfo.__dict__.items() if not k.startswith("__")}
            f.write(f"hist info: {hist_info}\n")
            f.write(f"model op size: {model_op_size}\n")
            f.write(f"OP_EXPECTED_FIELD_MAPPING: {OP_EXPECTED_FIELD_MAPPING}\n")
            f.write(f"OP_SCALE_HIST_FIELD_MAPPING: {OP_SCALE_HIST_FIELD_MAPPING}\n")


    def plot_metric(self, save_path: Optional[Path] = None):
        data = {"rmse": self.rmse, "r2": self.r2, "mape": self.mape}
        df = pd.DataFrame(data)
        sns.scatterplot(df)
        if save_path:
            plt.savefig(save_path.joinpath("metric.png"))
            plt.close()
        else:
            plt.show()


class ReqDecodePretrainModel(PretrainModel):
    def train(self, lines_data: Optional[DataFrame] = None,
              middle_save_path: Optional[Path] = None):
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=middle_save_path)
        rmse = self.model.train(self.dataset, middle_save_path=middle_save_path)
        self.rmse.append(rmse)

    def get_nodes_with_model_predict(self, features: DataFrame):
        # 使用模型进行预测
        target_data = []
        for _, row in features.iterrows():
            _predict = ceil(self.model.predict((row,))[0])
            target_data.append(_predict)
        return tuple(target_data)

    def predict_and_plot(self, features: DataFrame, labels: DataFrame, predict_field: str, save_path: Optional[Path]):
        origin_data = labels.values
        data = self.get_nodes_with_model_predict(features)

        r2 = r2_score(origin_data, data)
        self.r2.append(r2)
        mape = mean_absolute_percentage_error(origin_data,
                                              data)
        self.mape.append(mape)
        AnalysisState.plot_pred_and_real(data, origin_data, save_path)
        ax = sns.lineplot(pd.DataFrame({"predict": data, "real": origin_data}))
        if save_path:
            plt.savefig(save_path.joinpath("decode_num_predict_real.png"))
        else:
            plt.show()
        plt.close()

    def save_readme(self):
        with open(self.state_param.bak_dir.joinpath("../README.md"), 'w', encoding="utf-8") as f:
            f.write("模型训练参数\n")
            f.write(f"model: {type(self.model)}\n")
            f.write(f"dataset: {type(self.dataset)}\n")
            f.write(f"rmse: {self.rmse}, r2: {self.r2}, mape: {self.mape} \n")
            f.write(f"param: \n")
            for k, v in asdict(self.state_param).items():
                f.write(f"  {k}:{v}\n")


class TrainVersion1:
    @staticmethod
    def train_xgbmodel():
        file_paths = [
            Path(r"PyProject\ModelEvalState\data\v1\batch_max_seq_2_op\llama3-8b\feature.csv"),
            Path(r"PyProject\ModelEvalState\data\v1\batch_max_seq_2_op\llama3-8b1226-12\feature.csv"),
                      ]
        base_dir = get_train_sub_path()
        logging.info('base_dir', base_dir)
        sp = StateParam(
            base_path=base_dir,
            predict_field="model_execute_time",
            save_model=True,
            shuffle=True,
            plot_pred_and_real=True,
            plot_data_feature=True,
            start_num_lines=4000,
            op_algorithm=OpAlgorithm.EXPECTED,
            title = f"MixModel without warmup with batch max seq 2 op info"
        )
        model = StateXgbModel(
            train_param=sp.xgb_model_train_param,
            update_param=sp.xgb_model_update_param,
            save_model_path=sp.xgb_model_save_model_path,
            load_model_path=sp.xgb_model_save_model_path,
            show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
            show_feature_importance=sp.xgb_model_show_feature_importance,
        )
        custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=sp.ohe_path)
        custom_encoder.fit()
        dataset = MyDataSet(custom_encoder=custom_encoder, predict_field=sp.predict_field,
                            shuffle=sp.shuffle, op_algorithm=sp.op_algorithm)

        pm = PretrainModel(state_param=sp, dataset=dataset, model=model, plt_data=sp.plot_data_feature)
        # 自定义训练数据
        TrainVersion1.custom_train(file_paths, sp, pm)
        pm.plot_metric(sp.step_dir)

    @staticmethod
    def custom_train(file_paths: List[Path], sp: StateParam, pm: PretrainModel):
        # 训练模型，将全部数据1:9分，9进行训练，1进行预测。
        fl = FileReader(file_paths)
        line_data = fl.read_lines()
        train_data, test_data = train_test_split(line_data, test_size=0.1, shuffle=True)
        logging.info(train_data.shape)
        save_path = sp.step_dir.joinpath("base")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.train(train_data.reset_index(drop=True), middle_save_path=save_path)
        pm.dataset.save(save_path)
        logging.info('feature shape', pm.dataset.features.shape)
        sp.comments = f"data shuffle: True, \n train case: {pm.dataset.train_x.shape}, \
            validate case: {pm.dataset.test_x.shape}, predict case: {test_data.shape}"
        pm.bak_model()
        logging.info(test_data.shape)
        save_path = sp.step_dir.joinpath("1")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.predict(test_data.reset_index(drop=True), save_path)
        pm.dataset.save(save_path)
        pm.plot_metric(sp.step_dir)
        pm.save_readme()

    @staticmethod
    def train_with_prefill_with_decode(model_type: str = "prefill"):
        # 训练模型，训练一个prefill模型,或者decode模型
        file_paths = [
            Path(r"/data/v1/llama3-8b/feature.csv"),
            Path(r"/data/v1/llama3-8b1226-12/feature.csv"),
                      ]
        base_dir = get_train_sub_path()
        logging.info('base_dir', base_dir)
        sp = StateParam(
            base_path=base_dir,
            predict_field="model_execute_time",
            save_model=True,
            shuffle=True,
            plot_pred_and_real=True,
            plot_data_feature=True,
            start_num_lines=4000,
            op_algorithm=OpAlgorithm.EXPECTED,
            title = f"{model_type} without warm up."
        )
        model = StateXgbModel(
            train_param=sp.xgb_model_train_param,
            update_param=sp.xgb_model_update_param,
            save_model_path=sp.xgb_model_save_model_path,
            load_model_path=sp.xgb_model_save_model_path,
            show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
            show_feature_importance=sp.xgb_model_show_feature_importance,
        )
        custom_encoder = CustomLabelEncoder(preset_category_data, save_dir=sp.ohe_path)
        custom_encoder.fit()
        dataset = MyDataSet(custom_encoder=custom_encoder, predict_field=sp.predict_field,
                            shuffle=sp.shuffle, op_algorithm=sp.op_algorithm)
        pm = PretrainModel(state_param=sp, dataset=dataset, model=model, plt_data=sp.plot_data_feature)
        fl = FileReader(file_paths)
        line_data = fl.read_lines()
        # 只获取包含prefill/decode的行数的模型
        line_data = line_data[line_data[line_data.columns[0]].str.contains(model_type)]
        train_data, test_data = train_test_split(line_data, test_size=0.1, shuffle=True)

        logging.info(train_data.shape)
        save_path = sp.step_dir.joinpath("base")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.train(train_data.reset_index(drop=True), middle_save_path=save_path)
        pm.dataset.save(save_path)
        logging.info('feature shape', pm.dataset.features.shape)
        sp.comments = f"data shuffle: True, train case: {pm.dataset.train_x.shape}, validate case: \
            {pm.dataset.test_x.shape}, predict case: {test_data.shape}"
        pm.bak_model()
        logging.info(test_data.shape)
        save_path = sp.step_dir.joinpath("1")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.predict(test_data.reset_index(drop=True), save_path)
        pm.dataset.save(save_path)
        pm.plot_metric(sp.step_dir)
        pm.save_readme()

    @staticmethod
    def increment_train(fl: FileReader, sp: StateParam, pm: PretrainModel):
        # 增量训练
        count = 1
        while True:
            try:
                # 1000行
                lines = fl.read_lines()
                save_path = sp.step_dir.joinpath(str(count))
                save_path.mkdir(parents=True, exist_ok=True)
                pm.predict(lines, save_path=save_path)
                pm.partial_train(lines, middle_save_path=save_path)
                count += 1
            except StopIteration:
                break
        pm.bak_model(increment_stage="finished")
        pm.save_readme()

    @staticmethod
    def full_train(fl: FileReader, sp: StateParam, pm: PretrainModel):
        # 全量训练
        train_data = fl.read_lines()
        save_path = sp.step_dir.joinpath("base")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.train(train_data, middle_save_path=save_path)
        pm.bak_model()

    @staticmethod
    def train_req_xgb_model():
        file_paths = [Path(r"PyProject\state_eval\data\v1\llama3-8b-12-13\decode_num.csv")]
        base_dir = get_train_sub_path()
        logging.info('base_dir', base_dir)
        sp = StateParam(
            base_path=base_dir,
            predict_field="output_length",
            save_model=True,
            shuffle=True,
            plot_pred_and_real=True,
            plot_data_feature=True,
            start_num_lines=2000,
            op_algorithm=OpAlgorithm.EXPECTED,
            title = "DecodeNumOfREQModel without warmup"
        )
        model = StateXgbModel(
            train_param=sp.xgb_model_train_param,
            update_param=sp.xgb_model_update_param,
            save_model_path=sp.xgb_model_save_model_path,
            load_model_path=sp.xgb_model_save_model_path,
            show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
            show_feature_importance=sp.xgb_model_show_feature_importance,
        )

        dataset = DecodeDataSet(predict_field=sp.predict_field,
                            shuffle=sp.shuffle)

        pm = ReqDecodePretrainModel(state_param=sp, dataset=dataset, model=model, plt_data=sp.plot_data_feature)
        fl = FileReader(file_paths, num_lines=sp.start_num_lines)
        train_data = fl.read_lines()
        save_path = sp.step_dir.joinpath("base")
        save_path.mkdir(parents=True, exist_ok=True)
        pm.train(train_data, middle_save_path=save_path)
        pm.dataset.save(save_path)
        save_path = sp.step_dir.joinpath("1")
        save_path.mkdir(parents=True, exist_ok=True)
        test_data = fl.read_lines()
        pm.predict(test_data.reset_index(drop=True), save_path)
        pm.plot_metric(sp.step_dir)
        pm.dataset.save(save_path)
        pm.save_readme()


if __name__ == '__main__':
    TrainVersion1.train_xgbmodel()