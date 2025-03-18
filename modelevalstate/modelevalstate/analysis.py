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

"""
比较实用numpy计算出来的概率和解析文件的概率
"""
import copy
import json

import josn
from pathlib import Path
from typing import Dict, List, Optional
from matplotlib import pyplot as plt
from statistics import mean, stdev

import numpy as np

from modelevalstate.common import State, my_std

plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AnalysisState:

    @staticmethod
    def computer_mean_sigma(data: Dict[State, List], x_field: str, ):
        # 合并只有decode, prefill
        res = {}
        tmp_data = copy.deepcopy(data)
        for k, v in tmp_data.items():
            if k.batch_prefill:
                s = State(batch_prefill=k.batch_prefill)
            else:
                s = State(batch_decode=k.batch_decode)
            if s in res:
                res[s].extend(v)
            else:
                res[s] = v
        # 计算mean sigma
        _x = []
        _mean = []
        _positive_sigma = []
        _negative_sigma = []
        for k in stored(res.keys(), key=lambda x: getattr(x, x_field)):
            v = res[k]
            if len(v) < 2:
                _x.append(getattr(k, x_field))
                _mean.append(v[0])
                _positive_sigma.append(v[0])
                _negative_sigma.append(v[0])
                continue
            _x.append(getattr(k, x_field))
            _mean.append(mean(v))
            try:
                _sigma = stdev(v)
            except AssertionError:
                try:
                    _sigma = np.std(v)
                except Exception:
                    print('Failed stdenv', v)
                    _sigma = my_std(v)
            _positive_sigma.append(_mean[-1] + _sigma)
            _negative_sigma.append(_mean[-1] - _sigma)
        return _x, _mean, _positive_sigma, _negative_sigma

    @staticmethod
    def plot_input_velocity(data: Dict[State, List], x_field: str, title: str, x_label: str, y_label: str,
                            save_path=None):
        """
        绘制输入数据的平均值, 上波动和下波动曲线
        :return:
        """
        # 合并只有decode, prefill
        _x, _mean, _positive_sigma, _negative_sigma = AnalysisState.computer_mean_sigma(data, x_field)
        plt.plot(_x, _mean, label="mean")
        plt.plot(_x, _positive_sigma, label="positive std")
        plt.plot(_x, _negative_sigma, label="negative std")
        plt.title(title)
        plt.legend()
        plt.grid()
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if save_path:
            plt.savefig(Path(save_path)).joinpath(f"{x_label}_{y_label}_{title}.png")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_input_velocity_with_predict(data: Dict[State, List], predict_data: Dict[State, List], x_field: str,
                                         title: str, x_label: str, y_label: str, save_path=None):
        """
        绘制输入数据和预测数据的平均值, 上波动和下波动曲线。
        :return:
        """
        # 合并只有decode, prefill
        _x, _mean, _positive_sigma, _negative_sigma = AnalysisState.computer_mean_sigma(data, x_field)
        _x, _predict, _predict_positive_sigma, _predict_negative_sigma = AnalysisState.computer_mean_sigma(predict_data,
                                                                                                           x_field)
        plt.figure()
        plt.plot(_x, _mean, label="mean")
        plt.plot(_x, _positive_sigma, label="positive std")
        plt.plot(_x, _negative_sigma, label="negative std")
        plt.plot(_x, _predict, label="predict")
        plt.plot(_x, _predict_positive_sigma, label="predict positive std")
        plt.plot(_x, _predict_negative_sigma, label="predict negative std")
        plt.title(title)
        plt.legend()
        plt.grid()
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if save_path:
            plt.savefig(Path(save_path)).joinpath(f"{x_label}_{y_label}_{title}.png")
            plt.close()
        else:
            plt.show()
        with open(save_path.joinpath(f"{title}.txt"), "w") as f:
             f.write('mean\n')
             f.write(json.dumps(_mean))
             f.write('\n')
             f.write('positive std\n')
             f.write(json.dumps(_positive_sigma))
             f.write('negative std\n')
             f.write(json.dumps(_negative_sigma))
             f.write('\n')
             f.write('predict \n')
             f.write(json.dumps([float(i) for i in _predict]))

    @staticmethod
    def plot_pred_and_real(pred, real, save_path: Optional[Path] = None):
        plt.figure()
        plt.scatter(range(len(pred)), pred, label='pred', alpha=0.5)
        plt.scatter(range(len(real)), real, label='real', alpha=0.5)
        plt.title("predict value and real value")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.legend()
        if save_path:
            plt.savefig(save_path.joinpath("predict value and real value.png"))
            plt.close()
        else:
            plt.show()
