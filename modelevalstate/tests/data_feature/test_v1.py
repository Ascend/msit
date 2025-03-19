# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from pathlib import Path
from modelevalstate.data_feature.v1 import FileReader


def test_file_reader():
    file_paths = [Path(r"D:\PyProject\state_eval\tmp\test_x.csv"),
                  Path(r"D:\PyProject\state_eval\tmp\train_x.csv")]
    fr = FileReader(file_paths)
    res = fr.read_lines()
    assert res.shape[0] > 1977
    num_lines = 1000
    fr = FileReader(file_paths, num_lines=num_lines)
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
