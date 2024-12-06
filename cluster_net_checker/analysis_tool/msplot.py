# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import os.path
import random
import json

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, Select, CustomJS, BasicTicker, PrintfTickFormatter
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from bokeh.layouts import column, row


colors = ["#f5e0de", "#f8bfba", "#faa69e", "#ff9389", "#f0776b"]
colors_line = ["#3f98ce", "#eb17cb", "#42ccab"]
pos_dict = {}
ranks_relation_dict = {}
ranks_info_dict = {}
ranks_order_dict = {}

def get_rank_pos(rows=10, columns=10):
    m = rows
    n = columns
    for i in range(m):
        for j in range(n):
            rank_id = i * n + j
            x = j
            y = i
            pos_dict[rank_id] = (x, y)

    return pos_dict


def get_ranks_info(output_path):
    info_path = os.path.join(output_path, "output_data.json")
    with open(info_path, 'r') as file:
        temp = json.load(file)
    for key, value in temp.items():
        ranks_info_dict[int(key)] = value
    return ranks_info_dict


def get_ranks_relation(output_path):
    relation_path = os.path.join(output_path, "comm_group.json")
    with open(relation_path, 'r') as file:
        temp = json.load(file)
    for key, value in temp.items():
        ranks_relation_dict[int(key)] = value
    return ranks_relation_dict


def get_ranks_order(output_path):
    order_path = os.path.join(output_path, "output_data.json")
    with open(order_path, 'r') as file:
        temp = json.load(file)

    metric_list = list(temp.get("0").keys())
    for metric in metric_list:
        val = {key: value[metric] for key, value in temp.items()}
        ranks_order_dict[metric] = val

    return ranks_order_dict

def plot(rank_id=-1, metric="computing"):

    infolist = []
    for _ in range(len(pos_dict)):
        infolist.append(round(random.uniform(1, 5), 2))
    source = ColumnDataSource(data=dict(
        ids=list(pos_dict.keys()),
        x=[coord[0] for coord in pos_dict.values()],
        y=[coord[1] for coord in pos_dict.values()],
        width=[0.8] * len(pos_dict),  # 假设每个矩形的宽度为1
        height=[0.8] * len(pos_dict),  # 假设每个矩形的高度为1
        computing=[coord['computing'] for coord in ranks_info_dict.values()],
        communication=[coord['communication'] for coord in ranks_info_dict.values()],
        communication_not_overlapped=[coord['communication_not_overlapped'] for coord in ranks_info_dict.values()],
        free=[coord['free'] for coord in ranks_info_dict.values()]
    ))

    p = figure(title="网络拓扑图", sizing_mode = "stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset", )

    p.xgrid.grid_line_color = None  # 隐藏x轴的网格线
    p.ygrid.grid_line_color = None  # 隐藏y轴的网格线
    p.xaxis.axis_label_text_font_size = "0px"  # 隐藏x轴标签
    p.yaxis.axis_label_text_font_size = "0px"  # 隐藏y轴标签
    p.xaxis.visible = False  # 隐藏x轴刻度
    p.yaxis.visible = False  # 隐藏y轴刻度

    # 绘制矩形
    r = p.rect(x='x', y='y', width='width', height='height', source=source,
               fill_color=linear_cmap(metric, colors, low=0, high=2000000),
               line_color=None)

    p.add_layout(r.construct_color_bar(
        major_label_text_font_size="7px",
        ticker=BasicTicker(desired_num_ticks=len(colors)),
        formatter=PrintfTickFormatter(format="%d ms"),
        label_standoff=6,
        border_line_color=None,
        padding=5,
    ), 'right')

    if rank_id != -1:
        # 画连线
        rank_dict = ranks_relation_dict.get(rank_id)
        for key, value in rank_dict.items():
            rank_dict[key] = [int(item) for item in value if item.isdigit()]

        rank_grey = list(pos_dict.keys())

        for values in rank_dict.values():
            for val in values:
                if val in rank_grey:
                    rank_grey.remove(val)

        # 置灰
        x_grey = []
        y_grey = []
        for index in rank_grey:
            x_grey.append(pos_dict.get(index)[0])
            y_grey.append(pos_dict.get(index)[1])
        infolist_grey = []
        for _ in range(len(rank_grey)):
            infolist_grey.append(round(random.uniform(1, 5), 2))

        source_grey = ColumnDataSource(data=dict(
            ids=rank_grey,
            x=x_grey,
            y=y_grey,
            width=[0.8] * len(x_grey),  # 假设每个矩形的宽度为1
            height=[0.8] * len(y_grey),  # 假设每个矩形的高度为1
            time=infolist_grey
        ))

        p.rect(x='x', y='y', width='width', height='height', source=source_grey,
               fill_color="#ececec",
               line_color=None)
        index = 0
        for key, value in rank_dict.items():
            temp_x = []
            temp_y = []
            for rank in value:
                temp_x.append(pos_dict[rank][0])
                temp_y.append(pos_dict[rank][1])

            p.line(temp_x, temp_y, line_width=5, line_dash='dashed', line_color=colors_line[index], legend_label=key)
            p.circle(temp_x, temp_y, size=30, color=colors_line[index], alpha=0.2)
            index += 1

        p.legend.location = "bottom_center"
        p.legend.orientation = "horizontal"

    p.title.align = "center"
    p.text(x='x', y='y', text='ids', text_baseline='middle', text_align='center', text_color='black',
           text_font_size="20px", source=source)
    hover = HoverTool(tooltips=[("computing", f"@computing"), ("communication", "@communication"),
                                ("communication_not_overlapped", "@communication_not_overlapped"),
                                ("free", "@free"),
                                ])
    p.add_tools(hover)

    # 显示图形
    output_file("topo.html")
    show(p)

def plot_single(rank_id=-1, metric="computing"):
    p = figure(title="网络拓扑图", sizing_mode = "stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset", )

    p.xgrid.grid_line_color = None  # 隐藏x轴的网格线
    p.ygrid.grid_line_color = None  # 隐藏y轴的网格线
    p.xaxis.axis_label_text_font_size = "0px"  # 隐藏x轴标签
    p.yaxis.axis_label_text_font_size = "0px"  # 隐藏y轴标签
    p.xaxis.visible = False  # 隐藏x轴刻度
    p.yaxis.visible = False  # 隐藏y轴刻度

    rank_dict = ranks_relation_dict.get(rank_id)

    has_lagend = False
    y = 2
    for key, value in rank_dict.items():
        x = 2
        for rank in value:
            rank = int(rank)
            source_single = ColumnDataSource(data=dict(
                id=[rank],
                x=[x],
                y=[y],
                width=[0.8],  # 假设每个矩形的宽度为1
                height=[0.8],  # 假设每个矩形的高度为1
                computing=[ranks_info_dict.get(rank)['computing']],
                communication=[ranks_info_dict.get(rank)['communication']],
                communication_not_overlapped=[ranks_info_dict.get(rank)['communication_not_overlapped']],
                free = [ranks_info_dict.get(rank)['free']],
            ))

            r = p.rect(x='x', y='y', width='width', height='height', source=source_single,
                       fill_color=linear_cmap(metric, colors, low=-0, high=2000000),
                       line_color=None)
            p.text(x='x', y='y', text='id', text_baseline='middle', text_align='center', text_color='black',
                   text_font_size="20px", source=source_single)
            if x != 2:
                p.line([x, x-2], [y, y], line_width=5, line_dash='dashed', line_color=colors_line[int((y/2)-1)], legend_label=key)

            if has_lagend is False:
                p.add_layout(r.construct_color_bar(
                    major_label_text_font_size="7px",
                    ticker=BasicTicker(desired_num_ticks=len(colors)),
                    formatter=PrintfTickFormatter(format="%d ms"),
                    label_standoff=6,
                    border_line_color=None,
                    padding=5,
                ), 'right')
                has_lagend = True
            if rank == rank_id:
                p.rect(x='x', y='y',  width='width', height='height', line_color="black",
                       fill_color=None, line_width=5, source=source_single,)
            x += 2
        y += 2

    p.legend.location = "bottom_center"
    p.legend.orientation = "horizontal"
    hover = HoverTool(tooltips=[("computing", f"@computing"), ("communication", "@communication"),
                                ("communication_not_overlapped", "@communication_not_overlapped"),
                                ("free", "@free"),
                                ])
    p.add_tools(hover)

    # 显示图形
    output_file("7.html")
    show(p)

    p.add_layout(r.construct_color_bar(
        major_label_text_font_size="7px",
        ticker=BasicTicker(desired_num_ticks=len(colors)),
        formatter=PrintfTickFormatter(format="%d ms"),
        label_standoff=6,
        border_line_color=None,
        padding=5,
    ), 'right')


def plot_histogram(metric, top_num=10):
    order_info = ranks_order_dict.get(metric)

    data = {
        'Category': order_info.keys(),
        'Value': order_info.values()
    }

    # 将数据转换为DataFrame并按Value排序
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Value', ascending=False)

    df_top = df_sorted.head(top_num)
    # 创建一个ColumnDataSource
    source = ColumnDataSource(df_top)
    # 创建直方图
    p = figure(x_range=source.data['Category'], title=f"{metric}耗时排序",
               sizing_mode="stretch_both",
               x_axis_label='Rank', y_axis_label='Time/us',
               tools="pan,wheel_zoom,box_zoom,reset")
    p.xgrid.grid_line_color = None  # 隐藏x轴的网格线
    p.ygrid.grid_line_color = None  # 隐藏y轴的网格线
    p.title.align = "center"

    # 绘制直方图
    p.vbar(x='Category', top='Value', width=0.7, source=source, color="#b7c886")

    p.text(x='Category', y='Value', text='Value', source=source, text_align="center", text_baseline="bottom",
           text_font_size="10pt", text_color="black")

    # 显示图表
    output_file("histogram.html")
    show(p)

def plot_histogram_single(metric, rank_id):

    histogram_data_source = ranks_relation_dict.get(rank_id)
    select = Select(title="想看啥并行关系:", value='source1',
                    options=list(histogram_data_source.keys()))
    hhh = {}
    for i in range(6):
        hhh[f"{i}"] = random.randint(0, 8)

    data_ori = {
        'Category': hhh.keys(),
        'Value': hhh.values(),
    }
    c = list(data_ori.keys())
    v = list(data_ori.values())

    # 创建一个ColumnDataSource
    source = ColumnDataSource(data=dict(
        Category=['1', '2'],
        Value=[1, 2]
    ))
    data = dict(Category=['1', '2'], Value=[1, 2])

    p = figure(x_range=source.data['Category'], title=f"{metric}耗时排序",
               x_axis_label='Rank', y_axis_label='Time/us',
               tools="pan,wheel_zoom,box_zoom,reset")
    p.xgrid.grid_line_color = None  # 隐藏x轴的网格线
    p.ygrid.grid_line_color = None  # 隐藏y轴的网格线
    p.title.align = "center"
    # 绘制直方图
    p.vbar(x='Category', top='Value', width=0.7, source=source, color="#b7c886")

    p.text(x='Category', y='Value', text='Value', source=source, text_align="center", text_baseline="bottom",
           text_font_size="10pt", text_color="black")

    callback = CustomJS(args=dict(
        histogram_data_source=histogram_data_source,
        select=select,
        metric=metric,
        source=source,
        ranks_info_dict=ranks_info_dict,
        plot=p), code="""
        // 获取选中的排名数据
        var selected = select.value
        var ranks = histogram_data_source[selected];
        
        // 创建一个临时对象，映射排名到对应的信息
        var temp = {};
        for (var i = 0; i < ranks.length; i++) {
            var rank = ranks[i];
            temp[rank] = ranks_info_dict.get(0);
        }
        
        // 创建一个包含度量信息的对象
        var order_info = {};
        for (var key in temp) {
            if (temp.hasOwnProperty(key)) {
                order_info[key] = temp[key][metric];
            }
        }
        
        // 创建数据对象
        var data = {
            'Category': Object.keys(order_info),
            'Value': Object.values(order_info)
        };
        
        // 将数据转换为数组并按Value排序
        var df = [data.Category, data.Value];
        var df_sorted = df.map(function(_, idx) { return {idx: idx, Category: df[0][idx], Value: df[1][idx]}; })
            .sort(function(a, b) { return b.Value - a.Value; }); // 降序排序
        
        // 更新数据源
        var sorted_data = {};
        sorted_data['Category'] = df_sorted.map(function(item) { return item.Category; });
        sorted_data['Value'] = df_sorted.map(function(item) { return item.Value; });
        
        var categories = sorted_data['Category'];
        var values = sorted_data['Value'];
        
        // 添加新的类别和值
        var new_category = '66';
        var new_value = 3;  // 假设的新值
        categories.push(new_category);
        values.push(new_value);
        
        // 更新 ColumnDataSource
        source.data = { 'Category': categories, 'Value': values };
        source.change.emit();  // 通知 Bokeh 数据已更改
        
        // 更新 x_range
        var new_x_range = categories;
        plot.x_range = new_x_range;
        

    """)

    select.js_on_change('value', callback)

    # 布局
    layout = row(column(p), select)

    output_file("histogram_single.html")
    show(layout)

def plot_histogram_single_with_relation(metric, rank_id, relation):
    histogram_data_source = ranks_relation_dict.get(rank_id)
    ranks = histogram_data_source.get(relation)
    temp = {rank: ranks_info_dict[int(rank)] for rank in ranks}
    order_info = {key: value[metric] for key, value in temp.items()}
    data = {
        'Category': order_info.keys(),
        'Value': order_info.values()
    }

    # 将数据转换为DataFrame并按Value排序
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Value', ascending=False)

    source = ColumnDataSource(df_sorted)
    # 创建直方图
    p = figure(x_range=source.data['Category'], title=f"{metric}耗时排序",
               sizing_mode="stretch_both",
               x_axis_label='Rank', y_axis_label='Time/us',
               tools="pan,wheel_zoom,box_zoom,reset")
    p.xgrid.grid_line_color = None  # 隐藏x轴的网格线
    p.ygrid.grid_line_color = None  # 隐藏y轴的网格线
    p.title.align = "center"

    # 绘制直方图
    p.vbar(x='Category', top='Value', width=0.7, source=source, color="#b7c886")

    p.text(x='Category', y='Value', text='Value', source=source, text_align="center", text_baseline="bottom",
           text_font_size="10pt", text_color="black")


    output_file("histogram_single.html")
    show(p)


def data_parse(rows, columns, output_path):
    get_rank_pos(rows, columns)
    get_ranks_relation(output_path)
    get_ranks_info(output_path)
    get_ranks_order(output_path)


def run_topo(metric, rank_id=-1, is_all=True):
    if is_all:
        plot(rank_id, metric)
    else:
        plot_single(rank_id, metric)


def run_histogram(metric, relation, rank_id=-1, is_all=True, top_num=10):
    if is_all:
        plot_histogram(metric, top_num)
    else:
        plot_histogram_single_with_relation(metric, rank_id, relation)