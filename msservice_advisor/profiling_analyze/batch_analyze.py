import sys

sys.path.append("..")
from profiling_analyze.register import register_analyze, cached, answer
import numpy as np
import matplotlib.pyplot as plt
import random


def summary_batch_info(batch_info):
    summary = {}
    for batchsize, latency_list in batch_info.items():
        summary[batchsize] = {}
        latency_list.sort()

        summary[batchsize]["BSZ"] = batchsize
        summary[batchsize]["MIN"] = latency_list[0]
        summary[batchsize]["P10"] = latency_list[int(len(latency_list) * 0.1)]
        summary[batchsize]["P30"] = latency_list[int(len(latency_list) * 0.3)]
        summary[batchsize]["P50"] = latency_list[int(len(latency_list) * 0.5)]
        summary[batchsize]["P70"] = latency_list[int(len(latency_list) * 0.7)]
        summary[batchsize]["P90"] = latency_list[int(len(latency_list) * 0.9)]
        summary[batchsize]["MAX"] = latency_list[-1]
        summary[batchsize]["FIT_DATA"] = latency_list[int(len(latency_list) * 0.3): int(len(latency_list) * 0.7)]
    return summary


def print_list(array):
    for item in array:
        print(item)


@cached()
def read_batch_and_latency(pre_request):
    # 1. 获取所有的batchsize 对应的 latency
    # 2. 计算P50
    # 3. 打印出来
    prefill_batch_info = {}  # batchsize => list of latency
    decode_batch_info = {}  # batchsize => list of latency
    for request in pre_request.values():
        prefill_batch_size = request.get("prefill_bsz")
        decode_batch_size_list = request.get("decode_bsz")
        latency_list = request.get("latency")
        for index, latency in enumerate(latency_list):
            if index == 0:
                prefill_batch_info.setdefault(prefill_batch_size, [])
                prefill_batch_info[prefill_batch_size].append(latency)
            else:
                decode_batch_size = decode_batch_size_list[index - 1]
                decode_batch_info.setdefault(decode_batch_size, [])
                decode_batch_info[decode_batch_size].append(latency)

    prefill_summary = summary_batch_info(prefill_batch_info)  # batchsize => P50
    decode_summary = summary_batch_info(decode_batch_info)  # batchsize => P50

    return prefill_summary, decode_summary


def find_best_by_bayes(summary_P50):
    from bayes_opt import BayesianOptimization

    max_batch_size = summary_P50[-1]["BSZ"]

    # 定义参数搜索空间
    pbounds = {'BSZ': (0, max_batch_size * 2)}

    # 定义目标函数（占位符，实际使用已有数据）
    def target_function(bsz):
        # 此函数不会被实际调用，仅用于初始化
        return bsz * 1

    # 初始化贝叶斯优化器
    optimizer = BayesianOptimization(f=target_function, pbounds=pbounds, verbose=2, random_state=1)

    # 将已有数据转换为库需要的格式（字典列表）
    existing_points = [{"BSZ": x["BSZ"]} for x in summary_P50]
    existing_target = [x["BSZ"] / x["P50"] for x in summary_P50]

    # 将已有数据添加到优化器中
    for x, y in zip(existing_points, existing_target):
        optimizer.register(params=x, target=y)

    # 确保高斯过程模型已拟合数据
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)

    # ----------- 可视化分析 -----------
    # 生成单变量网格点用于模型预测
    x_values = np.linspace(0, max_batch_size * 2, 300).reshape(-1, 1)
    mu, sigma = optimizer._gp.predict(x_values, return_std=True)

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制模型预测均值和置信区间
    plt.plot(x_values, mu, label='Model Prediction (Mean)', color='blue')
    plt.fill_between(
        x_values.ravel(),
        mu - 1.96 * sigma,  # 95% 置信区间
        mu + 1.96 * sigma,
        alpha=0.2,
        color='blue',
        label='95% Confidence Interval',
    )

    # 绘制已有数据点
    plt.scatter(
        [x['BSZ'] for x in existing_points],
        existing_target,
        c='green',
        s=100,
        edgecolor='black',
        label='Existing Data Points',
    )

    # 标记模型预测的最优点
    best_x = optimizer.max['params']['BSZ']
    best_y = optimizer.max['target']
    plt.scatter(best_x, best_y, marker='*', s=200, color='red', label='Predicted Best Point')

    # 图表标注
    plt.title('Bayesian Optimization (1D Input)')
    plt.xlabel('x')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    best_predicted = optimizer.max
    print(f"Best bayes value: {best_predicted}")
    return int(best_predicted.get("params", {}).get("BSZ"))


def find_best_by_curve_fit(summary_fit_data, process_name):
    from scipy.optimize import curve_fit, minimize

    max_batch_size = summary_fit_data[-1]["BSZ"]
    print(process_name, "上次运行组的最大的 batch size 为", max_batch_size)

    if len(summary_fit_data) > 2:
        def func_curv(x, a, b, c):
            return a * x**b * np.exp(-c / x)

    else:
        def func_curv(x, a, b):
            return a * x + b

    points = []
    targets = []
    
    for x in summary_fit_data:
        bsz = x["BSZ"]
        for latency in x["FIT_DATA"]:
            points.append(bsz)
            targets.append(bsz * 1000 / latency)
            
    points.append(max_batch_size * 10)
    targets.append(0.00001)
    
    popt, pcov = curve_fit(func_curv, points, targets, maxfev=10000)
    print(process_name, "函数拟合后参数：", popt)

    # 或者使用数值优化（通用方法，适用于任何模型）
    def negative_func(x):
        return -func_curv(x, *popt)  # 最小化负函数即最大化原函数

    best_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 2)])
    aggressive_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 5)])
    print(process_name, f"搜索范围 2 倍当前最大batchsize. 结果是: {best_predicted.x[0]} {best_predicted}")
    print(process_name, f"搜索范围 5 倍当前最大batchsize. 结果是:  {aggressive_predicted.x[0]} {aggressive_predicted}")

    # 开始画图
    x_values = np.linspace(0, max_batch_size * 5, 300)

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制模型预测均值和置信区间
    plt.plot(x_values, func_curv(x_values, *popt), label=f'Model Prediction', color='blue')
    plt.scatter(points, targets, c='green', s=100, edgecolor='black', label='Existing Data Points')

    plt.title(f'Curve Fit Optimization({process_name})')
    plt.xlabel('Batch Size')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 生成一个指定范围内的随机整数
    png_name = f'func_curv_{process_name}.png'
    print(process_name, "拟合画图路径：", png_name)
    plt.savefig(png_name)
    plt.close()

    return int(best_predicted.x[0])



@register_analyze()
def find_best_batch_size(config, benchmark, output_log, limit, target_metrics):
    if "results_per_request" not in benchmark:
        return

    prefill_summary, decode_summary = read_batch_and_latency(benchmark.get("results_per_request", {}))
    
    def divide_fit_and_print(summary):
        summary.sort(key=lambda x: x["BSZ"])
        to_fit = [dict(BSZ=x["BSZ"], FIT_DATA=x.pop("FIT_DATA")) for x in summary]
        return to_fit, summary
        
    prefill_to_fit, prefill_to_print = divide_fit_and_print(list(prefill_summary.values()))
    decode_to_fit, decode_to_print = divide_fit_and_print(list(decode_summary.values()))

    print("==decode==")
    print_list(decode_to_print)
    print("==prefill==")
    print_list(prefill_to_print)
    
    if len(decode_to_fit) <= 1:
        answer(config="maxBatchSize", action=f"set bigger", reason="目前batch样本太小，建议调大点试试")

    if len(prefill_to_fit) <= 1:
        answer(config="maxPrefillBatchSize", action=f"set bigger", reason="目前batch样本太小，建议调大点试试")
    
    if len(decode_to_fit) > 1:
        best_decode_batchsize = find_best_by_curve_fit(decode_to_fit, "decode")
        answer(
            config="maxBatchSize",
            action=f"set to {best_decode_batchsize}",
            reason="经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize",
        )

    

    if len(prefill_to_fit) > 1:
        best_prefill_batchsize = find_best_by_curve_fit(prefill_to_fit, "prefill")
    
    
        answer(
            config="maxPrefillBatchSize",
            action=f"set to {best_prefill_batchsize}",
            reason="经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize",
        )


if __name__ == "__main__":
    from profiling_analyze.register import print_answer

    print("<think>")
    find_best_batch_size(
        None,
        dict(
            results_per_request=dict(
                ee=dict(
                    input_len=132,
                    output_len=12,
                    prefill_bsz=15,
                    decode_bsz=[2, 2, 1, 2, 4, 4, 7, 7, 67, 8, 20, 9, 20, 4],
                    latency=[234, 234, 456, 34, 54, 4457, 5, 678, 67, 45, 645, 76, 8, 345645, 5467],
                ),
                ee2=dict(
                    input_len=132,
                    output_len=12,
                    prefill_bsz=15,
                    decode_bsz=[2, 2, 1, 2, 4, 4, 7, 7, 67, 8, 20, 9, 20, 4, 5, 6, 7, 8, 9, 4, 5],
                    latency=[224, 2654, 476, 4565, 756, 7, 6, 8, 34, 5, 2, 5, 4, 75, 634, 5, 34, 54, 56, 54634, 534, 6],
                ),
            ),
        ),
        None,
        None,
        None,
    )
    print("</think>")
    print_answer()
