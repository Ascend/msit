
# 基于torch图模式（torchair）推理场景
- 注：在跑推理之前需要确认torchvision版本与torch版本是否匹配，torch2.1.0版本对应torchvision的版本为0.16.0
- torch版本与torchvision版本匹配表格：

  | torch版本 | torchvision版本 | 
  |---------|---------------|
  | 2.3.0   | 0.18.0        |
  | 2.2.0   | 0.17.0        |
  | 2.1.0   | 0.16.0        | 
  | 2.0.0   | 0.15.1        | 

## 1. GE融合模式（默认） dump 数据与 FX dump 数据精度比对

### Dump 数据

- **GE**: Graph Engine，基于昇腾AI软件栈对不同的机器学习框架提供统一的IR接口，对接上层网络模型框架。
- **FX**：功能类似于pytorch框架的FX工具包，用于消除动态图和静态图之间的gap，使我们对于nn.Model的各种操作变得更加简单。
- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `config` 实例，配置模型 compile，并执行推理。

  ```py
  import torch, torch_npu, torchair
  from msit_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- get_ge_dump_config 参数列表

  | 参数名                | 参数描述                                               | 是否必选                |
  |--------------------|----------------------------------------------------|---------------------|
  | dump_path          | dump数据的存放路径                                        | 否(默认为ait_ge_dump)                   |
  | dump_mode         | data dump模式，用于指定dump算子输入还是输出数据                     | 否                   |
  | fusion_switch_file | 是否关闭融合dump功能                                       | 否(默认为None，开启融合)    | 
  | dump_token         | 指定token进行dump,格式：[1,2,5],代表dump第1、2、5个token数据      | 否(默认为None，dump全量数据) |, 
  | dump_layer         | 指定layer进行dump，格式：["Add","Conv_1"],代表dump Add和Conv_1两层数据 | 否(默认为None，dump全量数据) | 


  **GE模式 [开启融合（默认） Dump 案例](TorchAir场景Dump案例.md)**
- **FX 模式 dump 数据** 添加 `get_fx_dump_config`，该配置与get_ge_dump_config的不同处在于，不能提供参数，如dump_path等。接下来配置 `config` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from msit_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_fx_dump_config()  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径与 torchair 版本相关，新版本中为当前文件夹下的 `data_dump/{token_id}/gm_{time stamp}_dump`，老版本中为 `gm_{time stamp}_dump`
  
  **其中 `{token_id}` 是从 1 开始的，相对于 GE 模式是从 0 开始的，比对时会将 FX 模式的 token_id 减 1**

  **FX模式 [Dump 案例](TorchAir场景Dump案例.md)**
### Compare 精度比对

- 执行 `msit llm compare --my-path [GE dump data] --golden-path [FX dump data]`，输出比对结果 csv 文件

  ```sh
  msit llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path data_dump
  ```

***

## 2. GE融合模式（默认）dump 数据与GE关闭融合模式 dump 数据精度比对

### Dump 数据

- **GE 模式（默认）开启融合 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `config` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from msit_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- **GE 模式关闭融合 dump 数据** 添加 `get_ge_dump_config`，指定 `fusion_switch_file` 文件，获取配置后的 `config` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from msit_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file="fusion_switch.json")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

- 参考的 `fusion_switch.json` 文件

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- fusion_switch相关参数

  | 参数名          | 参数描述                                                                                                                                         | 
  |--------------|----------------------------------------------------------------------------------------------------------------------------------------------|
  | GraphFusion  | 根据融合规则进行改图的过程，该过程主要通过拆分/合并计算图中的算子来提升计算效率，以实现加速运算的目的，与硬件无关                                                                                    
  | UBFusion     | 对图上算子进行硬件UB相关的融合，全称为：UnifiedBuffer。例如两个算子a和b单独运行时，算子a的计算结果在UB上，需要搬移到DDR（双倍速率同步动态随机存储器）。算子b在执行时，需要将算子a的输出由DDR再搬移到UB，进行算子b的计算逻辑，计算之后又从UB搬移回DDR 

  **GE模式 [关闭融合Dump 案例](TorchAir场景Dump案例.md)**

### Compare 精度比对

- 执行 `msit llm compare --my-path [GE dump data] --golden-path [fusion off GE dump data]`，输出比对结果 csv 文件

  ```sh
  msit llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path {dump_path}/dump_{time_stamp}
  ```
***

## 3. 结果查看

参考[精度比对结果参数说明](/msit/docs/llm/精度比对结果参数说明.md)

***

## (定向客户提供) 将 dump 数据转化为指定信息以压缩数据量
- dump 过程中生成的数据量可能占用大量磁盘空间，可以在 dump 过程中启用后台进程，将完整的数据提取为指定的信息。以下参考脚本将数据转化为最大最小值，并删除原数据
  ```py
  #!/bin/env python3
  import os
  import time
  import argparse

  surfix = "_min_max"  # Converted data save surfix

  # Define how single data is converted
  def convert_data_to_info(data):
      return [data.min(), data.max()]

  def convert(data_path):
      import numpy as np
      from components.utils.acc_cmp import parse_torchair_dump_data

      npz_surfix, npy_surfix = "{}.npz".format(surfix), "{}.npy".format(surfix)
      for cur_path, dirs, files in os.walk(data_path):
          for file in files:
              if file.endswith(npy_surfix):  # already converted FX data
                  continue

              cur = os.path.join(cur_path, file)
              if file.endswith(".npy"):  # FX saved npy data
                  file_name = os.path.splitext(cur)[0]
                  np.save(file_name + surfix, convert_data_to_info(np.load(cur)))
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, file_name, npy_surfix))
              elif not file.endswith(npz_surfix) and not file.endswith(".txt") and not file.endswith(".swp"):
                  inputs, outputs = parse_torchair_dump_data(cur)
                  inputs = [convert_data_to_info(ii) for ii in inputs]
                  outputs = [convert_data_to_info(ii) for ii in outputs]

                  np.savez(cur + npz_surfix, inputs=inputs, outputs=outputs)
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, cur, npz_surfix))
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("data_path", help="GE or FX data dump path")
      args = parser.parse_args()
      while True:
          convert(args.data_path)
          time.sleep(0.5)
          print("Wmsiting...")
  ```
  在 dump 过程中后台执行该脚本，将 dump 数据转化为 info 数据，以减少内存占用
  ```sh
  # 将 msit_ge_dump 下的 GE dump 数据转化为 info
  python3 convert.py msit_ge_dump
  ```
  同时在 FX 或关闭融合的 GE dump 时也执行该脚本，将 dump 数据转化为 info。
  ```sh
  # 将 msit_ge_dump 下的 FX dump 数据转化为 info
  python3 convert.py data_dump
  ```
  最后执行比对
  ```sh
  msit llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path data_dump
  ```
  