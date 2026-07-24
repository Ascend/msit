# msit tensor view功能使用指南

## 简介

tensor-view工具提供了查看tensor的接口，源数据是dump后生成的bin文件。对其进行链式切片、转置操作。默认每次都会打印统计信息和tensor.shape。可以选择打印tensor本身和保存到文件，文件格式可以选择标准torch格式和ATB格式（与dump生成的相同）

暂不支持Windows

## 工具安装

### 安装前置说明

- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [CANN](https://www.hiascend.com/cann/download)。
- 安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest。
- 若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。

### 安装方式说明

安装方式包括：**源代码安装**和**pip源安装**，用户可以按需选取。

- [源代码安装](#源代码安装): 使用源码安装，保证是最新的 msit 功能。
- [pip源安装](#pip源安装): pip 安装 msit 包，一个季度发包一次。

#### 源代码安装

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin 更新最新代码 
cd msit/msit

# 2. 安装 msit 包
pip install .

# 3. 安装 tensor-view：
msit install tensor-view

# 4. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check tensor-view
```

#### pip源安装

```shell
# 1. 安装 msit 包
pip install msit

# 2. 安装 tensor-view：
msit install tensor-view

# 3. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check tensor-view 
```

## 数据准备

tensor参数文件(.bin/.pth)路径，支持ATB格式（.bin）/Torch格式（.pth）

## 命令示例

```sh
# 输入ATB文件，应用连续的切片、转置、切片、转置，operations仅作演示之用，这个示例必定会失败，因为第一个转置不合规【0，2，1，4】
msit tensor-view --bin intensor0.bin --operations "[1:2, ..., 3::2];(0,2,1,4);[1:3];(2,0,1)" --output tmp_view/output_view.bin
```

```sh
# 输入PTH，输出ATB格式文件，使用缩略参数名
msit tensor-view -b rand.pth -op "[1];(2,0,1)" -o out/processed_rand.bin
```

## 参数说明

| 参数名               | 描述                                                                                                                                                                                                                                                       | 必选 |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| --bin, -b         | Tensor参数文件路径，支持ATB格式（.bin）/Torch格式（.pth）                                                                                                                                                                                                                 | 是  |
| --print, -p       | 是否在控制台打印Tensor，默认不打印。需要注意的是，打印操作在operations之后                                                                                                                                                                                                            | 否  |
| --operations, -op | Tensor切片和permute操作的字符串，需要使用**双引号**包裹，多个操作使用 **;** 分割，切片操作需要使用 **[ ]** 包裹，转置操作需要使用 **( )** 包裹，这些操作字符串会按顺序应用在Tensor上，执行过程中会检查切片、转置字符串是否valid，切片索引是否超出边界，转置字符串是否与维度对应。需要注意的是，切片支持的包括，**标准切片**、**数字索引**、**Ellipsis**，这三种形式在一个扩展切片中可以自由组合，不支持True/False等形式的切片 | 否  |
| --output, -o      | 经过处理的Tensor的存放路径，没有默认值，当文件扩展名为.bin时，保存为ATB格式，当文件扩展名为.pth时，使用torch.save，如果扩展名省略，保存格式会与【--bin】的格式保持一致                                                                                                                                                      | 否  |
| -h, --help        | 工具使用帮助信息                                                                                                                                                                                                                                                 | 否  |
