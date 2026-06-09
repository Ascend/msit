# 特性设计

## 功能描述

MindStudio 预检工具 msprechecker 在文件读写、目录遍历及命令行参数校验等环节广泛依赖 msguard 安全库。该库按 SSHD 与数据库核心进程规范实现属主一致性、权限位、软链接拒绝及固定长度上限等预检查逻辑。在 Docker 容器、NFS 共享挂载、多用户协作集群及跨 UID 映射等典型 AI 开发场景中，上述校验与真实威胁模型不匹配，导致预检、落盘、规则执行等流程在文件本身可读可写时仍被提前拦截，并引入递归权限修正与全量路径规则遍历等额外开销。

本次改造将 msprechecker 的文件与路径安全策略与 MindStudio 工具链统一原则对齐，在保留读取侧存在性与大小等底线校验的前提下，移除冗余输入限制并脱离 msguard 依赖；工具自行创建的输出文件与目录遵循当前进程 umask。具体功能点如下。

1. 移除读取侧文件属主校验。所有原通过 msguard `open_s` 打开的路径不再校验 inode 属主是否与当前进程用户一致，能否读取完全交由操作系统权限机制决定。
2. 移除读取侧文件权限位校验。不再因组或其他用户可写、权限位不符合固定模板而拒绝读取用户配置文件、权重文件及系统信息文件；不再在读取前递归修正第三方文件或目录权限。
3. 移除输入路径软链接拦截。用户显式传入的配置文件路径、权重目录路径等若为软链接，不再在打开前拒绝；符号链接由内核解析，工具仅对解析后的目标执行存在性与大小检查。目录递归遍历时不得跟随软链接，须跳过链接目录或文件以避免循环引用，该约束与输入路径放行策略相互独立。
4. 放宽路径与环境变量长度限制。命令行参数及配置文件路径不再使用 msguard 内置固定上限，读取侧不对长度做额外校验；若工具代为创建文件或环境变量且长度超出操作系统 `PATH_MAX` 或 `ARG_MAX`，则按 EAFP 方式捕获系统错误并返回明确失败信息。
5. 替换 msguard 文件 API 并统一采用 pathlib。约 15 处 `open_s` 调用改为 `pathlib.Path` 的 `open`、`read_text`、`write_text` 等方法；权重采集模块的 `walk_s` 改为 `path_io.iter_regular_files` 栈式递归遍历，子项跳过软链接且允许输入根路径本身为软链接，并在每个普通文件上校验大小不超过 10 GiB。全改造过程禁止新增 `os.path` 调用，路径在 CLI 边界处一次性转换为 `Path` 后向内传递。
6. 替换 msguard 参数校验 API。`validate_args(Rule.input_file_read)` 改为 `path_io.readable_file` 或 `as_arg_type` 组合；`Rule.input_file_exec.is_satisfied_by` 改为模块级一次性 `os.access` 判定。用户输入路径在 argparse 或 Coordinator 入口完成 normalize 与校验，下游不再重复判断。
7. 输出侧权限由用户环境决定。工具自行写入的落盘 JSON 及 precheck 流程生成的 `msprechecker_env.sh` 等输出文件，以及创建过程中涉及的输出目录，均遵循当前进程 umask 创建。该策略与读取侧一致：权限由用户及操作系统环境决定，工具不修改用户已有文件的权限，也不对读取侧输入文件做权限位预检。
8. 移除 msguard 依赖。从 `pyproject.toml` 依赖列表删除 `msguard` 包，同步清理全部 import 语句及相关测试 mock 路径。
9. 用户文档约束说明。README 建议非 root 用户安装前执行 `umask 0027`；说明读取侧与输出侧权限均由用户及管理员自行管理，工具不校验权限位、属主一致性及软链接安全性。

对用户而言，在 Docker 容器内挂载异构 UID 权重目录、通过 NFS 读取共享模型文件、以 root 或非属主身份访问配置文件等场景下，msprechecker 的 precheck、dump、compare、run、inspect 等子命令不再因属主或权限位预检失败而中断；权重哈希采集、规则文件加载等操作的启动延迟预计降低，因不再执行 msguard 附带的权限与软链接全量扫描；工具生成的落盘文件与脚本权限遵循当前进程 umask，用户可通过安装前设置 `umask 0027` 等方式自行管控输出暴露面。对系统而言，依赖项减少一项第三方安全库，文件访问路径统一为 pathlib 与 EAFP 模式，与 Python 编码规范及 MindStudio 工具链安全策略一致；在验收所要求的 Docker、NFS、共享集群多用户及 root 运行场景测试中，功能用例中断率目标为零，且代码大模型扫描与回归测试均不应报告新增注入、路径遍历等安全问题。

## 实现思路

改造按依赖顺序分六步推进：先在入口层集中完成路径规范化与校验收口，再向下游传递已解析的 `Path` 对象，随后替换读取与遍历实现，统一输出侧写入行为，最后清理依赖并同步测试与文档。核心约束为：用户输入路径仅在 CLI 或 Coordinator 边界校验一次并完成 `expanduser` 与 `resolve`；下游模块信任入参类型，不再重复 `is_file` 或权限位判断；argparse 的 `type` 通过可组合校验器表达多条件，避免为每种组合单独写函数。

### 第 1 步：新增路径规范化与可组合校验模块

**文字描述：** 在 `msprechecker/utils/path_io.py` 中定义路径处理的唯一入口逻辑。`normalize_user_path` 负责将用户输入字符串转为规范化的绝对真实路径：先 `Path.expanduser` 展开波浪号与用户主目录，再 `Path.resolve` 解析符号链接并消除 `..` 分量，返回可直接使用的 `Path`。校验器采用轻量组合：`check` 将具名谓词或 `functools.partial` 绑定后的谓词包装为 `PathCheck`；`as_arg_type` 串联 normalize 与若干 `PathCheck`，供 argparse 的 `type` 直接使用。预置 `readable_file`、`existing_dir` 两个常用组合；其余场景在调用处写 `as_arg_type(is_file, has_suffix(".txt"))` 即可，无需为每种组合新增模块级函数。实现遵循 Python 编码规范，谓词优先具名函数与 `partial`，不使用 lambda；模块不引入 Protocol、注册表等额外抽象层。

**落地示例：**

```python
# msprechecker/utils/path_io.py
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

PathCheck = Callable[[Path], Path]

DEFAULT_MAX_FILE_BYTES = 10 * 1024 ** 3


def normalize_user_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _path_is_file(path: Path) -> bool:
    return path.is_file()


def _path_is_dir(path: Path) -> bool:
    return path.is_dir()


def _path_access(path: Path, mode: int) -> bool:
    return os.access(path, mode)


def _path_has_suffix(path: Path, suffix: str) -> bool:
    return path.suffix == suffix


def check(predicate: Callable[[Path], bool], message: str) -> PathCheck:
    def _check(path: Path) -> Path:
        if not predicate(path):
            raise argparse.ArgumentTypeError(message.format(path=path))
        return path
    return _check


def as_arg_type(*checks: PathCheck) -> Callable[[str], Path]:
    def _parse(value: str) -> Path:
        path = normalize_user_path(value)
        for fn in checks:
            path = fn(path)
        return path
    return _parse


is_file = check(_path_is_file, "{path!r} is not a file")
is_dir = check(_path_is_dir, "{path!r} is not a directory")
is_readable = check(partial(_path_access, mode=os.R_OK), "{path!r} is not readable")

readable_file = as_arg_type(is_file, is_readable)
existing_dir = as_arg_type(is_dir)


def has_suffix(suffix: str) -> PathCheck:
    message = f"{{path!r}} must end with {suffix!r}"
    return check(partial(_path_has_suffix, suffix=suffix), message)


def iter_regular_files(root: Path, *, suffix: str = "", max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Iterator[Path]:
    """root 须为入口已 normalize 的 Path；允许 root 为软链接目录，子项不跟随软链接。"""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            if not current.is_dir():
                continue
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file() and entry.suffix == suffix and entry.stat().st_size <= max_bytes:
                    yield entry
            except OSError:
                continue
```

同步在 `msprechecker/utils/__init__.py` 导出 `normalize_user_path`、`as_arg_type`、`readable_file`、`existing_dir`、`has_suffix`、`iter_regular_files`。

**补充说明：** 仅需 normalize、无需存在性校验的参数可直接写 `type=normalize_user_path`，不必经过 `as_arg_type`。内部固定路径如 `/proc/cpuinfo` 在模块中以 `Path` 常量定义，不经过 `normalize_user_path`。手动拼接相对路径时，须对拼接结果再次 `resolve`，必要时以 `relative_to` 确认未逃逸根目录。

### 第 2 步：CLI 入口统一挂载校验并向下游传递 Resolved Path

**文字描述：** 所有用户可见的路径类 argparse 参数在本步一次性接入第 1 步的 `as_arg_type` 或预置组合，完成规范化与存在性校验。涉及 `commands/precheck.py`、`commands/dump.py`、`commands/compare.py`、`commands/_cmate.py`、`commands/legacy.py` 及 `cmate/cmate.py` 中仍使用 `validate_args` 的 argparse 定义，覆盖配置文件路径、权重目录、规则文件、落盘输出路径、rank table 路径等。`Coordinator.execute`、Dump 策略及 RunStrategy 从 `args` 取出的路径字段类型均为已 `resolve` 的 `Path`，向 collector、checker、cmate 引擎传递时不再调用 `Path()` 或 `is_file`。`--configs` 等复合字符串参数在 Coordinator 或 Dump 层解析出路径分量后，对每个路径分量调用一次 `readable_file`，解析结果以 `Path` 写入内部数据结构。

**落地示例：**

```python
# msprechecker/commands/precheck.py
from ..utils.path_io import readable_file, existing_dir

group.add_argument(
    "--mies-config-path",
    type=readable_file,
    help="Path to MindIE service config.json",
)
group.add_argument(
    "--weight-dir",
    type=existing_dir,
    help="Directory containing model weight files",
)
```

```python
# msprechecker/commands/_cmate.py
from ..utils.path_io import readable_file

run_parser.add_argument("rule", type=readable_file, help="...")
```

```python
# msprechecker/cmate/cmate.py — 独立 cmate 入口 argparse 同步改造
from ..utils.path_io import readable_file, normalize_user_path

run_parser.add_argument("rule", type=readable_file, help="...")
inspect_parser.add_argument("rule", type=readable_file, help="...")
run_parser.add_argument("--output-path", type=normalize_user_path, help="...")
# 落盘目录在写入侧创建，不在 argparse 层 mkdir（见第 5 步 _actual_run）
```

```python
# msprechecker/cmate/cmate.py — 落盘时创建输出目录
output_dir = Path(output_path)
output_dir.mkdir(parents=True, exist_ok=True)
saved_json = output_dir / msprechecker_output_name
with saved_json.open("w", encoding="utf-8") as f:
    json.dump(msprechecker_output, f, ...)
```

```python
# msprechecker/commands/coordinator.py — configs 复合参数路径分量收口
from pathlib import Path
from typing import Tuple

from ..utils.path_io import readable_file

def _parse_config_entry(entry: str) -> Tuple[str, Path]:
    name, _, raw_path = entry.partition(":")
    path = readable_file(raw_path.split("@", 1)[0])
    return name, path
```

**补充说明：** 同一 `args` 字段若被多个 handler 读取，校验仍只在 argparse 层发生一次；handler 之间传递 `Path` 引用，禁止在中间层再次包装或校验。

第 1 步产出的组合校验器在本步全部消费，后续模块仅接收规范化后的 `Path`。

### 第 3 步：下游模块去除冗余校验，直接 EAFP 读写

**文字描述：** 在 12 个源文件中删除 `open_s` 及一切对用户输入路径的重复存在性判断，含 `cmate/cmate.py` 中规则读取与落盘写入路径。collector、checker、cmate 等模块的构造函数参数类型标注为 `Path`，实现体内直接 `path.open` 或 `read_text`；打开失败时由现有 `error_handler` 捕获 `OSError`，不在 open 前再次调用 `is_file`。类级系统路径常量保持硬编码 `Path` 对象，不经过用户路径规范化流程。`presets/manager.py`、`utils/ascend.py` 等间接接收路径的模块，假定上游已完成收口，移除内部的二次转换。

**落地示例：**

```python
# msprechecker/collectors/config.py — 接收已校验 Path，不再判断 is_file
class ConfigCollector(BaseCollector):
    def __init__(self, error_handler=None, *, config_path: Path):
        super().__init__(error_handler)
        self.config_path = config_path  # 已是 resolve 后的 Path

    def _collect_data(self):
        with self.config_path.open(encoding="utf-8") as f:
            ...
```

```python
# msprechecker/presets/manager.py — 规则路径由 RuleManager 入口传入，不再 normalize
def _load_rule_file(self, rule_path: Path) -> dict:
    with rule_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
```

**补充说明：** 读取失败时沿用各模块现有错误处理，异常类型为标准 `OSError` 与 `PermissionError`，不在捕获前做额外推断。

第 2 步保证进入本模块的路径均已规范化，本步只做 IO 替换，不再引入任何路径校验逻辑。

### 第 4 步：改造 WeightCollector 遍历与内部命令路径一次性判定

**文字描述：** `collectors/weight.py` 移除 msguard 的 `walk_s` 与规则表达式。`weight_dir` 由 CLI 层以 `existing_dir` 校验并传入，collector 内不再调用 `Path()` 或 `is_dir`。`_get_tensor_files` 仅调用 `iter_regular_files` 做后缀过滤与 10 GiB 大小约束，遍历时不跟随软链接。`collectors/network.py` 与 `collectors/hccl.py` 中的 `/usr/bin/ping` 与 `hccn_tool` 为工具内置固定路径，在模块级常量处通过 `Path.resolve` 解析一次，并以模块级布尔标志缓存可执行性判定结果，`_collect_data` 内不再重复检查。

**落地示例：**

```python
# msprechecker/collectors/weight.py
class WeightCollector(BaseCollector):
    def __init__(self, error_handler=None, *, weight_dir: Path, chunk_size=None):
        super().__init__(error_handler)
        self.weight_dir = weight_dir  # CLI 已校验的 existing_dir

    def _get_tensor_files(self, tensor_suffix: str):
        return list(iter_regular_files(
            self.weight_dir, suffix=tensor_suffix, max_bytes=DEFAULT_MAX_FILE_BYTES,
        ))
```

```python
# msprechecker/collectors/network.py — 内置命令路径模块级一次性判定
PING_CMD = Path("/usr/bin/ping").resolve()
_PING_AVAILABLE = PING_CMD.is_file() and os.access(PING_CMD, os.X_OK)

class PingCollector(BaseCollector):
    def __init__(self, ...):
        self._ping_cmd = None if not _PING_AVAILABLE else f"{PING_CMD} -c 3 -q -W 2 {{}}"
```

**补充说明：** 内置命令路径不经过 `expanduser`，仅 `resolve` 消除潜在符号链接；与用户输入路径的处理路径相互独立，避免混淆。

第 3 步完成的 EAFP 读取为本步提供无冗余校验前提，本步完成后 msguard 在业务路径上完全退出。

### 第 5 步：统一输出侧写入行为

**文字描述：** 工具主动创建的文件遵循当前进程 umask。用户指定的输出路径若在 CLI 层传入，须先经 `normalize_user_path` 规范化后再写入。落盘 JSON 由 `cmate/cmate.py` 写入，环境脚本由 `reporters/strategy.py` 写入。若父目录不存在，使用 `mkdir(parents=True, exist_ok=True)` 创建目录链。输出路径在 Coordinator 或 argparse 层完成 normalize 后向下传递，写入模块不再重复 resolve。

**落地示例：**

```python
# msprechecker/cli.py
def main():
    if os.geteuid() == 0:
        global_logger.warning(
            "WARNING: Running as root is not suggested.\n\n"
            "This may lead to unexpected privilege escalation and system modifications."
        )
    ...
```

```python
# msprechecker/reporters/strategy.py
OUTPUT_ENV_SCRIPT = Path("msprechecker_env.sh").resolve()

class EnvErrorDisplay(ErrorDisplayStrategy):
    def display(self, error_handler):
        with OUTPUT_ENV_SCRIPT.open("w", encoding="utf-8") as f:
            f.write(script_content)
```

```python
# msprechecker/cmate/cmate.py
def _write_output(saved_json: Path, payload: dict) -> None:
    saved_json.parent.mkdir(parents=True, exist_ok=True)
    with saved_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ...)
```

**补充说明：** 读取侧 open 不受 umask 影响；输出文件与目录权限由用户 shell 环境 umask 决定，README 建议非 root 用户安装前执行 `umask 0027`。

第 4 步清除 msguard 后，本步完成读写策略分离。

### 第 6 步：移除 msguard 依赖并同步测试与文档

**文字描述：** 确认全仓库无 msguard import 后，从 `pyproject.toml` 删除 `"msguard"`。测试中将 patch `open_s` 的用例改为 `tmp_path` 夹具配合 `normalize_user_path` 构造输入，或直接向被测函数传入已 resolve 的 `Path`。新增 `tests/test_utils/test_path_io.py` 覆盖 `as_arg_type`、`has_suffix` 及 `normalize_user_path` 对 `~` 与软链接输入路径的解析。README 载明非 root 用户安装前建议 `umask 0027`、入口校验一次、下游信任 Path，以及工具不校验权限位、属主一致性及软链接安全性。

**落地示例：**

```python
# tests/test_utils/test_path_io.py
def test_readable_file_rejects_missing(tmp_path):
    missing = tmp_path / "nope.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=readable_file)
    with pytest.raises(SystemExit):
        parser.parse_args(["--cfg", str(missing)])

def test_has_suffix_compose_ok(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    custom = as_arg_type(is_file, has_suffix(".txt"))
    assert custom(str(f)) == f.resolve()
```

```toml
# pyproject.toml — dependencies 删除 msguard
dependencies = ["pyyaml", "psutil", "ply", "colorama", "packaging", ...]
```

**补充说明：** 改造完成后执行全量 pytest 与 Docker、NFS、共享目录场景验证，确认功能中断率为零。

第 5 步输出侧写入落地后，本步完成依赖清理与质量闭环，整体改造至此交付。

### 逻辑流程图

本图描述安全策略简化改造后 msprechecker 的运行时主流程，涵盖用户从命令行触发到结果输出或落盘的完整链路，以及路径校验与文件访问环节的分支行为。

![image.png](https://raw.gitcode.com/user-images/assets/8428112/0b8b5eda-3217-4c12-8038-bf4eea5ec1d6/image.png 'image.png')

正常路径下，用户通过 shell 调用 `msprechecker precheck`、`dump`、`compare`、`run` 或 `inspect` 之一后，进程入口保留 root 运行告警（若适用），随后 argparse 对所有路径类参数执行 `expanduser` 与 `resolve`，并按参数语义调用 `readable_file`、`existing_dir` 或 `normalize_user_path` 完成唯一一次入口校验。校验通过后，`Coordinator` 将已规范化的 `Path` 对象传递给对应策略：precheck 与 dump 路径依次驱动 collector 采集、checker 校验与 reporter 报告；其中配置文件与系统信息由 collector 直接 `Path.open` 读取，权重目录由 `iter_regular_files` 在后缀与大小约束下递归扫描且跳过软链接子项；compare 路径直接打开多个 dump JSON 并交由 `Comparator` 输出差异；run 与 inspect 路径加载规则文件后分别执行校验或格式化展示。若 precheck 发现环境变量问题，或 dump 与 run 指定输出位置，工具按当前进程 umask 写入 `msprechecker_env.sh` 或 JSON 落盘文件及目录链。

异常路径方面，入口路径校验失败时 argparse 抛出 `ArgumentTypeError`，进程以非零退出码终止，不进入任何 collector 或 cmate 逻辑。collector 在 `Path.open` 阶段若遭遇 `OSError` 或 `PermissionError`，由现有 `error_handler` 记录错误并纳入 reporter 输出，不因 msguard 式预检提前中断整个流程。权重目录经入口校验存在但内部无符合后缀与大小约束的文件时，`WeightCollector` 记录采集错误并返回空结果，后续 checker 按既有逻辑处理。内置命令路径不可执行时，`PingCollector` 与 `HCCNCollector` 在初始化阶段已缓存不可用状态，采集环节输出明确错误而非重复判定。输出路径父目录不可创建或磁盘不可写时，写入操作捕获 `OSError` 并向用户返回失败信息。

### 时序图

本图描述改造后用户以不同子命令触发预检时，各组件之间按时间顺序发生的交互，重点呈现路径入口校验一次、下游直接消费已 resolve 的 `Path` 对象，以及读取与写入环节与操作系统之间的调用关系。

![image.png](https://raw.gitcode.com/user-images/assets/8428112/414bd417-bd7d-4a07-8cfa-daea37405bcb/image.png 'image.png')

正常路径下，用户发起 precheck 时，cli 入口依次调用 path_io 对配置文件执行 `readable_file`、对权重目录执行 `existing_dir`，两次校验均完成 normalize 后立即返回 resolve 后的 `Path`，Coordinator 将其交给 PrecheckStrategy 而不再二次校验。PrecheckStrategy 创建 collector 列表后，WeightCollector 通过 `iter_regular_files` 向文件系统请求目录内容且跳过软链接子项，其余 collector 直接 `Path.open` 读取；采集结果依次流经 checker 与 reporter 并在终端呈现。若 reporter 检测到环境变量问题，则按当前进程 umask 写入 `msprechecker_env.sh` 并提示用户 source。

用户发起 dump 时，输出路径仅经 `normalize_user_path` 规范化，采集链路同样直接 open 读取，最终由 dump 策略按当前进程 umask 创建目录链并写入 JSON。用户发起 compare 时，多个 dump 文件在 argparse 层各校验一次，CompareStrategy 直接 open 读取并驱动 reporter 输出差异。用户发起 run 时，规则文件在 argparse 层校验，复合 configs 参数中的路径分量在 RunStrategy 解析阶段各校验一次，随后 cmate 引擎 open 读取并执行规则，可选 output_path 按当前进程 umask 落盘。

异常路径下，入口 readable_file 探测时若文件系统返回不存在或不可读，path_io 向 argparse 抛出 `ArgumentTypeError`，cli 向用户输出参数错误并以非零码退出，Coordinator 及以下组件均不被调用。collector 或 cmate 在 open 阶段遭遇权限或 IO 错误时，错误沿既有 error_handler 上报至 reporter，不触发 msguard 式前置拦截。

### 代码结构设计

本类图聚焦本次新增的路径 IO 模块及与其直接发生依赖变更的类，展示改造后路径校验收口与读写职责的划分关系。

![image.png](https://raw.gitcode.com/user-images/assets/8428112/875e172e-b07b-4353-972b-dc21042eccbd/image.png 'image.png')

`path_io` 模块作为路径处理的单一职责单元，对外暴露 normalize、组合校验与目录遍历三类能力，不引入类层次。`cli` 与各 parser 通过 `readable_file`、`existing_dir` 等 callable 挂载 argparse 校验；`Coordinator` 及各类 Strategy 仅传递 `Path` 对象。`WeightCollector` 是唯一仍做文件级过滤的模块，但仅执行后缀与大小约束，不再重复入口存在性校验。`ConfigCollector` 等读取类 collector 构造函数接收 `Path` 后直接 open。`EnvErrorDisplay` 写入环境脚本时遵循当前进程 umask。其余 collector、checker、Reporter、Comparator、cmate 解析器等既有类结构不变，仅 import 来源由 msguard 改为标准库 pathlib 与 path_io。

### 接口设计

#### 对外接口

下表描述用户可通过命令行触达的、与本次改造相关的参数接口；均在 argparse 解析阶段完成唯一一次路径校验。

| 参数 | 可选/必选 | 说明 |
|------|-----------|------|
| `--mies-config-path` | 可选 | MindIE 配置文件路径。经 `readable_file` 校验：expanduser、resolve 为绝对真实路径，须为可读普通文件。不存在或不可读时抛出 `ArgumentTypeError`，进程退出。 |
| `--user-config-path` | 可选 | PD 分离场景 user_config.json 路径。校验规则同 `readable_file`。 |
| `--mindie-env-path` | 可选 | PD 分离场景 mindie_env.json 路径。校验规则同 `readable_file`。 |
| `--custom-config-path` | 可选 | 自定义校验规则 YAML 路径。校验规则同 `readable_file`。 |
| `--config-parent-dir` | 可选 | PD 分离配置根目录。经 `existing_dir` 校验；其下相对路径拼接后须 resolve 并 relative_to 防逃逸。 |
| `--weight-dir` | 可选 | 模型权重目录。经 `existing_dir` 校验：须为已存在目录，允许目录本身为软链接。下游遍历时不跟随子级软链接。 |
| `--rank-table` 及同类 rank table 路径 | 可选 | 多机拓扑 JSON 路径。经 `readable_file` 校验。 |
| `--output-path` | 可选 | dump 或 run 落盘路径。经 `normalize_user_path` 规范化，不强制文件预先存在。写入时遵循当前进程 umask。 |
| `compare` positional `dumped_path` | 必选，至少 2 个 | 各 dump JSON 文件路径。每个经 `readable_file` 独立校验。 |
| `run`/`inspect` positional `rule` | 必选 | CMATE 规则文件路径。经 `readable_file` 校验。inspect 与 run 共用同一规则路径校验逻辑。 |
| `--configs` 中路径分量 | 可选 | 格式 `name:path` 或 `name:path@type`。路径分量在 RunStrategy、Dump 或 `_parse_configs` 解析阶段经 `readable_file` 校验一次，结果以 `Path` 存入内部字典。 |
| legacy 别名参数 | 可选 | `--service_config_path`、`--weight_dir` 等 legacy 参数在 `legacy.py` 映射至 canonical 参数后，由 canonical 参数的 type 完成校验，不单独维护第二套逻辑。 |

#### 内部关键接口

下表描述模块间传递路径与遍历文件的核心内部接口；调用方须保证入参已在入口完成 normalize，被调方不再重复校验。

| 参数 | 可选/必选 | 说明 |
|------|-----------|------|
| `normalize_user_path(value: str) -> Path` | 必选字符串入参 | 展开 `~` 后 resolve 为绝对真实路径。供仅需规范化、不需预先存在的输出路径使用。异常时 propagate OS 错误。 |
| `as_arg_type(*checks: PathCheck) -> Callable[[str], Path]` | checks 至少 1 个 | 先 normalize，再依次执行 PathCheck 链。任一 check 失败抛出 `ArgumentTypeError`。仅需 normalize 而无存在性校验时，直接使用 `normalize_user_path` 作为 argparse `type`，不调用本函数。 |
| `readable_file` | 预置组合 | 等同 `as_arg_type(is_file, is_readable)`。用于 argparse `type=`。 |
| `existing_dir` | 预置组合 | 等同 `as_arg_type(is_dir)`。 |
| `has_suffix(suffix: str) -> PathCheck` | suffix 必选 | 返回可嵌入 `as_arg_type` 的后缀 check。suffix 须含 leading dot，如 `".json"`。 |
| `iter_regular_files(root: Path, *, suffix: str, max_bytes: int) -> Iterator[Path]` | root 必选，须已 normalize | 从 root 栈式递归遍历；root 可为软链接目录并正常展开，子项中软链接一律跳过。仅 yield 普通文件且 `stat.st_size <= max_bytes`。默认 max 为 10 GiB。 |

## 模块与周边关系

本组件图描述 msprechecker 与操作系统、Python 运行时及既有依赖之间的边界关系，以及 msguard 移除后的依赖变化。

![image.png](https://raw.gitcode.com/user-images/assets/8428112/8493271a-3dcc-443c-bbdb-d64f7947d5d4/image.png 'image.png')

msprechecker 运行于 Python 3.7 及以上环境，依赖 `pyyaml`、`psutil`、`ply`、`colorama`、`packaging` 等既有第三方包，不依赖 msguard。所有用户输入文件与目录均通过 path_io 在本进程内完成规范化后，由 pathlib 发起系统调用，实际权限判定与软链接解析由 Linux 内核 VFS 完成。工具不另起网络服务或守护进程，ping 与 hccn_tool 等外部命令通过 subprocess 调用。输出文件权限由用户 shell 环境的 umask 决定；README 建议非 root 用户安装前执行 `umask 0027`。

## DFX 能力设计

### 安全性

| 风险点 | 应对措施 |
|--------|----------|
| 移除 msguard 后路径遍历或越权读取 | 用户输入路径在入口一次性 resolve；手动拼接相对路径时对结果 resolve 并用 relative_to 约束在配置根目录内；下游不再重复打开未校验字符串路径。 |
| 目录递归跟随软链接导致死循环或越界 | `iter_regular_files` 对子项跳过软链接；输入根路径为软链接目录时仍展开其子树，但不进入任何软链接子目录或文件。 |
| 超大文件读取导致内存或 IO 耗尽 | 权重文件单文件上限 10 GiB，超限文件跳过；哈希计算采用分块读取，chunk_size 上限 256 MiB。 |
| 输出文件权限过宽 | README 建议非 root 用户安装前执行 `umask 0027`，由用户及管理员自行管控输出暴露面。 |
| subprocess 命令注入 | 保持 shlex.split 解析命令行，shell 参数为 False，与改造前一致。 |

### 可靠性

| 异常场景 | 容错机制 |
|----------|----------|
| 入口路径不存在或不可读 | argparse 层 ArgumentTypeError，进程立即退出，exit code 非零。 |
| collector open 遭遇 PermissionError | error_handler 记录错误，reporter 输出警告或错误项，同场景其余 collector 继续执行。 |
| 权重目录为空或无匹配后缀 | WeightCollector 记录采集错误，返回空 dict，checker 按空数据处理。 |
| ping 或 hccn_tool 不可执行 | 模块级缓存不可用标志，采集阶段输出明确错误信息，不抛未捕获异常。 |
| 输出路径父目录不可创建 | mkdir 或 open 捕获 OSError，向 stderr 输出错误并返回非零 exit code。 |
| NFS 或 Docker 挂载延迟导致 stat 失败 | EAFP 模式：首次 open 失败即上报，不做重试；用户重试命令。 |

### 可用性 / 性能指标

| 指标 | 目标值 | 设计考量 |
|------|--------|----------|
| Docker/NFS/共享目录场景 precheck 完成率 | 100%，不因属主或权限位预检中断 | 移除 msguard 前置扫描。 |
| 权重目录采集启动延迟 | 相较改造前降低，目标为去除全量权限树遍历的可感知等待 | 栈式 iter 仅 stat 与普通 open，无 chmod。 |
| 入口路径校验耗时 | 单次 readable_file 小于 50 ms 本地 ext4 | 仅 normalize 加一次 access，无递归。 |

### 可服务性

改造后日志与错误输出沿用现有 global_logger 与 error_handler 机制。入口校验失败时 argparse 将 ArgumentTypeError 消息直接打印至 stderr，用户可见具体路径与失败原因。collector 读取失败时 reporter 输出含文件名与 OSError 原因的检查项。输出文件写入成功后，precheck 仍提示 `source msprechecker_env.sh` 用法。问题定位时，用户可通过 `--verbose` 获取更详细的 cmate 规则执行信息，与改造前一致。无需新增运维接口。

### 其他指标

不涉及。本次改造不新增监控埋点或度量上报，性能评估通过本地计时与场景测试完成。

### 安全设计及安全 checklist

| Checklist 内容 | 检查结果 |
|----------------|----------|
| 1. 是否新增输入 | N |
| 2. 是否新增输出 | N |
| 3. 是否存在文件操作 | Y |
| 3.1 是否读取外部文件 | Y |
| 3.2 是否生成文件输出 | Y |
| 3.3 是否生成临时文件 | N |
| 3.4 是否解压缩文件 | N |
| 4. 是否涉及网络通信 | Y |
| 4.1 是否对外提供网络服务 | N |
| 4.2 是否访问外部网络 | Y |
| 5. 是否涉及注入风险 | Y |
| 5.1 是否涉及执行命令 | Y |
| 6. 是否引入第三方库 | N |
| 7. 是否新增二进制交付件 | N |
| 8. 是否存在加密、认证 | N |
| 9. 是否存在敏感信息 | N |
| 10. 是否使用安全函数库 | N |

读取外部文件时，YAML 仍使用 `yaml.safe_load`，JSON 使用标准库 `json.load`，不对不可信二进制做反序列化；读取前仅校验存在性与大小，内容格式校验由业务 checker 负责。生成输出文件时遵循当前进程 umask，不跟随用户提供的软链接创建输出。网络访问仍通过 subprocess 调用 ping，命令行经 shlex 拆分，不引入 shell=True。本次改造移除 msguard 而非新增第三方库，故第 6 项填 N；未引入公司安全函数库，第 10 项填 N。

### 可测试性

测试用例按正常、异常、边缘三类组织，覆盖 path_io 单元测试、子命令集成测试及资料与安全扫描场景测试。每类均包含 UT、IT、ST 不同粒度的用例。

#### 正常场景

正常场景验证改造后核心路径在典型部署环境下按预期完成校验、读取、遍历与输出，不产生 msguard 相关阻断。

| 用例名 | 前置操作 | 操作方式 | 预期结果 |
|--------|----------|----------|----------|
| UT_readable_file_symlink_input | 创建真实文件及指向它的软链接 | 调用 `readable_file(str(symlink))` | 返回 resolve 后真实文件 Path，不报错 |
| UT_has_suffix_compose_ok | 在 tmp_path 下创建 `a.txt` | 调用 `as_arg_type(is_file, has_suffix(".txt"))(str(path))` | 返回 resolve 后的 Path |
| IT_precheck_docker_uid | Docker 容器内挂载异 UID 权重目录与可读配置文件 | 执行 `msprechecker precheck --mies-config-path <cfg> --weight-dir <dir>` | 命令完整执行至结束，无 msguard 权限或属主拦截 |
| IT_precheck_nfs_shared | NFS 挂载共享模型目录，文件属主为非当前用户但 OS 允许读 | 同上 | 权重哈希采集成功，reporter 正常输出 |
| IT_precheck_root_run | 以 root 身份运行 precheck，配置与权重路径属主为普通用户 | 同上 | 不因属主不一致中断，采集与校验流程完整 |
| IT_compare_two_dumps | 准备两份格式合法的 dump JSON | 执行 `msprechecker compare old.json new.json` | 正常输出 diff 报告，无 ImportError |
| IT_run_cmate_rule | 准备合法 `.cmate` 规则与对应 config.json | 执行 `msprechecker run rule.cmate --configs cfg:config.json` | 规则执行完成，退出码符合规则结果 |
| IT_regression_baseline | 安装 msprechecker | 运行仓库现有 precheck、dump、compare、run 回归用例全集 | 全部通过，无功能退化 |
| ST_readme_constraint | 安装 msprechecker | 检查 README 安全与权限章节 | 含非 root 用户安装前 `umask 0027` 建议，以及工具不校验权限位、属主一致性及软链接安全性的说明 |

#### 异常场景

异常场景验证非法输入、权限不足、依赖不可用等条件下，工具以明确错误退出或上报，不发生未捕获异常或 msguard 残留依赖错误。

| 用例名 | 前置操作 | 操作方式 | 预期结果 |
|--------|----------|----------|----------|
| UT_readable_file_missing | tmp_path 下不存在目标文件 | argparse 以 `type=readable_file` 解析该路径 | 抛出 ArgumentTypeError，进程 exit code 非零 |
| UT_existing_dir_not_dir | tmp_path 下创建普通文件 `not_a_dir` | argparse 以 `type=existing_dir` 解析该路径 | 抛出 ArgumentTypeError，提示非目录 |
| UT_has_suffix_compose_reject | tmp_path 下创建 `a.json` | 调用 `as_arg_type(is_file, has_suffix(".txt"))(str(path))` | 抛出 ArgumentTypeError |
| IT_precheck_config_unreadable | 配置文件存在但 chmod 000 | 执行 precheck 并传入该 `--mies-config-path` | 入口 readable_file 阶段即失败，exit code 非零 |
| IT_collector_open_permission_denied | 入口校验通过后，运行中文件被移除读权限 | 模拟 ConfigCollector 对不可读 Path 执行 open | error_handler 记录 PermissionError，reporter 输出错误项，进程不崩溃 |
| IT_weight_dir_empty | 空权重目录 | 执行 precheck 并传入 `--weight-dir` | WeightCollector 记录无匹配文件错误，后续 checker 按空结果处理 |
| IT_ping_cmd_unavailable | 环境中 `/usr/bin/ping` 不存在或不可执行 | 执行含 PingCollector 的 precheck 网络场景 | 采集阶段输出明确错误，不重复判定，不抛未捕获异常 |
| IT_output_path_not_writable | 输出目录父路径不可写 | 执行 dump 并指定不可写 `--output-path` | 捕获 OSError，stderr 输出错误，exit code 非零 |
| IT_compare_missing_dump | 仅准备一个 dump 文件 | 执行 `msprechecker compare only.json` | argparse 或 CompareStrategy 报错，提示至少需要两个文件 |
| IT_run_missing_rule | 规则文件路径不存在 | 执行 `msprechecker run /no/such/rule.cmate` | 入口 readable_file 失败，exit code 非零 |
| ST_no_msguard_import | 改造代码合入分支 | 全仓库 grep 与 import 测试 | 无 msguard 残留 import，pytest 无 ImportError |
| ST_llm_code_scan | 改造代码合入分支 | 代码大模型安全扫描 | 无新增命令注入、路径遍历、不安全反序列化高风险项 |

#### 边缘场景

边缘场景验证软链接、大小边界、路径规范化特例等容易遗漏的约束，确保设计决策在边界条件下仍成立。

| 用例名 | 前置操作 | 操作方式 | 预期结果 |
|--------|----------|----------|----------|
| UT_iter_regular_files_skip_symlink | 权重目录含 1 个真实 `.safetensors` 与指向外部的 symlink 子目录，子目录内另有权重文件 | 调用 `list(iter_regular_files(root, suffix=".safetensors"))` | 结果仅含真实文件，symlink 子目录内文件不被纳入 |
| UT_iter_regular_files_size_at_limit | 目录含大小恰好 10 GiB 的 `.safetensors` | 调用 `iter_regular_files`，max_bytes 为 10 GiB | 该文件被 yield |
| UT_iter_regular_files_size_over_limit | 目录含 10 GiB 与 10 GiB 加 1 字节两个 `.safetensors` | 同上 | 仅 10 GiB 文件被 yield，超大文件被跳过 |
| UT_weight_root_is_symlink | 真实权重目录 `real_dir` 含 `.safetensors`，`link_dir` 为指向它的软链接 | CLI 传入 `--weight-dir link_dir` | 入口 existing_dir 通过，采集可发现 `real_dir` 内文件 |
| UT_normalize_dotdot_path | 在 tmp_path 下创建 `sub/cfg.json` | 调用 `normalize_user_path("sub/../sub/cfg.json")` 相对于 tmp_path | 返回消除 `..` 后的绝对 Path，指向 `sub/cfg.json` |
| UT_manual_join_relative_to | 配置根目录 `/data/config` 已 resolve，相对分量 `rules/extra.yaml` | 执行 `(base / relative).resolve()` 并 `relative_to(base)` | 解析成功，路径仍在 base 下 |
| UT_manual_join_escape_detect | 配置根目录 `/data/config`，相对分量 `../outside.yaml` | 同上，捕获 relative_to 失败 | 判定为路径逃逸，拒绝继续处理并返回错误 |
| IT_precheck_group_writable_config | 配置文件权限为 664，属主为其他用户 | 执行 precheck | 不因组可写被预检拒绝，OS 允许则正常读取 |
| IT_precheck_symlink_cycle_in_subdir | 权重目录子级存在互相指向的软链接环 | 执行 precheck 权重采集 | 遍历跳过软链接，进程不挂死，在超时时间内返回 |
| IT_long_output_path_near_limit | 输出路径长度接近 OS PATH_MAX | 执行 dump 并指定该 `--output-path` | 若 OS 允许则成功写入；若超限则 EAFP 返回明确 OSError |

## 特性规格与限制

### 平台限制

本次改造仅涉及 msprechecker Python 源码与测试，面向 Linux 环境。路径规范化依赖 `pathlib.Path.resolve` 与 `os.access`，在 WSL2、原生 Linux 及 Docker 容器内均可运行。`/proc` 等伪文件系统路径仍由 collector 只读访问，不受用户路径规范化流程约束。Python 版本要求保持 `>=3.7`，`iter_regular_files` 的栈式实现不依赖 Python 3.12 的 `follow_symlinks` 参数。

### 软件依赖

改造后运行时依赖为 `pyyaml`、`psutil`、`ply`、`colorama`、`packaging` 及 Python 3.7 以上标准库；`msguard` 从 `pyproject.toml` 移除，不再作为安装或运行前置条件。测试依赖仍为 `pytest` 与 `pytest-mock`，无新增第三方包。

### 功能约束

用户输入路径须在 CLI 或 Coordinator 入口完成唯一一次校验与 normalize，下游模块不得再对同一路径重复 `is_file` 或 `is_dir` 判断。读取侧不校验文件属主、权限位；输入路径为软链接时由 resolve 解析目标。目录递归遍历时仅跳过子级软链接，允许权重根目录本身为软链接。权重文件单文件大小上限为 10 GiB，超出文件被跳过且不参与哈希。工具输出文件与目录权限遵循当前进程 umask；README 建议非 root 用户安装前执行 `umask 0027`。路径长度不在读取侧做人为截断，超长路径由操作系统在创建或 open 时返回错误。手动拼接相对路径时须对结果 resolve，并在需要约束目录边界时使用 `relative_to` 检测逃逸。

### 已知约束

改造范围限于 msprechecker 包内 msguard 替换与 path_io 收口，不包含 cmate 内部 `set_env.sh` 写入逻辑的重命名或删除。内置命令 `/usr/bin/ping` 与 `hccn_tool` 的可执行性在模块加载时判定一次，运行中路径变化不会被动态感知。NFS 或高延迟挂载下，首次 stat 或 open 失败不做自动重试，需用户重新执行命令。

## 兼容性声明

不涉及。

## 拓展性

path_io 模块的 `check` 与 `as_arg_type` 组合机制可在未来新增 argparse 路径约束时复用，例如新增 `has_prefix` 或 `under_directory` 谓词，无需修改下游 collector。`iter_regular_files` 的后缀与 max_bytes 参数可在 WeightCollector 层扩展以支持其他权重格式，不必重新引入 msguard 遍历。若 MindStudio 工具链后续统一路径 IO 库，path_io 可整体提取为公共模块，msprechecker 仅保留 import 切换。输出权限策略由用户 shell 环境 umask 决定，README 提供 `umask 0027` 安装建议。不预留插件钩子或策略注册表；扩展需求出现时优先在 path_io 增加谓词或在入口 parser 增加 type 组合。
