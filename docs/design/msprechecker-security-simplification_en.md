# Feature Design

## Function Description

The MindStudio pre-check tool msprechecker extensively relies on the msguard security library for file read/write, directory traversal, and command-line parameter validation. This library implements owner consistency, permission bits, symbolic link rejection, and fixed-length upper limit pre-check logic according to SSHD and database core process specifications. In typical AI development scenarios such as Docker containers, NFS shared mounts, multi-user collaboration clusters, and cross-UID mapping, these checks do not match the actual threat model. As a result, pre-check, file writing, and rule execution processes are intercepted early even when the files themselves are readable and writable. This also introduces additional overhead from recursive permission correction and full-path rule traversal.

This modification aligns the file and path security policies of msprechecker with the unified principles of the MindStudio toolchain. It preserves the baseline checks for existence and size on the read side, removes redundant input restrictions, and eliminates the dependency on msguard. Output files and directories created by the tool follow the current process umask. The specific function points are as follows.

1. Remove file owner validation on the read side. All paths previously opened through msguard `open_s` no longer validate whether the inode owner matches the current process user. Whether the file can be read is entirely determined by the operating system permission mechanism.
2. Remove file permission bit validation on the read side. The tool no longer rejects reading user configuration files, weight files, and system information files because the group or other users have write access or the permission bits do not match a fixed template. The tool also no longer recursively modifies permissions of third-party files or directories before reading.
3. Remove symbolic link interception for input paths. If the user explicitly passes a configuration file path or weight directory path that is a symbolic link, the tool no longer rejects it before opening. The kernel resolves the symbolic link, and the tool only performs existence and size checks on the resolved target. During directory recursive traversal, the tool must not follow symbolic links. It must skip linked directories or files to avoid circular references. This constraint is independent of the input path release policy.
4. Relax path and environment variable length limits. Command-line parameters and configuration file paths no longer use the built-in fixed upper limit of msguard. The read side does not perform additional length validation. If the tool creates files or environment variables on behalf of the user and the length exceeds the operating system `PATH_MAX` or `ARG_MAX`, the tool catches the system error using the EAFP pattern and returns a clear failure message.
5. Replace the msguard file API with pathlib. Approximately 15 `open_s` calls are changed to `pathlib.Path` methods such as `open`, `read_text`, and `write_text`. The `walk_s` in the weight collection module is changed to `path_io.iter_regular_files` for stack-based recursive traversal. Sub-items skip symbolic links, while the input root path itself is allowed to be a symbolic link. Each regular file is validated to have a size no larger than 10 GiB. The entire modification process prohibits new `os.path` calls. Paths are converted to `Path` at the CLI boundary and passed inward.
6. Replace the msguard parameter validation API. `validate_args(Rule.input_file_read)` is changed to a combination of `path_io.readable_file` or `as_arg_type`. `Rule.input_file_exec.is_satisfied_by` is changed to a module-level one-time `os.access` check. User input paths are normalized and validated at the argparse or Coordinator entry point. Downstream modules do not repeat the validation.
7. Output-side permissions are determined by the user environment. Output files written by the tool, such as the JSON files and `msprechecker_env.sh` generated during the precheck process, as well as the output directories involved in the creation process, follow the current process umask. This policy is consistent with the read side: permissions are determined by the user and the operating system environment. The tool does not modify permissions of existing user files and does not perform permission bit pre-checks on read-side input files.
8. Remove the msguard dependency. The `msguard` package is removed from the `pyproject.toml` dependency list. All import statements and related test mock paths are cleaned up.
9. User documentation constraint description. The README recommends that non-root users execute `umask 0027` before installation. The documentation states that read-side and output-side permissions are managed by the user and administrator. The tool does not validate permission bits, owner consistency, or symbolic link security.

For users, in scenarios such as mounting heterogeneous UID weight directories in Docker containers, reading shared model files through NFS, or accessing configuration files as root or non-owner, the precheck, dump, compare, run, and inspect subcommands of msprechecker no longer fail due to owner or permission bit pre-checks. The startup latency of weight hash collection and rule file loading is expected to decrease because the full permission and symbolic link scan of msguard is no longer executed. The permissions of files and scripts generated by the tool follow the current process umask. Users can control the output exposure by setting `umask 0027` before installation. For the system, one third-party security library dependency is removed. The file access path is unified to pathlib and the EAFP pattern, which is consistent with the Python coding standard and the MindStudio toolchain security policy. In the acceptance tests for Docker, NFS, shared cluster multi-user, and root execution scenarios, the target case interruption rate is zero. The code large model scan and regression tests should not report new injection, path traversal, or other security issues.

## Implementation Approach

The modification is carried out in six steps in dependency order. First, the path normalization and validation are centralized at the entry layer. Then, the resolved `Path` objects are passed downstream. After that, the read and traversal implementations are replaced. Next, the output-side write behavior is unified. Finally, the dependencies are cleaned up and the tests and documentation are synchronized. The core constraints are: user input paths are validated only once at the CLI or Coordinator boundary, and `expanduser` and `resolve` are completed. Downstream modules trust the input type and do not repeat `is_file` or permission bit checks. The argparse `type` uses composable validators to express multiple conditions, avoiding a separate function for each combination.

### Step 1: Add Path Normalization and Composable Validation Module

**Text description:** Define the single entry logic for path handling in `msprechecker/utils/path_io.py`. `normalize_user_path` converts a user input string into a normalized absolute real path: first `Path.expanduser` expands the tilde and user home directory, then `Path.resolve` resolves symbolic links and eliminates `..` components, returning a directly usable `Path`. Validators use lightweight composition: `check` wraps a named predicate or a `functools.partial`-bound predicate into a `PathCheck`. `as_arg_type` chains normalize and several `PathCheck` instances for direct use as the argparse `type`. Two commonly used combinations, `readable_file` and `existing_dir`, are predefined. For other scenarios, write `as_arg_type(is_file, has_suffix(".txt"))` at the call site. There is no need to add module-level functions for each combination. The implementation follows Python coding standards. Predicates prefer named functions and `partial`, not lambda. The module does not introduce additional abstraction layers such as Protocol or registries.

**Implementation sample:**

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
    """root must be a Path already normalized at the entry; root is allowed to be a symbolic link directory, sub-items do not follow symbolic links."""
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

Also export `normalize_user_path`, `as_arg_type`, `readable_file`, `existing_dir`, `has_suffix`, and `iter_regular_files` in `msprechecker/utils/__init__.py`.

**Additional notes:** Parameters that only require normalization without existence validation can directly use `type=normalize_user_path` without going through `as_arg_type`. Internal fixed paths such as `/proc/cpuinfo` are defined as `Path` constants in the module and do not go through `normalize_user_path`. When manually concatenating relative paths, the result must be `resolve`d again. If necessary, use `relative_to` to confirm that the path has not escaped the root directory.

### Step 2: CLI Entry Point Unified Validation and Passing Resolved Path Downstream

**Text description:** All user-visible path-type argparse parameters are connected to the `as_arg_type` or predefined combinations from Step 1 in this step. This completes normalization and existence validation. The argparse definitions that still use `validate_args` in `commands/precheck.py`, `commands/dump.py`, `commands/compare.py`, `commands/_cmate.py`, `commands/legacy.py`, and `cmate/cmate.py` are covered. This includes configuration file paths, weight directories, rule files, output paths, and rank table paths. The path fields extracted from `args` by `Coordinator.execute`, Dump strategy, and RunStrategy are all `Path` objects that have been `resolve`d. When passing to collector, checker, and cmate engines, `Path()` or `is_file` is no longer called. For composite string parameters such as `--configs`, the path components are parsed in the Coordinator or Dump layer. Each path component is validated once through `readable_file`, and the parsed result is stored in the internal data structure as a `Path`.

**Implementation sample:**

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
# msprechecker/cmate/cmate.py — Independent cmate entry argparse synchronization
from ..utils.path_io import readable_file, normalize_user_path

run_parser.add_argument("rule", type=readable_file, help="...")
inspect_parser.add_argument("rule", type=readable_file, help="...")
run_parser.add_argument("--output-path", type=normalize_user_path, help="...")
# The output directory is created on the write side, not at the argparse layer (see Step 5 _actual_run)
```

```python
# msprechecker/cmate/cmate.py — Create output directory during file writing
output_dir = Path(output_path)
output_dir.mkdir(parents=True, exist_ok=True)
saved_json = output_dir / msprechecker_output_name
with saved_json.open("w", encoding="utf-8") as f:
    json.dump(msprechecker_output, f, ...)
```

```python
# msprechecker/commands/coordinator.py — configs composite parameter path component consolidation
from pathlib import Path
from typing import Tuple

from ..utils.path_io import readable_file

def _parse_config_entry(entry: str) -> Tuple[str, Path]:
    name, _, raw_path = entry.partition(":")
    path = readable_file(raw_path.split("@", 1)[0])
    return name, path
```

**Additional notes:** If the same `args` field is read by multiple handlers, the validation still occurs only once at the argparse layer. Handlers pass `Path` references. Wrapping or validating again in intermediate layers is prohibited.

The composable validators produced in Step 1 are all consumed in this step. Subsequent modules only receive the normalized `Path`.

### Step 3: Downstream Modules Remove Redundant Validation and Directly Read/Write Using EAFP

**Text description:** In 12 source files, delete `open_s` and all duplicate existence checks for user input paths, including the rule read and file write paths in `cmate/cmate.py`. The constructor parameter types of modules such as collector, checker, and cmate are annotated as `Path`. The implementation directly uses `path.open` or `read_text`. When opening fails, the existing `error_handler` catches `OSError`. `is_file` is not called again before opening. Class-level system path constants remain hardcoded `Path` objects and do not go through the user path normalization process. Modules that indirectly receive paths, such as `presets/manager.py` and `utils/ascend.py`, assume that the upstream has completed the consolidation and remove internal secondary conversions.

**Implementation sample:**

```python
# msprechecker/collectors/config.py — Receives validated Path, no longer checks is_file
class ConfigCollector(BaseCollector):
    def __init__(self, error_handler=None, *, config_path: Path):
        super().__init__(error_handler)
        self.config_path = config_path  # Already a resolve'd Path

    def _collect_data(self):
        with self.config_path.open(encoding="utf-8") as f:
            ...
```

```python
# msprechecker/presets/manager.py — Rule path passed in by RuleManager entry, no longer normalized
def _load_rule_file(self, rule_path: Path) -> dict:
    with rule_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
```

**Additional notes:** When reading fails, the existing error handling of each module is used. The exception types are standard `OSError` and `PermissionError`. No additional inference is made before catching.

Step 2 ensures that paths entering this module are already normalized. This step only performs IO replacement and does not introduce any path validation logic.

### Step 4: Refactor WeightCollector Traversal and One-Time Internal Command Path Validation

**Text description:** `collectors/weight.py` removes the msguard `walk_s` and rule expressions. `weight_dir` is validated by the CLI layer using `existing_dir` and passed in. The collector no longer calls `Path()` or `is_dir`. `_get_tensor_files` only calls `iter_regular_files` for suffix filtering and the 10 GiB size constraint. During traversal, symbolic links are not followed. The `/usr/bin/ping` and `hccn_tool` in `collectors/network.py` and `collectors/hccl.py` are built-in fixed paths of the tool. They are resolved once through `Path.resolve` at the module-level constant, and the executability check result is cached as a module-level boolean flag. `_collect_data` no longer performs repeated checks.

**Implementation sample:**

```python
# msprechecker/collectors/weight.py
class WeightCollector(BaseCollector):
    def __init__(self, error_handler=None, *, weight_dir: Path, chunk_size=None):
        super().__init__(error_handler)
        self.weight_dir = weight_dir  # CLI-validated existing_dir

    def _get_tensor_files(self, tensor_suffix: str):
        return list(iter_regular_files(
            self.weight_dir, suffix=tensor_suffix, max_bytes=DEFAULT_MAX_FILE_BYTES,
        ))
```

```python
# msprechecker/collectors/network.py — Built-in command path module-level one-time validation
PING_CMD = Path("/usr/bin/ping").resolve()
_PING_AVAILABLE = PING_CMD.is_file() and os.access(PING_CMD, os.X_OK)

class PingCollector(BaseCollector):
    def __init__(self, ...):
        self._ping_cmd = None if not _PING_AVAILABLE else f"{PING_CMD} -c 3 -q -W 2 {{}}"
```

**Additional notes:** Built-in command paths do not go through `expanduser`. Only `resolve` is used to eliminate potential symbolic links. This is independent of the user input path processing path to avoid confusion.

The EAFP reading completed in Step 3 provides the prerequisite of no redundant validation for this step. After this step is completed, msguard is completely removed from the business path.

### Step 5: Unify Output-Side Write Behavior

**Text description:** Files actively created by the tool follow the current process umask. If the user-specified output path is passed in at the CLI layer, it must be normalized through `normalize_user_path` before writing. The JSON file is written by `cmate/cmate.py`, and the environment script is written by `reporters/strategy.py`. If the parent directory does not exist, use `mkdir(parents=True, exist_ok=True)` to create the directory chain. The output path is normalized at the Coordinator or argparse layer and passed downstream. The write module does not repeat `resolve`.

**Implementation sample:**

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

**Additional notes:** The read-side `open` is not affected by umask. The permissions of output files and directories are determined by the user shell environment umask. The README recommends that non-root users execute `umask 0027` before installation.

After Step 4 removes msguard, this step completes the separation of read and write policies.

### Step 6: Remove msguard Dependency and Synchronize Tests and Documentation

**Text description:** After confirming that there are no msguard imports in the entire repository, remove `"msguard"` from `pyproject.toml`. In tests, change the cases that patch `open_s` to use the `tmp_path` fixture with `normalize_user_path` to construct inputs, or directly pass resolved `Path` objects to the function under test. Add `tests/test_utils/test_path_io.py` to cover `as_arg_type`, `has_suffix`, and the resolution of `~` and symbolic link input paths by `normalize_user_path`. The README documents the recommendation that non-root users execute `umask 0027` before installation, that entry validation occurs once, that downstream trusts Path, and that the tool does not validate permission bits, owner consistency, or symbolic link security.

**Implementation sample:**

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
# pyproject.toml — Remove msguard from dependencies
dependencies = ["pyyaml", "psutil", "ply", "colorama", "packaging", ...]
```

**Additional notes:** After the modification is completed, execute the full pytest and Docker, NFS, and shared directory scenario verification to confirm that the function interruption rate is zero.

After Step 5 completes the output-side write implementation, this step completes the dependency cleanup and quality loop. The overall modification is delivered at this point.

### Logic Flow Diagram

This diagram describes the runtime main flow of msprechecker after the security policy simplification. It covers the complete chain from the user triggering through the command line to result output or file writing, as well as the branch behavior of path validation and file access.

![image.png](https://raw.gitcode.com/user-images/assets/8428112/0b8b5eda-3217-4c12-8038-bf4eea5ec1d6/image.png 'image.png')

On the normal path, after the user invokes `msprechecker precheck`, `dump`, `compare`, `run`, or `inspect` through the shell, the process entry retains the root run warning (if applicable). Then argparse performs `expanduser` and `resolve` on all path-type parameters. It calls `readable_file`, `existing_dir`, or `normalize_user_path` according to the parameter semantics to complete the one-time entry validation. After validation passes, `Coordinator` passes the normalized `Path` objects to the corresponding strategy. The precheck and dump paths sequentially drive collector collection, checker validation, and reporter reporting. Among them, configuration files and system information are read directly by the collector through `Path.open`. The weight directory is recursively scanned by `iter_regular_files` under suffix and size constraints, and symbolic link sub-items are skipped. The compare path directly opens multiple dump JSON files and the `Comparator` outputs differences. The run and inspect paths load rule files and then execute validation or formatted display respectively. If precheck detects environment variable issues, or if dump and run specify output locations, the tool writes `msprechecker_env.sh` or JSON files and directory chains according to the current process umask.

On the exception path, when entry path validation fails, argparse raises `ArgumentTypeError`. The process terminates with a non-zero exit code and does not enter any collector or cmate logic. When the collector encounters `OSError` or `PermissionError` during `Path.open`, the existing `error_handler` records the error and includes it in the reporter output. The process is not interrupted early due to msguard-style pre-checks. When the weight directory passes entry validation but contains no files matching the suffix and size constraints, `WeightCollector` records a collection error and returns an empty result. The subsequent checker processes it according to the existing logic. When a built-in command path is not executable, `PingCollector` and `HCCNCollector` cache the unavailable status during initialization. The collection phase outputs a clear error instead of repeated checks. When the output path parent directory cannot be created or the disk is not writable, the write operation catches `OSError` and returns a failure message to the user.

### Sequence Diagram

This diagram describes the interactions between components in chronological order when the user triggers pre-checks through different subcommands after the modification. It focuses on the one-time path entry validation, downstream direct consumption of resolved `Path` objects, and the call relationships between read and write operations and the operating system.

![image.png](https://raw.gitcode.com/user-images/assets/8428112/414bd417-bd7d-4a07-8cfa-daea37405bcb/image.png 'image.png')

On the normal path, when the user initiates precheck, the cli entry sequentially calls path_io to execute `readable_file` on the configuration file and `existing_dir` on the weight directory. Both validations complete normalize and immediately return the resolved `Path`. The Coordinator passes them to PrecheckStrategy without secondary validation. After PrecheckStrategy creates the collector list, WeightCollector requests directory contents from the file system through `iter_regular_files` and skips symbolic link sub-items. Other collectors directly read through `Path.open`. The collection results sequentially flow through the checker and reporter and are displayed in the terminal. If the reporter detects environment variable issues, it writes `msprechecker_env.sh` according to the current process umask and prompts the user to source it.

When the user initiates dump, the output path is only normalized through `normalize_user_path`. The collection chain also directly reads through `open`. The dump strategy creates the directory chain and writes JSON according to the current process umask. When the user initiates compare, multiple dump files are each validated once at the argparse layer. CompareStrategy directly reads through `open` and drives the reporter to output differences. When the user initiates run, the rule file is validated at the argparse layer. The path components in the composite configs parameter are each validated once during the RunStrategy parsing phase. Then the cmate engine reads through `open` and executes rules. The optional output_path is written according to the current process umask.

On the exception path, when the entry `readable_file` probe detects that the file system returns not-found or not-readable, path_io raises `ArgumentTypeError` to argparse. The cli outputs a parameter error to the user and exits with a non-zero code. The Coordinator and components below are not called. When the collector or cmate encounters permission or IO errors during `open`, the error is reported through the existing `error_handler` to the reporter. msguard-style pre-interception is not triggered.

### Code Structure Design

This class diagram focuses on the newly added path IO module and the classes whose dependencies have directly changed. It shows the division of path validation consolidation and read/write responsibilities after the modification.

![image.png](https://raw.gitcode.com/user-images/assets/8428112/875e172e-b07b-4353-972b-dc21042eccbd/image.png 'image.png')

The `path_io` module serves as the single-responsibility unit for path handling. It exposes three types of capabilities: normalize, composable validation, and directory traversal. It does not introduce a class hierarchy. `cli` and each parser mount argparse validation through callables such as `readable_file` and `existing_dir`. `Coordinator` and various Strategy classes only pass `Path` objects. `WeightCollector` is the only module that still performs file-level filtering, but it only executes suffix and size constraints and does not repeat entry existence validation. Read-only collectors such as `ConfigCollector` receive `Path` and directly open. `EnvErrorDisplay` follows the current process umask when writing environment scripts. Other existing class structures such as collector, checker, Reporter, Comparator, and cmate parser remain unchanged. Only the import source changes from msguard to the standard library pathlib and path_io.

### Interface Design

#### External Interface

The following table describes the parameter interfaces that users can reach through the command line and are related to this modification. All path validation is completed once during the argparse parsing phase.

| Parameter | Required/Optional | Description |
|-----------|-------------------|-------------|
| `--mies-config-path` | Optional | MindIE configuration file path. Validated through `readable_file`: expanduser, resolve to an absolute real path, must be a readable regular file. If it does not exist or is not readable, `ArgumentTypeError` is raised and the process exits. |
| `--user-config-path` | Optional | PD separation scenario user_config.json path. Validation rules are the same as `readable_file`. |
| `--mindie-env-path` | Optional | PD separation scenario mindie_env.json path. Validation rules are the same as `readable_file`. |
| `--custom-config-path` | Optional | Custom validation rule YAML path. Validation rules are the same as `readable_file`. |
| `--config-parent-dir` | Optional | PD separation configuration root directory. Validated through `existing_dir`. Relative paths concatenated under it must be resolved and checked with relative_to to prevent escape. |
| `--weight-dir` | Optional | Model weight directory. Validated through `existing_dir`: must be an existing directory. The directory itself is allowed to be a symbolic link. Downstream traversal does not follow sub-level symbolic links. |
| `--rank-table` and similar rank table paths | Optional | Multi-node topology JSON path. Validated through `readable_file`. |
| `--output-path` | Optional | dump or run output path. Normalized through `normalize_user_path`. The file is not required to pre-exist. Writing follows the current process umask. |
| `compare` positional `dumped_path` | Required, at least 2 | Each dump JSON file path. Each is independently validated through `readable_file`. |
| `run`/`inspect` positional `rule` | Required | CMATE rule file path. Validated through `readable_file`. inspect and run share the same rule path validation logic. |
| Path components in `--configs` | Optional | Format `name:path` or `name:path@type`. The path component is validated once through `readable_file` during the RunStrategy, Dump, or `_parse_configs` parsing phase. The result is stored in the internal dictionary as a `Path`. |
| Legacy alias parameters | Optional | `--service_config_path`, `--weight_dir`, and other legacy parameters are mapped to canonical parameters in `legacy.py`. Validation is completed by the type of the canonical parameter. A second set of logic is not maintained separately. |

#### Internal Key Interface

The following table describes the core internal interfaces for passing paths and traversing files between modules. The caller must ensure that the input parameter has been normalized at the entry. The callee does not repeat validation.

| Parameter | Required/Optional | Description |
|-----------|-------------------|-------------|
| `normalize_user_path(value: str) -> Path` | Required string input | Expands `~` and resolves to an absolute real path. For output paths that only require normalization without pre-existence. Propagates OS errors on failure. |
| `as_arg_type(*checks: PathCheck) -> Callable[[str], Path]` | At least 1 check | First normalizes, then executes the PathCheck chain in sequence. If any check fails, `ArgumentTypeError` is raised. When only normalization is needed without existence validation, use `normalize_user_path` directly as the argparse `type`. Do not call this function. |
| `readable_file` | Predefined combination | Equivalent to `as_arg_type(is_file, is_readable)`. Used for argparse `type=`. |
| `existing_dir` | Predefined combination | Equivalent to `as_arg_type(is_dir)`. |
| `has_suffix(suffix: str) -> PathCheck` | suffix required | Returns a suffix check that can be embedded in `as_arg_type`. suffix must contain a leading dot, such as `".json"`. |
| `iter_regular_files(root: Path, *, suffix: str, max_bytes: int) -> Iterator[Path]` | root required, must be normalized | Stack-based recursive traversal from root. root can be a symbolic link directory and expands normally. Symbolic links among sub-items are always skipped. Only yields regular files with `stat.st_size <= max_bytes`. Default max is 10 GiB. |

## Module and Peripheral Relationships

This component diagram describes the boundary relationships between msprechecker and the operating system, Python runtime, and existing dependencies, as well as the dependency changes after msguard removal.

![image.png](https://raw.gitcode.com/user-images/assets/8428112/8493271a-3dcc-443c-bbdb-d64f7947d5d4/image.png 'image.png')

msprechecker runs on Python 3.7 and later environments. It depends on existing third-party packages such as `pyyaml`, `psutil`, `ply`, `colorama`, and `packaging`. It does not depend on msguard. All user input files and directories are normalized within the process through path_io. System calls are then initiated through pathlib. Actual permission validation and symbolic link resolution are performed by the Linux kernel VFS. The tool does not start a network service or daemon. External commands such as ping and hccn_tool are called through subprocess. Output file permissions are determined by the umask of the user shell environment. The README recommends that non-root users execute `umask 0027` before installation.

## DFX Capability Design

### Security

| Risk | Mitigation |
|------|------------|
| Path traversal or unauthorized read after msguard removal | User input paths are resolved once at the entry. When manually concatenating relative paths, the result is resolved and constrained within the configuration root directory using relative_to. Downstream modules do not reopen unvalidated string paths. |
| Directory recursive following symbolic links causing infinite loops or boundary violations | `iter_regular_files` skips symbolic links among sub-items. When the input root path is a symbolic link directory, its subtree is still expanded, but no symbolic link sub-directory or file is entered. |
| Excessive file read causing memory or IO exhaustion | Weight file single file upper limit is 10 GiB. Files exceeding the limit are skipped. Hash calculation uses chunked reading. The chunk_size upper limit is 256 MiB. |
| Output file permissions too broad | The README recommends that non-root users execute `umask 0027` before installation. The output exposure is managed by the user and administrator. |
| subprocess command injection | `shlex.split` is maintained for command-line parsing. The shell parameter is False. This is consistent with the pre-modification behavior. |

### Reliability

| Exception Scenario | Fault Tolerance Mechanism |
|--------------------|--------------------------|
| Entry path does not exist or is not readable | argparse layer `ArgumentTypeError`. The process exits immediately with a non-zero exit code. |
| collector open encounters `PermissionError` | `error_handler` records the error. The reporter outputs a warning or error item. Other collectors in the same scenario continue to execute. |
| Weight directory is empty or has no matching suffix | `WeightCollector` records a collection error and returns an empty dict. The checker processes it as empty data. |
| ping or hccn_tool is not executable | The module-level cache stores the unavailable flag. The collection phase outputs a clear error message. No uncaught exception is thrown. |
| Output path parent directory cannot be created | `mkdir` or `open` catches `OSError`. An error is output to stderr and a non-zero exit code is returned. |
| NFS or Docker mount delay causes stat failure | EAFP pattern: the first `open` failure is reported immediately. No retry is performed. The user retries the command. |

### Usability / Performance Metrics

| Metric | Target Value | Design Consideration |
|--------|-------------|----------------------|
| Docker/NFS/shared directory scenario precheck completion rate | 100%, not interrupted by owner or permission bit pre-checks | Remove msguard pre-scan. |
| Weight directory collection startup latency | Reduced compared to pre-modification. Target is to remove the perceptible wait of full permission tree traversal | Stack-based iter only does stat and regular open, no chmod. |
| Entry path validation time | Single `readable_file` less than 50 ms on local ext4 | Only normalize plus one access, no recursion. |

### Serviceability

After modification, logs and error output continue to use the existing `global_logger` and `error_handler` mechanisms. When entry validation fails, argparse prints the `ArgumentTypeError` message directly to stderr. The user can see the specific path and failure reason. When collector reading fails, the reporter outputs a check item containing the file name and the `OSError` reason. After the output file is written successfully, precheck still prompts the user to `source msprechecker_env.sh`. During troubleshooting, the user can use `--verbose` to obtain more detailed cmate rule execution information, consistent with the pre-modification behavior. No new O&M interface is needed.

### Other Metrics

Not applicable. This modification does not add monitoring instrumentation or metric reporting. Performance evaluation is completed through local timing and scenario testing.

### Security Design and Security Checklist

| Checklist Item | Result |
|----------------|--------|
| 1. Whether new input is added | N |
| 2. Whether new output is added | N |
| 3. Whether file operations are involved | Y |
| 3.1 Whether external files are read | Y |
| 3.2 Whether file output is generated | Y |
| 3.3 Whether temporary files are generated | N |
| 3.4 Whether file decompression is involved | N |
| 4. Whether network communication is involved | Y |
| 4.1 Whether network services are provided externally | N |
| 4.2 Whether external networks are accessed | Y |
| 5. Whether injection risks are involved | Y |
| 5.1 Whether command execution is involved | Y |
| 6. Whether third-party libraries are introduced | N |
| 7. Whether new binary deliverables are added | N |
| 8. Whether encryption or authentication is involved | N |
| 9. Whether sensitive information is involved | N |
| 10. Whether a security function library is used | N |

When reading external files, YAML still uses `yaml.safe_load`, and JSON uses the standard library `json.load`. No untrusted binary deserialization is performed. Before reading, only existence and size are validated. Content format validation is the responsibility of the business checker. When generating output files, the current process umask is followed. The tool does not create output following symbolic links provided by the user. Network access still calls ping through subprocess. The command line is split through shlex. `shell=True` is not introduced. This modification removes msguard rather than adding a third-party library, so item 6 is N. No company security function library is introduced, so item 10 is N.

### Testability

Test cases are organized into three categories: normal, exception, and edge. They cover path_io unit tests, subcommand integration tests, and documentation and security scan scenario tests. Each category includes cases at different granularities: UT, IT, and ST.

#### Normal Scenarios

Normal scenarios verify that the core path after modification completes validation, reading, traversal, and output as expected in typical deployment environments, without producing msguard-related blocking.

| Case Name | Prerequisite | Operation | Expected Result |
|-----------|--------------|-----------|-----------------|
| UT_readable_file_symlink_input | Create a real file and a symbolic link pointing to it | Call `readable_file(str(symlink))` | Returns the resolved real file Path, no error |
| UT_has_suffix_compose_ok | Create `a.txt` under tmp_path | Call `as_arg_type(is_file, has_suffix(".txt"))(str(path))` | Returns the resolved Path |
| IT_precheck_docker_uid | Mount heterogeneous UID weight directory and readable configuration file in Docker container | Execute `msprechecker precheck --mies-config-path <cfg> --weight-dir <dir>` | Command completes execution, no msguard permission or owner interception |
| IT_precheck_nfs_shared | NFS mount shared model directory, file owner is not the current user but OS allows reading | Same as above | Weight hash collection succeeds, reporter outputs normally |
| IT_precheck_root_run | Run precheck as root, configuration and weight path owner is a regular user | Same as above | Not interrupted due to owner mismatch, collection and validation flow complete |
| IT_compare_two_dumps | Prepare two valid dump JSON files | Execute `msprechecker compare old.json new.json` | Outputs diff report normally, no ImportError |
| IT_run_cmate_rule | Prepare a valid `.cmate` rule and corresponding config.json | Execute `msprechecker run rule.cmate --configs cfg:config.json` | Rule execution completes, exit code matches rule result |
| IT_regression_baseline | Install msprechecker | Run the full set of existing precheck, dump, compare, run regression test cases | All pass, no functional regression |
| ST_readme_constraint | Install msprechecker | Check README security and permissions section | Contains the recommendation for non-root users to execute `umask 0027` before installation, and the statement that the tool does not validate permission bits, owner consistency, or symbolic link security |

#### Exception Scenarios

Exception scenarios verify that under illegal input, insufficient permissions, unavailable dependencies, and other conditions, the tool exits with a clear error or reports the issue. No uncaught exceptions or msguard residual dependency errors occur.

| Case Name | Prerequisite | Operation | Expected Result |
|-----------|--------------|-----------|-----------------|
| UT_readable_file_missing | Target file does not exist under tmp_path | argparse parses the path with `type=readable_file` | Raises ArgumentTypeError, process exit code non-zero |
| UT_existing_dir_not_dir | Create a regular file `not_a_dir` under tmp_path | argparse parses the path with `type=existing_dir` | Raises ArgumentTypeError, prompts not a directory |
| UT_has_suffix_compose_reject | Create `a.json` under tmp_path | Call `as_arg_type(is_file, has_suffix(".txt"))(str(path))` | Raises ArgumentTypeError |
| IT_precheck_config_unreadable | Configuration file exists but chmod 000 | Execute precheck with `--mies-config-path` pointing to it | Fails at entry readable_file phase, exit code non-zero |
| IT_collector_open_permission_denied | After entry validation passes, file read permission is removed during runtime | Simulate ConfigCollector opening an unreadable Path | error_handler records PermissionError, reporter outputs error item, process does not crash |
| IT_weight_dir_empty | Empty weight directory | Execute precheck with `--weight-dir` | WeightCollector records no matching file error, subsequent checker processes as empty result |
| IT_ping_cmd_unavailable | `/usr/bin/ping` does not exist or is not executable in the environment | Execute precheck network scenario with PingCollector | Collection phase outputs clear error, no repeated check, no uncaught exception |
| IT_output_path_not_writable | Output directory parent path is not writable | Execute dump with unwritable `--output-path` | Catches OSError, stderr outputs error, exit code non-zero |
| IT_compare_missing_dump | Only one dump file prepared | Execute `msprechecker compare only.json` | argparse or CompareStrategy reports error, prompts at least two files needed |
| IT_run_missing_rule | Rule file path does not exist | Execute `msprechecker run /no/such/rule.cmate` | Entry readable_file fails, exit code non-zero |
| ST_no_msguard_import | Modified code merged into branch | Full repository grep and import test | No msguard residual import, pytest has no ImportError |
| ST_llm_code_scan | Modified code merged into branch | Code large model security scan | No new command injection, path traversal, or unsafe deserialization high-risk items |

#### Edge Scenarios

Edge scenarios verify constraints that are easy to overlook, such as symbolic links, size boundaries, and path normalization special cases. They ensure that design decisions still hold under boundary conditions.

| Case Name | Prerequisite | Operation | Expected Result |
|-----------|--------------|-----------|-----------------|
| UT_iter_regular_files_skip_symlink | Weight directory contains 1 real `.safetensors` and a symlink sub-directory pointing outside, with weight files inside the sub-directory | Call `list(iter_regular_files(root, suffix=".safetensors"))` | Result contains only real files, files inside symlink sub-directory are not included |
| UT_iter_regular_files_size_at_limit | Directory contains a `.safetensors` file of exactly 10 GiB | Call `iter_regular_files` with max_bytes of 10 GiB | The file is yielded |
| UT_iter_regular_files_size_over_limit | Directory contains two `.safetensors` files: 10 GiB and 10 GiB plus 1 byte | Same as above | Only the 10 GiB file is yielded, oversized file is skipped |
| UT_weight_root_is_symlink | Real weight directory `real_dir` contains `.safetensors`, `link_dir` is a symbolic link pointing to it | CLI passes `--weight-dir link_dir` | Entry existing_dir passes, collection finds files in `real_dir` |
| UT_normalize_dotdot_path | Create `sub/cfg.json` under tmp_path | Call `normalize_user_path("sub/../sub/cfg.json")` relative to tmp_path | Returns absolute Path with `..` eliminated, pointing to `sub/cfg.json` |
| UT_manual_join_relative_to | Configuration root directory `/data/config` is resolved, relative component `rules/extra.yaml` | Execute `(base / relative).resolve()` and `relative_to(base)` | Resolution succeeds, path is still under base |
| UT_manual_join_escape_detect | Configuration root directory `/data/config`, relative component `../outside.yaml` | Same as above, catch relative_to failure | Determined as path escape, processing is refused and error is returned |
| IT_precheck_group_writable_config | Configuration file permission is 664, owner is another user | Execute precheck | Not rejected by pre-check due to group-writable, OS allows reading normally |
| IT_precheck_symlink_cycle_in_subdir | Weight directory sub-level has mutually pointing symbolic link cycle | Execute precheck weight collection | Traversal skips symbolic links, process does not hang, returns within timeout |
| IT_long_output_path_near_limit | Output path length approaches OS PATH_MAX | Execute dump with this `--output-path` | If OS allows, writes successfully; if exceeds limit, EAFP returns clear OSError |

## Feature Specifications and Constraints

### Platform Constraints

This modification only involves msprechecker Python source code and tests, targeting Linux environments. Path normalization relies on `pathlib.Path.resolve` and `os.access`, which can run on WSL2, native Linux, and Docker containers. Pseudo-file system paths such as `/proc` are still read-only accessed by collectors and are not subject to the user path normalization process. The Python version requirement remains `>=3.7`. The stack-based implementation of `iter_regular_files` does not depend on the `follow_symlinks` parameter of Python 3.12.

### Software Dependencies

After modification, the runtime dependencies are `pyyaml`, `psutil`, `ply`, `colorama`, `packaging`, and the Python 3.7 standard library. `msguard` is removed from `pyproject.toml` and is no longer an installation or runtime prerequisite. Test dependencies remain `pytest` and `pytest-mock`, with no new third-party packages.

### Function Constraints

User input paths must be validated and normalized once at the CLI or Coordinator entry. Downstream modules must not repeat `is_file` or `is_dir` checks on the same path. The read side does not validate file owner or permission bits. When the input path is a symbolic link, `resolve` parses the target. During directory recursive traversal, only sub-level symbolic links are skipped. The weight root directory itself is allowed to be a symbolic link. The single file size upper limit for weight files is 10 GiB. Files exceeding the limit are skipped and do not participate in hashing. Tool output file and directory permissions follow the current process umask. The README recommends that non-root users execute `umask 0027` before installation. Path length is not artificially truncated on the read side. Overly long paths return errors from the operating system during creation or open. When manually concatenating relative paths, the result must be resolved. When directory boundary constraints are needed, use `relative_to` to detect escape.

### Known Constraints

The scope of modification is limited to msguard replacement and path_io consolidation within the msprechecker package. It does not include renaming or deleting the `set_env.sh` write logic inside cmate. The executability of built-in commands `/usr/bin/ping` and `hccn_tool` is determined once at module load time. Path changes during runtime are not dynamically detected. Under NFS or high-latency mounts, the first stat or open failure is not automatically retried. The user needs to re-execute the command.

## Compatibility Statement

Not applicable.

## Extensibility

The `check` and `as_arg_type` composable mechanism of the path_io module can be reused when adding new argparse path constraints in the future. For example, adding `has_prefix` or `under_directory` predicates does not require modifying downstream collectors. The suffix and max_bytes parameters of `iter_regular_files` can be extended at the WeightCollector layer to support other weight formats, without re-introducing msguard traversal. If the MindStudio toolchain later unifies the path IO library, path_io can be extracted as a public module, and msprechecker only needs to retain the import switch. The output permission policy is determined by the user shell environment umask. The README provides the `umask 0027` installation recommendation. No plugin hooks or strategy registries are reserved. When extension needs arise, prioritize adding predicates in path_io or adding type combinations at the entry parser.
