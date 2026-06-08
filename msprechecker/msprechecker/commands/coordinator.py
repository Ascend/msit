# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=duplicate-code

import argparse
import json
import os
from pathlib import Path

import yaml

from ..checkers import (
    AscendChecker,
    EnvChecker,
    HCCLChecker,
    LinkChecker,
    MIESConfigChecker,
    MindIEEnvChecker,
    ModelConfigChecker,
    PDChecker,
    PingChecker,
    StressChecker,
    SysChecker,
    TlsChecker,
    UserConfigChecker,
    VnicChecker,
)
from ..cmate import inspect, run
from ..cmate.cmate import _parse_configs, _parse_contexts
from ..collectors import (
    AscendCollector,
    BaseCollector,
    CollectResult,
    CPUStressCollector,
    EnvCollector,
    HCCLCollector,
    LinkCollector,
    MIESConfigCollector,
    MindIEEnvCollector,
    ModelConfigCollector,
    NPUStressCollector,
    PingCollector,
    SysCollector,
    TlsCollector,
    UserConfigCollector,
    VnicCollector,
    WeightCollector,
)
from ..comparators import Comparator
from ..presets import RuleManager
from ..reporters import Reporter
from ..utils import (
    CheckErrorHandler,
    ConfigErrorHandler,
    Framework,
    global_logger,
    parse_rank_table,
    singleton,
    update_model_type,
)
from .banner import BannerPresenter
from .base import CommandStrategy, CommandType
from .dump import Dump
from .legacy import show_legacy_warnings
from .logo import print_logo


class CollectorFactory:
    @staticmethod
    def create(args: argparse.Namespace):
        default_collectors = [
            SysCollector(),
            AscendCollector(),
        ]  # all scenes applies
        if getattr(args, "framework", "mindie") == "vllm" or args.command == CommandType.CMD_DUMP:
            default_collectors.append(EnvCollector(filter_env=getattr(args, "filter_env", False)))

        special_collectors = CollectorFactory.dispatch_collectors_by_scene(args)
        extra_collectors = CollectorFactory.dispatch_extra_collectors(args)

        return list(set(default_collectors + special_collectors + extra_collectors))

    @staticmethod
    def dispatch_collectors_by_scene(args: argparse.Namespace) -> list[BaseCollector]:
        collectors = []

        # 大 EP
        if getattr(args, "user_config_path", None) or getattr(args, "mindie_env_path", None):
            if getattr(args, "user_config_path", None):
                collectors.append(UserConfigCollector(config_path=args.user_config_path))
            if getattr(args, "mindie_env_path", None):
                collectors.append(MindIEEnvCollector(config_path=args.mindie_env_path))

        # PD Disaggregation (single container)

        # PD Mix
        elif getattr(args, "mies_config_path", None):
            collectors.append(EnvCollector(filter_env=getattr(args, "filter_env", False)))
            collectors.append(MIESConfigCollector(config_path=args.mies_config_path))

        return collectors

    @staticmethod
    def dispatch_extra_collectors(args: argparse.Namespace) -> list[BaseCollector]:
        collectors = []

        if getattr(args, "rank_table_path", None):
            if not getattr(args, "framework", None):
                global_logger.warning(
                    "Passing '--rank-table-path' without providing '--scene', "
                    "msprechecker cannot determine the exact framework type of the rank table. "
                    "Will use 'mindie' as the default framework."
                )
                args.framework = Framework.MINDIE.value

            framework_type = Framework(args.framework)
            rank_table = parse_rank_table(args.rank_table_path, framework_type)

            collectors.extend(
                (
                    PingCollector(rank_table=rank_table),
                    TlsCollector(),
                    HCCLCollector(rank_table=rank_table),
                    LinkCollector(),
                    VnicCollector(),
                )
            )

        if getattr(args, "weight_dir", None):
            model_config_path = args.weight_dir / "config.json"
            collectors.append(ModelConfigCollector(config_path=model_config_path))

            if getattr(args, "command", None) == CommandType.CMD_DUMP:
                chunk_size = getattr(args, "chunk_size", 32)
                chunk_size *= 1024**2
                collectors.append(WeightCollector(weight_dir=args.weight_dir, chunk_size=chunk_size))

        if getattr(args, "hardware", False):
            collectors.extend((CPUStressCollector(), NPUStressCollector()))

        return collectors


@singleton
class CheckerFactory:
    def __init__(self):
        self._registry = {}
        self._init()

    @staticmethod
    def default_param_extractor(args, collect_result):
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": CheckErrorHandler(severity=args.severity_level),
        }

    @staticmethod
    def config_param_extractor(args, collect_result):
        data, file_lines, key_mapping, context_hierarchy = collect_result.data
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": ConfigErrorHandler(args.severity_level, file_lines, key_mapping, context_hierarchy),
        }

    @staticmethod
    def stress_param_extractor(args, collect_result):
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": CheckErrorHandler(severity=args.severity_level),
            "threshold": getattr(args, "threshold", None),
        }

    def register(self, collector_class, checker_cls, param_extractor=None) -> None:
        param_extractor = param_extractor or self.default_param_extractor
        self._registry[collector_class] = (checker_cls, param_extractor)

    def create(self, collector_cls, args, collect_result):
        if collector_cls not in self._registry:
            raise KeyError(f"No checker registered for collector: {collector_cls.__name__}")

        checker_cls, param_extractor = self._registry[collector_cls]
        params = param_extractor(args, collect_result)
        return checker_cls(**params)

    def _init(self):
        self.register(EnvCollector, EnvChecker)
        self.register(SysCollector, SysChecker)
        self.register(AscendCollector, AscendChecker)
        self.register(HCCLCollector, HCCLChecker)
        self.register(CPUStressCollector, StressChecker, self.stress_param_extractor)
        self.register(NPUStressCollector, StressChecker, self.stress_param_extractor)
        self.register(UserConfigCollector, UserConfigChecker, self.config_param_extractor)
        self.register(MindIEEnvCollector, MindIEEnvChecker, self.config_param_extractor)
        self.register(ModelConfigCollector, ModelConfigChecker, self.config_param_extractor)
        self.register(MIESConfigCollector, MIESConfigChecker, self.config_param_extractor)
        self.register(TlsCollector, TlsChecker)
        self.register(VnicCollector, VnicChecker)
        self.register(LinkCollector, LinkChecker)
        self.register(PingCollector, PingChecker)


class PrecheckStrategy(CommandStrategy):
    @staticmethod
    def execute_pd_disagg(args):
        rule_manager = RuleManager(scene=args.scene, custom_rule_path=args.custom_config_path)
        reporter = Reporter()

        paths_to_find = rule_manager.get_rules().keys()

        collect_data = {}
        for path in paths_to_find:
            if "ref" in path:
                continue
            if os.path.isabs(path):
                global_logger.warning("unsafe, key should not be abspath: {path!r}")
                continue
            full_path = Path(args.config_parent_dir) / path
            load_fn = json.load if full_path.suffix == ".json" else lambda f: list(yaml.safe_load_all(f))
            try:
                with full_path.open(encoding="utf-8") as f:
                    data = load_fn(f)
            except Exception:
                global_logger.error("missing file: %r", full_path)
                return 1

            collect_data[path] = data

        error_handler = CheckErrorHandler(severity=args.severity_level, type_="PD Disaggregation")
        collect_result = CollectResult(collect_data, error_handler)
        checker = PDChecker(rule_manager=rule_manager, error_handler=error_handler)
        check_result = checker.check(collect_result)
        reporter.report(check_result)
        return 0

    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        if args.scene and "pd_disaggregation" in args.scene:
            if not args.config_parent_dir:
                global_logger.error(
                    "Passing '--scene' without providing '--config-parent-dir' will not take any effect!"
                )
            return PrecheckStrategy.execute_pd_disagg(args)

        if args.scene and "," in args.scene:
            parts = args.scene.split(",", 1)
            if len(parts) != 2 or not all(parts):
                global_logger.error("Invalid scene format! Use 'framework,scene'")
                return 1
            args.framework = parts[0].strip()
            args.scene = parts[1].strip()
        else:
            args.framework = args.scene
            args.scene = "default"

        reporter = Reporter()
        collectors = CollectorFactory.create(args)
        checker_factory = CheckerFactory()

        for collector in collectors:
            collect_result = collector.collect()
            checker = checker_factory.create(collector.__class__, args, collect_result)
            check_result = checker.check(collect_result)
            reporter.report(check_result)

        return 0


class CompareStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        if len(args.dumped_path) < 2:
            global_logger.error("You need two or more files to compare!")
            return 1

        path_to_data = CompareStrategy._load_dumped_files(args.dumped_path)
        reporter = Reporter()

        comparator = Comparator()
        reporter.report(comparator.compare(path_to_data))
        return 0

    @staticmethod
    def _load_dumped_files(file_paths):
        """Load dumped JSON files for comparison"""
        path_to_data = {}

        for path in file_paths:
            with path.open(encoding="utf-8") as f:
                path_to_data[path] = json.load(f)

        return path_to_data


class RunStrategy(CommandStrategy):
    @staticmethod
    def execute(args):
        BannerPresenter().print_banner()

        ret, configs = _parse_configs(args.configs)
        if not ret:
            return 1

        ret, contexts = _parse_contexts(args.contexts)
        if not ret:
            return 1

        return run(
            args.rule,
            configs,
            contexts,
            args.failfast,
            args.verbose,
            args.collect_only,
            args.output_path,
            args.severity,
        )


class InspectStrategy(CommandStrategy):
    @staticmethod
    def execute(args):
        return inspect(args.rule, args.format)


class CommandStrategyFactory:
    def __init__(self) -> None:
        self._registry = {
            CommandType.CMD_PRECHECK: PrecheckStrategy,
            CommandType.CMD_DUMP: Dump,
            CommandType.CMD_COMPARE: CompareStrategy,
            CommandType.CMD_RUN: RunStrategy,
            CommandType.CMD_INSPECT: InspectStrategy,
        }

    def register(self, cmd_type, strategy_class) -> None:
        self._registry[cmd_type] = strategy_class

    def create_strategy(self, cmd: CommandType) -> CommandStrategy:
        if cmd not in self._registry:
            raise ValueError(f"No strategy registered for command: {cmd}")

        return self._registry[cmd]()


class Coordinator:
    def __init__(self) -> None:
        self._strategy_factory = CommandStrategyFactory()

    def execute(self, parser: argparse.ArgumentParser) -> int:
        """Execute the appropriate action based on command"""
        args = parser.parse_args()
        update_model_type(args)
        show_legacy_warnings(args)

        cmd = getattr(args, "command", None)
        if not cmd:
            parser.print_help()
            return 1

        print_logo()
        args.command = CommandType(cmd)
        strategy = self._strategy_factory.create_strategy(args.command)
        return strategy.execute(args)
