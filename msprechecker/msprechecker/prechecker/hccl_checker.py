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

from glob import glob
from concurrent import futures
from msprechecker.prechecker.register import GroupPrechecker, PrecheckerBase
from msprechecker.prechecker.register import show_check_result, CheckResult
from msprechecker.prechecker.utils import logger
from msprechecker.prechecker.utils import parse_ranktable_file, run_shell_command, get_interface_by_ip

_DAVINCI_DEVICES = sorted(glob("/dev/davinci*"))
NPU_DEVICES = [int(ii.split("davinci")[-1]) for ii in _DAVINCI_DEVICES if str.isdigit(ii.split("davinci")[-1])]
INDENT = " " * 6
ACTION_WHEN_NO_DATA_COLLECTED = (
    f"解决方案：1. 通过 `-ranktable` 提供 ranktable\n{INDENT}2. 在容器外检查\n{INDENT}3. 检查 hccn_tool"
)
REASON_WHEN_NO_DATA_COLLECTED = (
    f"可能原因：1. ranktable 中没有找到远端 device ip 的信息\n{INDENT}2. 当前环境没有 hccn_tool\n{INDENT}3. 执行失败"
)


def run_hccl_command(command_formatter):
    from shutil import which

    if not which("hccn_tool"):
        return []  # Just return if hccn_tool command not exists. It's common if in docker.

    results = []
    with futures.ThreadPoolExecutor(max_workers=len(NPU_DEVICES)) as executor:
        map_args = [command_formatter.format(device_id=device_id) for device_id in NPU_DEVICES]
        for result in executor.map(run_shell_command, map_args):
            results.append([ii.strip() for ii in result.stdout.split("\n")] if result else [])
    return results


class HcclIfnameChecker(PrecheckerBase):
    __checker_name__ = "HcclIfname"

    def collect_env(self, **kwargs):
        logger.debug(f"Starting HcclIfnameChecker")
        results = run_hccl_command("hccn_tool -i {device_id} -lldp -g")
        ifnames = []
        for result in results:
            cur_ifname = ""
            for line in result:
                if "Ifname:" in line:
                    cur_ifname = line.split(":", 1)[-1].strip()
                    break
            ifnames.append(cur_ifname)
        logger.debug(f"ifnames = {ifnames}")
        return ifnames

    def do_precheck(self, envs, **kwargs):
        if not envs:
            show_check_result(
                "hccl",
                "lldp Ifname",
                CheckResult.UNFINISH,
                action=ACTION_WHEN_NO_DATA_COLLECTED,
                reason=REASON_WHEN_NO_DATA_COLLECTED
            )
            return
        if not all(len(ii) > 0 for ii in envs):
            show_check_result(
                "hccl",
                "lldp Ifname",
                CheckResult.ERROR,
                action=f"检查服务器上 NPU 对应的交换机连接，如果是光纤直连，忽略该条",
                reason=f"HCCL Ifname 存在空值 {envs}",
            )
        else:
            show_check_result("hccl", f"lldp Ifname: {envs}", CheckResult.OK)


class HcclLinkChecker(PrecheckerBase):
    __checker_name__ = "HcclLink"

    def collect_env(self, **kwargs):
        logger.debug(f"Starting HcclLinkChecker")
        results = run_hccl_command("hccn_tool -i {device_id} -link -g")
        link_status = []
        for result in results:
            cur_link_status = ""
            for line in result:
                if "link status:" in line:
                    cur_link_status = line.split("link status:")[-1].strip()
                    break
            link_status.append(cur_link_status)
        logger.debug(f"link_status = {link_status}")
        return link_status

    def do_precheck(self, envs, **kwargs):
        if not envs:
            show_check_result(
                "hccl",
                "lldp Ifname",
                CheckResult.UNFINISH,
                action=ACTION_WHEN_NO_DATA_COLLECTED,
                reason=REASON_WHEN_NO_DATA_COLLECTED
            )
            return
        if not all(ii == "UP" for ii in envs):
            show_check_result(
                "hccl",
                "link",
                CheckResult.ERROR,
                action=f"检查服务器上 NPU 连接情况",
                reason=f"HCCL link 存在 down 值 {envs}",
            )
        else:
            show_check_result("hccl", f"link: {envs}", CheckResult.OK)


class HcclTlsSwitchChecker(PrecheckerBase):
    __checker_name__ = "HcclTlsSwitch"

    def collect_env(self, **kwargs):
        logger.debug(f"Starting HcclTlsSwitchChecker")
        results = run_hccl_command("hccn_tool -i {device_id} -tls -g")
        tls_switch = []
        for result in results:
            cur_tls_switch = ""
            for line in result:
                if "tls switch[" in line:
                    cur_tls_switch = line.split("tls switch[")[-1].split("]")[0].strip()
                    break
            tls_switch.append(cur_tls_switch)
        logger.debug(f"tls_switch = {tls_switch}")
        return tls_switch

    def do_precheck(self, envs, **kwargs):
        if not envs:
            show_check_result(
                "hccl",
                "tls switch",
                CheckResult.UNFINISH,
                action=ACTION_WHEN_NO_DATA_COLLECTED,
                reason=REASON_WHEN_NO_DATA_COLLECTED
            )
            return
        if not all(ii == "0" for ii in envs):
            show_check_result(
                "hccl",
                "tls switch",
                CheckResult.ERROR,
                action=f"检查服务器上 HCCL tls 状态，推荐关闭：for i in {{0..7}}; do hccn_tool -i $i -tls -s enable 0; done",
                reason=f"HCCL tls 打开可能影响多机连接",
            )
        else:
            show_check_result("hccl", f"tls_switch: {envs}", CheckResult.OK)


class HcclPingChecker(PrecheckerBase):
    __checker_name__ = "HcclPing"

    def collect_env(self, ranktable_file=None, **kwargs):
        logger.debug(f"Starting HcclPingChecker.")
        logger.info(f"ranktable_file={ranktable_file}")
        ranktable = parse_ranktable_file(ranktable_file)
        if not ranktable:
            logger.warning("ranktable is missing, will skip HcclPingChecker between servers.")
            return None

        ranktable_ips = {}
        for server in ranktable.get("server_list", []):
            server_id = server.get("server_id", None)
            device_ips = [ii.get("device_ip", None) for ii in server.get("device", [])]
            ranktable_ips[server_id] = device_ips
        logger.debug(f"ranktable_ips={ranktable_ips}")

        _, self.local_ip = get_interface_by_ip(list(ranktable_ips.keys()))
        if self.local_ip is None:
            logger.error(f"Current server is not set in ranktable={ranktable_file}")
            return None

        multi_server_results = {}
        for server_ip, device_ips in ranktable_ips.items():
            if server_ip == self.local_ip:
                continue
            logger.info(f"HCCL Ping server_ip={server_ip}, devices={len(device_ips)} ...")
            for device_ip in device_ips:
                if device_ip is None:
                    continue
                logger.debug(f"HcclPingChecker server_ip={server_ip}, device_ip={device_ip}")
                results = run_hccl_command("hccn_tool -i {device_id} -ping -g address " + device_ip)
                bool_results = [not any("100.00% packet loss" in ii for ii in result) for result in results]
                if bool_results:
                    multi_server_results.setdefault(server_ip, {}).update({device_ip: bool_results})
        logger.debug(f"ping results = {multi_server_results}")
        return multi_server_results

    def do_precheck(self, envs, **kwargs):
        if not envs:
            show_check_result(
                "hccl",
                "ping",
                CheckResult.UNFINISH,
                action=ACTION_WHEN_NO_DATA_COLLECTED,
                reason=REASON_WHEN_NO_DATA_COLLECTED
            )
            return
        
        for server_ip, device_connect_result in envs.items():
            if server_ip == self.local_ip:
                continue
            is_connect_server_pass = True
            for device_id, (_, connect_result) in enumerate(device_connect_result.items()):
                if all(connect_result):
                    # 路由器连接，所有卡间都能 ping 通
                    continue
                if all([res if cur_id == device_id else not res for cur_id, res in enumerate(connect_result)]):
                    # 光纤连接，对应卡之间可以 ping 通
                    continue
                show_check_result(
                    "hccl",
                    "ping",
                    CheckResult.ERROR,
                    action=f"检查本机到服务器 {server_ip} {device_id} 卡的连接状态",
                    reason=f"本机到对端卡的 ping 结果存在失败 {connect_result}",
                )
                is_connect_server_pass = False
            if is_connect_server_pass:
                show_check_result("hccl", f"ping server {server_ip} all pass", CheckResult.OK)


class HCCLChecker(GroupPrechecker):
    __checker_name__ = "HCCL"

    def init_sub_checkers(self):
        return [
            HcclIfnameChecker(),
            HcclLinkChecker(),
            HcclTlsSwitchChecker(),
            HcclPingChecker(),
        ]


hccl_checker_instance = HCCLChecker()
