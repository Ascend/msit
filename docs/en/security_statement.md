# msit Security Statement

## System Security Hardening

You are advised to enable the **address space layout randomization** (ASLR) (level 2) in the system. Run the following command to enable it:

    echo 2 > /proc/sys/kernel/randomize_va_space

## User Account Recommendations

1. All tools in this repository are designed to run with minimal permissions. For security reasons, do not use `root` or other privileged accounts. Always install and execute tools as a regular user.

2. If a tool depends on CANN, install the CANN package under the same non-privileged user. After running the `source` command, do not modify the environment variables in `set_env.sh`.

3. Before using any tools, set umask to `0027` or stricter to ensure generated files meet minimum security requirements.

## File Permission Control

 1. When providing input files to the tools, the file owner should match the user running the tool process, and permissions should prevent modification by group or others. By default, tool files written to disk are created restrictive permissions that prevent other users from modifying them. You can adjust these permissions after file creation.

 2. Proper permission control is essential during installation and use. For details, see the following table.

## File permission control 

| Type                              | Maximum Permission in Linux|
| ---------------------------------- | ------------------- |
| Home directory                        | 750 (rwxr-x---)   |
| Program files (including scripts and library files)    | 550 (r-xr-x---)   |
| Program file directory                      | 550 (r-xr-x---)   |
| Configuration files                          | 640 (rw-r-----)   |
| Configuration file directory                      | 750 (rwxr-x---)   |
| Log files (recorded or archived)    | 440 (r--r-----)   |
| Log files (being recorded)                | 640 (rw-r-----)   |
| Log file directory                      | 750 (rwxr-x---)   |
| Debug files                         | 640 (rw-r-----)   |
| Debug file directory                     | 750 (rwxr-x---)   |
| Temporary file directory                      | 750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  | 770 (rwxrwx---)   |
| Service data files                      | 640 (rw-r-----)   |
| Service data file directory                  | 750 (rwxr-x---)   |
| Key component, private key, certificate, and ciphertext file directory| 700 (rwx------)    |
| Key components, private keys, certificates, and ciphertext files    | 600 (rw-------)   |
| APIs and script files for encryption and decryption            | 500 (r-x------)   |

## Vulnerability Security Statement 

For details, see [MindStudio Vulnerability Handling Mechanism Description] (./vulnerability_handling_procedure.md).

## Data Security

1. When using the tool, data is loaded from and saved to disk. Some interfaces (e.g., `torch.load`) directly or indirectly use the unsafe `pickle` module, which can introduce security vulnerabilities. For details, see [torch.load](https://pytorch.org/docs/main/generated/torch.load.html#torch.load).

2. The ONNX model loading and parsing features depend on the third-party ONNX library. Versions prior to 1.15.0 are vulnerable to out-of-bounds read attacks. Always ensure that any ONNX model you load comes from a trusted source.

3. The tool reads model weights from disk—typically files you've downloaded from the internet and specified via command-line arguments or configuration files. Using untrusted model weights can introduce unknown security risks. Before passing any model weights to the tool, verify their integrity using methods like SHA256 to ensure they come from a trusted source.

## Build Security

​`msit` and `msmodelslim` can be installed from source. During compilation, third-party dependencies are downloaded, the shell build scripts are executed, and temporary program files and compilation directories are generated. To minimize security risks, you can set appropriate permissions on files in the source directory. During compilation, review and modify build scripts as needed before execution, to ensure the final build artifacts are securely stored.

## Runtime Security

1. When loading a dataset that exceeds available memory, or when monitoring runs too long and generated data fills the available disk space, the tool may exit unexpectedly.

2. If the tool encounters an error, it will exit and print error messages. This is expected behavior. To locate the error cause, you are advised to view the log file or collect the result file generated during parsing.

3. To prevent remote code injection attacks, set the `--trust-remote-code` parameter to `False` if it is used.

## Public Network Addresses

For details about the public network information in the configuration files and scripts within the msit repository, see [Public Network Address](./public_ip_address.md).

## Public APIs

The msit project is developed in C++ and Python. All officially supported APIs are documented and exposed only through Python APIs. The dynamic libraries are for internal use only and should not be called directly.

The tool is distributed as source code (Python and other scripting languages). Always use the documented public APIs. Avoid calling undocumented internal functions.

ATB pre-check, OM model saving, and AIE model conversion are compiled during installation. The APIs exposed by the resulting dynamic libraries are for internal use only. Do not modify or call these APIs.

## Safe Function Usage

While unsafe functions are not forcibly disabled, you are advised to use their safe variants that explicitly take a buffer length parameter, for example, `memset_s` and `memcpy_s`.

## Communication security hardening

This tool does not involve remote communication. However, you are advised to run it in a secure network environment, for example, behind a firewall or within a local area network (LAN). Be mindful of potential communication security risks when using third-party software.

## Communication Matrix

| Code Repository| Function| Source Device| Source IP Address| Source Port| Destination Device| Destination IP Address|Destination Port (Listening)| Protocol| Port Description| Port Configuration| Listening Port Configurable (Yes/No)| Authentication Mode| Encryption Mode| Plane| Version| Exceptions| Remarks|
| ------- | ------- | -------- | ------- | ------- | ------ | ------ | ----- | ---- | ------ | -------- | ------ | -------- | -------- | -------- | -------- | -------- | ----- |
|msserviceprofiler | vLLM server communication| vLLM server started by Serviceparam Optimizer| IP address matching the port parameter for the inference service| Fixed port configured based on the customer's requirements on the live network, which corresponds to the --port field during service startup. The default value is 8000.| vLLM client| IP address for vLLM client communication| **8000** (default)| HTTP | Starts the inference service in the customer environment through the CLI. If the customer does not specify the port, the default port 8000 is used. Otherwise, start the vLLM service using the port specified by the customer.| N/A  | Yes| N/A| N/A| Data plane| All| None| None|
|msprechecker | Network connectivity and hardware check (ping/hccn_tool)| Node where the pre-check tool is located| IP address of the node where the pre-check tool is located| ICMP (ping): no port; hccn_tool: uses system-reserved ports by default (e.g., RDMA ports)| Target host or NIC| Target host/NIC IP address| No ICMP port| ICMP | 1. Ping: tests the network connectivity (ICMP, no port).<br> 2. hccn_tool: checks the Ascend NPU NIC status based on the default RDMA/IB port (such as 3225 and 18515).| N/A| No (ICMP)| None| None| Control plane| All| Some commands need to be executed by the `root` user.| 1. The ping operation may be blocked by the firewall. 2. hccn_tool requires the NPU driver.|

`msserviceprofiler optimizer` runs on the server through `mindie-service`. `mindie-service` uses the following ports:

1. Inference service EndPoint, which provides a RESTful API for the service-plane inference service. The client initiates inference requests through this EndPoint. This EndPoint configuration corresponds to the `port` field in the `config.json` file. The value ranges from 1024 to 65535. The default value is `1025`. In the PD separation scenario, the default port number is `31015`.

2. Inference service EndPoint, which provides a RESTful API for the management-plane service status. The client initiates requests through this EndPoint for querying the inference service status. This EndPoint configuration corresponds to the `managementPort` field (when the management plane and the service plane use different ports) or `port` (when they share the same port) in the `config.json` file . The value ranges from 1024 to 65535. The default value is `1026`.

3. Inference service EndPoint, which provides a RESTful API for the management-plane service status. The client initiates requests through this EndPoint for querying the inference service status. This EndPoint configuration corresponds to the `metricsPort` field in the `config.json` configuration file. The value ranges from 1024 to 65535. The default value is `1027`.

For details, see [MindIE Documentation](https://www.hiascend.com/en/document).

`msserviceprofiler optimizer` uses tools such as `aisbench` and `vllm_benchmark` to interact with the mindie-service and vllm serve servers through ports. You need to configure the ports to ensure that the ports match those on the servers.

msprechecker uses `ping` and `hccn_tool` to check the network connectivity of multiple hosts and verify the HCCL communication status.<br>

`ping` checks network connectivity by sending ICMP packets directly at the IP layer. `hccn_tool` depends on the RoCE (RDMA) protocol of the Ascend NPU. By default, the hardware communication port 3225 (or 18515 in some scenarios) is used.<br>

`msprechecker tool` calls `hccn_tool` subcommands `vinc`, `tls`, `link`, `ping`, and `hccs_ping` through different default ports. `hccn_tool` ports are related to the hardware ports. For details about a hardware-specific port, see its communication matrix at [Ascend Hardware](https://support.huawei.com/enterprise/en/category/ascend-computing-pid-1557196528909). .

`msprechecker` uses the protocols and ports above for its checks. No additional port configuration is needed, but the network must support these protocols.
