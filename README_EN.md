<h1 align="center">MindStudio Inference Tools</h1>

<div align="center">

<p><b><span style="font-size:24px;">Ascend AI Inference Development Toolchain</span></b></p>

 [![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](./docs/en/quick_start/msit_quick_start.md)
 [![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q%26A/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/26.2.0)
 [![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q%26A/ZRead/blue)](https://zread.ai/mindstudio-docs/26.2.0)
 [![Exact Search](https://badgen.net/badge/Exact%20Search/ReadTheDocs/blue)](https://mindstudio-docs-26.2.0.readthedocs.io)
 [![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/eng/developer/software/mindstudio)
 [![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://atomgit.com/Ascend/msit/issues)
</div>

English | [简体中文](./README.md)

## ✨ What's New

<span style="font-size:14px;">

🔹 **[Sep 9, 2026]**: The msIT repository has announced the sunset of its profile and benchmark modules. For details, see the [announcement](https://atomgit.com/Ascend/msit/discussions/4)    
🔹 **[Mar 30, 2026]**: The msIT repository has announced the sunset of its precision debugging, inference service tuning, and model quantization modules. For details, see the [announcement](https://atomgit.com/Ascend/msit/discussions/2)  
🔹 **[Jan 12, 2026]**: The license of this repository has changed. For details, see the [announcement](https://atomgit.com/Ascend/msit/discussions/1)    
🔹 **[Dec 31, 2025]**: The MindStudio Inference Development Toolchain was fully open-sourced

</span>

## ℹ️ Overview

MindStudio Inference Tools (msIT) is an inference development toolchain that focuses on key challenges in the inference development of LLMs and traditional models. By providing capabilities such as model compression, debugging, and tuning, it efficiently addresses issues such as low inference efficiency and high resource consumption, helping users achieve optimal inference performance.

<img src="./docs/en/figures/readme/fullview.svg" width="1200"/>

## ⚙️ Features

The inference development toolchain provides the following series of tools:

| Category | Tool                                                                          | Description                                               |
|:--:|:-----------------------------------------------------------------------------|:---------------------------------------------------|
| Precheck | [**msPrechecker**](https://atomgit.com/Ascend/msit/tree/26.2.0/msprechecker) | **[Precheck tool]** Supports environment prechecks, connectivity prechecks, and dumping and comparing inference process data to help users identify issues before deployment.  |
| Quantization | [**msModelSlim**](https://atomgit.com/Ascend/msmodelslim)                    | **[Model compression]** Provides quantization, compression, and other inference optimization techniques, supporting dense LLMs, MoE models, and multimodal models. |
| Precision | [**msProbe**](https://atomgit.com/Ascend/msprobe)                            | **[Precision debugging]** A full-scenario precision tool for Ascend, used for precision debugging and issue localization.                  |
| Performance | [**msProf**](https://atomgit.com/Ascend/msprof)                              | **[Model tuning]** A full-scenario performance tuning foundation that collects full-stack hardware and software performance data to improve device tuning efficiency.    |
| Performance | [**msprof-analyze**](https://atomgit.com/Ascend/msprof-analyze)              | **[Performance analysis]** Analyzes collected data to quickly identify performance bottlenecks.                   |
| Performance | [**msServiceProfiler**](https://atomgit.com/Ascend/msserviceprofiler)        | **[Service tuning]** Supports request scheduling and visualization of model execution to improve service performance analysis efficiency.           |
| Performance | [**msMemScope**](https://atomgit.com/Ascend/msmemscope)                      | **[Memory tuning]** A dedicated memory tuning tool that collects multidimensional memory data for the entire network and supports automatic diagnosis and optimization analysis.        |
| Performance | [**msInsight**](https://atomgit.com/Ascend/msinsight)                        | **[Visual tuning]** Provides visual performance analysis across system, operator, and service scenarios to assist with performance diagnosis.        |
| Performance | [**msModeling**](https://atomgit.com/Ascend/msmodeling)                      | **[Modeling and simulation]** A neural network inference performance simulation framework that helps developers predict performance, identify bottlenecks, and optimize configurations when hardware is unavailable or before deployment. |
| Monitoring | [**msMonitor**](https://atomgit.com/Ascend/msmonitor)                        | **[Online monitoring]** A one-stop monitoring tool that supports both data dumping and online data collection for cluster monitoring and issue localization.           |

## 🚀 Quick Start

MindStudio Inference Tools (msIT) is an open-source inference development toolchain for the Ascend AI platform. It provides developers with end-to-end tool support from model deployment to online operations and maintenance. The toolchain covers five areas: precheck, quantization, precision debugging, performance tuning, and online monitoring. It also includes multiple specialized tools that help developers efficiently develop and optimize inference for LLMs (including dense models and MoE architectures), multimodal models, and traditional models on Ascend hardware. To quickly get started with the tools, see [Inference Development Toolchain Quick Start](docs/en/quick_start/msit_quick_start.md).

## 📦 Installation Guide

Follow the links in the [Features](#️-features) table above to access the corresponding source repositories. For details, see the "Installation Guide" section in the repository README file.

## 📘 User Guide

Follow the links in the table above to access the corresponding source repositories, and see the *User Guide* in their README files. To select tools by scenario, see the [msIT Tool Selection Guide](./docs/en/user_guide/msit_user_guide.md).    

## 🌌 Intelligent Search

To make it easier to access information in the documentation, we provide several efficient search methods:  
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/mindstudio-docs/26.2.0): Ask questions in natural language to quickly understand the project architecture and relationships between modules.   
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/26.2.0): Provides a better Chinese Q&A experience and accurately locates feature usage and details.   
🔹 [Precise Search (ReadTheDocs)](https://mindstudio-docs-26.2.0.readthedocs.io): Search the full text by keyword to find information about interfaces, parameters, and error messages directly.  

## 🛠️ Contribution Guide

You are welcome to contribute to the project. For details, see the [Contribution Guide](https://atomgit.com/Ascend/msit/blob/26.2.0/docs/en/contributing/contributing_guide.md).

## ⚖️ Related Information
 
🔹 [License Statement](https://atomgit.com/Ascend/msit/blob/26.2.0/docs/en/legal/license_notice.md)     
🔹 [msIT Security Statement](https://atomgit.com/Ascend/msit/blob/26.2.0/docs/en/legal/security_statement.md)     
🔹 [Disclaimer](https://atomgit.com/Ascend/msit/blob/26.2.0/docs/en/legal/disclaimer.md)     

## 🤝 Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [Issues](https://atomgit.com/Ascend/msit/issues). We will reply as soon as possible. Thank you for your support.

|                                                                         Live Interaction (WeChat Group)                                                                          |                                                                               Official Updates (Official Account)                                                                                | In-Depth Support (Assistant/Forum)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://raw.atomgit.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical discussion group*</sub> | <img src="https://raw.atomgit.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR codes to join the group and follow the official account, providing the fastest way to connect with MindStudio users and developers:<br> **Ask questions quickly:** Discuss technical issues with community members in real time<br>**Stay up to date:** Get notifications about version releases and feature updates as soon as they are available<br> **Share experience:** Exchange best practices and hands-on experience with developers  <br> <br> **More support channels**: 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](./docs/en/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by Huawei Technologies Co., Ltd. and developers.
Thanks for every PR from the community, and contributions are welcome.
