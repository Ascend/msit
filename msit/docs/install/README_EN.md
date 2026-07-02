# Installing the msit tool

> The following is the latest installation method. Versions earlier than 7.0.0 are installed through shell scripts. For details, see the[Installation Guide (Earlier Version)](./history.md)    "

## Environment and Dependency

The installation of the msit inference tool includes the installation of the msit package and the dependent component packages. You can add only the required component packages as required.

| Dependent Software Name | Mandatory or not | Version                                                                                                                                                                              | Remarks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN                    | Mandatory.       | You are advised to install CANN 8.0.RC1 or later.                                                                                                                                    | For details, see the[CANN-8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)    Install the Ascend development or running environment, that is, the toolkit software package. After the installation, configure environment variables as prompted. For details, see.[Configuring Environment Variables](#installation-check-command-msit-check).                                                                                                                                                                                                                                                                                          |
| GCC                     | Mandatory.       | 7.3.0 version                                                                                                                                                                        | For details, see the[GCC Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/softwareinst/instg/instg_0123.html)    Install the GCC compiler. (By default, the GCC 4.8 compiler is used on CentOS 7.6. Therefore, this tool may not be installed. You are advised to update the GCC compiler before installing it.)                                                                                                                                                                                                                                                                                                                                                                                |
| Python                  | Mandatory.       | Supports Python3.7.5+, Python3.8.x, Python3.9.x, and Python3.10.x.                                                                                                                   | To use the precision comparison function of the TensorFlow model, install Python 3.7.5. For other functions, install Python 3.7.0 or later. Pay attention to the dependency between Python and Torch. For example, Python 3.8 corresponds to Torch 2.1.0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| TensorFlow              | Not mandatory.   | TensorFlow1.15.0 (corresponding to Python versions 3.7.5 to 3.7.11) and TensorFlow2.6.5 are supported. (corresponding to Python versions 3.7.5-3.7.11, 3.8. 0-3.8.11, 3.9. 0-3.9. 2) | Reference[Installing TensorFlow1.15.0 on CentOS 7.6](https://bbs.huaweicloud.com/blogs/181055)    Install the TensorFlow1.15.0 environment. (If the precision comparison function of the TensorFlow model is not used, the installation is not required.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Caffe                   | Not mandatory.   | Match the Python version.                                                                                                                                                            | Reference[Caffe Installation](http://caffe.berkeleyvision.org/installation.html)    Install the Caffe environment. (If the precision comparison function of the Caffe model is not used, the installation is not required.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Clang                   | Not mandatory.   | Match the Python version.                                                                                                                                                            | Depends on LLVM Clang and needs to be installed.[Clang tool](https://releases.llvm.org/)    .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ATB                     | Not mandatory.   | This version matches the MindIE-LLM version.                                                                                                                                         | Depends on the acceleration library ATB. The MindIE image is recommended. (including the ATB, MindIE-LLM, and MindIE-Service components). For details about how to obtain the image, see.[MindIE image](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)    Install the required MindIE image. You can also compile and install the source code. For details, see.[MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM)    This section describes how to install the acceleration library ATB tool. You need to apply for joining the Ascend organization before viewing the code repository. If the msit llm dump, opcheck, and errcheck functions are not used, the installation is not required. |
| MindIE-LLM              | Optional.        | Supports 1.0.RC3 and 1.0.0, which correspond to CANN package versions 8.0.RC1 and 8.0.RC2, respectively.                                                                             | Depends on the MindIE-LLM component. The MindIE image is recommended. (including the ATB, MindIE-LLM, and MindIE-Service components). For details about how to obtain the image, see.[MindIE image](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)    Install the required MindIE image. You can also compile and install the source code. For details, see.[MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM)    This section describes how to install MindIE-LLM. You need to apply for joining the Ascend organization before viewing the code repository. If the msit llm dump, opcheck, and errcheck functions are not used, the installation is not required.                            |

## Installing the MSite

### Pre-Installation Description

 * Install the drivers, firmware, and CANN packages related to the Ascend AI inference in the development running environment. For details, see.[CANN-8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)    .
 * After the installation, you can set the CANN_PATH environment variable to specify the path of the CNAN version, for example, export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest.
 * If this parameter is not set, the tool attempts to obtain the CANN version from the ASCEND_TOOLKIT_HOME and /usr/local/Ascend/ascend-toolkit/latest directories by default.

### Installation Mode Description

The installation modes include source code installation and pip source installation. You can select the installation mode as required.

 * [Source Code Installation](#source-code-installation)\: Use the source code to install the msit function of the latest version.
 * [pip source installation](#pip-source-installation)\: pip installs the msit package. Generally, the package is delivered once a quarter.

For details about common errors, see.[FAQs about msit installation](#faq-qa)    

#### Source Code Installation

```shell
git clone https://gitcode.com/Ascend/msit.git
#1. Run git pull origin to update the latest code. 
cd msit/msit

#2. Install the msit package.
pip install .

#3. Run the following command to view the component name and install the component based on the service requirements:
#For details, see the function description of each component: (https://gitcode.com/Ascend/msit/tree/master/msit#%E5%90%84%E7%BB%84%E4%BB%B6%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D).
msit install -h

#4. To install the llm, run the following command:
msit install llm

#5. After the installation, run the msit check command to check whether the installation is successful.
msit check llm
```

**Note:**

 * Through the`msit install llm`When the LLM component is installed, the tool automatically downloads the nlohmann C++ JSON library on which the ATB precheck function (opcheck) depends. If you encounter any problem during the downloading process`ERROR: cannot verify xxx.com's certificate`If the certificate is untrusted, perform the following operations:
    
    (1) If the ATB pre-check function is not used, ignore the error. Other functions of the llm component can be used normally.
    
    (2) If the ATB pre-check function is required, manually download the nlohmann JSON library file, and then run the`--find-links`The parameter specifies the library file path. For example, if the library file is downloaded in`/root/pkg`In the directory, run the following command:`msit install llm --find-links=/root/pkg`command to install the LLM component. The nlohmann JSON library can be downloaded from the following website:

| File Name      | Download URL                                                      | SHA256 Sum                                                       |
| -------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| v3.11.1.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.1.tar.gz | 598becb62ee0e01cf32795073c8ae09b6e95335cd43a4417b785d93ce105b0d0 |
| v3.11.2.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz | d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273 |
| v3.11.3.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz | 0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406 |

#### pip source installation

```shell
#1. Install the msit package.
pip install msit

#2. Run the following command to view the component name and install the component based on the service requirements:
#For details, see the function description of each component: (https://gitcode.com/Ascend/msit/tree/master/msit#%E5%90%84%E7%BB%84%E4%BB%B6%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D).
msit install -h

#3. To install the llm, run the following command:
msit install llm

#4. After the installation, run the msit check command to check whether the installation is successful.
msit check llm
```

> In Windows, only the Surgeon component can be installed.

## Uninstalling

Run the pip uninstallation command to uninstall the software.

1. It is recommended that you check the installed msit components first.
    
    ```bash
    #linux 
    pip list | grep -E "acl|msit|ais"
    ```
    
    ```shell
    #Windows
    pip list | findstr msit
    ```
    
    Result reference:
    
    ```log
    (base) PS C:\workspce\msit> pip list | findstr msit
    msit-surgeon             7.0.0rc2
    msit                  7.0.0rc2
    ```

2. Run the pip command to uninstall the components.
    
    ```shell
    #Uninstall only one component.
    pip uninstall msit-surgeon
    #Uninstall all
    pip uninstall msit-surgeon msit
    ```
    
    > If you uninstall only a few components, do not uninstall the msit. Otherwise, other components will be affected. Note: Ais-bench and aclruntime are also installed when the benchmark is installed. Therefore, the msit benchmark is uninstalled only during the uninstallation. You also need to manually uninstall ais-bench and aclruntime.

# Offline Installation (Linux Only)

Some users may need to install the msit on a computer that is not connected to the network. The offline installation guide is as follows:

## Download all software packages

This step needs to be performed on a machine that can be connected to the network. (Ensure that the two servers have the same system, platform, and Python version.). Install the msit first, and then run the msit download command to download the dependent packages.

```bash
#1. Install the msit first and use the source code mode.
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit
pip install .

#2. Run the msit download command to download the corresponding component to the directory.
#2.1 Download only certain components. For example, download llm to the ./pkg-cache directory.
msit download llm --dest ./pkg-cache 
#2.2 Downloading All Components
msit download all --dest ./pkg-cache
```

## Offline machine installation

In this step, you need to copy the related files, including the msit source code and downloaded dependency packages, to the offline host and start the installation.

```bash
#1. Install the msit first.
cd msit/msit
pip install .

#2. Install the components, for example, llm.
msit install llm --find-links=./pkg-cache

#3. After the installation, run the msit check command to check whether the installation is successful.
msit check all
```

**Note:**

 * Offline installation is not supported in Windows.
 * For offline installation, you need to download the installation package first. Ensure that the computer where the installation package is downloaded is the same as the computer where the installation package is installed offline. including the Python version, platform, and operating system.
 * Offline installation does not support the llvm installation.

# Installation-related command parameters

## Installation command: msit install

| Parameter name   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Mandatory. |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| comp_names       | is a location parameter, which specifies the components to be installed. Currently, the options are as follows:`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`, the specific options can be`msit install --help`view. But specified as`all`, indicates that all components need to be installed.                                                                                                                                                                                                                                                                                                                              | Yes        |
| --find-links, -f | Path of the search package, which is usually used during offline installation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | No.        |
| --no-check       | Specifies whether to skip the certificate information check of the target website when the benchmark function is installed and the dependency package is downloaded. Only when an error is reported during installation, the system displays a message indicating that the system needs to be installed.`use --no-check-certificate`, the --no-check parameter is used. If this parameter is used, the system skips checking the certificate information of the target website, which poses security risks. Exercise caution when using this parameter and bear the consequences. By default, this parameter is not configured, indicating that the certificate information is checked. | No.        |
| --help, -h       | Help information  | No.        |

```bash
msit install llm
```

## Installation check command: msit check

| Parameter name | Description                                                                                                                                                                                                                                                                                                                                                  | Mandatory. |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| comp_names     | is a location parameter, which specifies the component to be checked. Currently, the following options are available:`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`, the specific options can be`msit check --help`view. But specified as`all`, indicates that all components need to be checked. | Yes        |
| --help, -h     | Help information                                                                                                                                                                                                                                                                                                                                             | No.        |

```bash
msit check all
```

## Additional build command: msit build-extra

After some components are installed, some additional building actions are required. This step is automatically executed in msit install. However, the build may fail due to some reasons, for example, the pre-package is not installed. After the installation, you can run the msit check command to check the installation. A message is displayed. You can use msit build-extra to rebuild.

| Parameter name   | Description                                                                                                                                                                                                                                                              | Mandatory. |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| comp_names       | is a location parameter, which specifies the component to be built. Currently, the following options are available:`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`, you can use the`msit build-extra --help`View the | Yes        |
| --find-links, -f | Path of the search package, which is usually used during offline installation.                                                                                                                                                                                           | No.        |
| --help, -h       | Help information                                                                                                                                                                                                                                                         | No.        |

```bash
msit build-extra llm
```

## Download command: msit download

Download the installation package for offline installation.

| Parameter name | Description                                                                                                                                                                                                                                                                         | Mandatory. |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| comp_names     | is a location parameter, which specifies the components to be downloaded. Currently, the following options are available:`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`graph`, the specific options can be`msit download --help`View the | Yes        |
| --dest         | Target Path                                                                                                                                                                                                                                                                         | No.        |
| --help, -h     | Help information                                                                                                                                                                                                                                                                    | No.        |

```bash
msit download llm
```

# FAQ Q&A

[Reference: FAQs About MSite Installation](./FAQ.md)    
