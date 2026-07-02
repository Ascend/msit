# Installing the msit tool

## Environment and Dependency

The installation of the msit inference tool includes the installation of the msit package and the dependent component packages. You can add only the required component packages as required.

| Dependent Software Name | Mandatory or not | Version                                                             | Remarks                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------- | ---------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN                    | Mandatory.       | You are advised to install CANN commercial version 6.0.1. or later. | For details, see the[CANN-6.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/601/overview/index.html)    Install the Ascend development or running environment, that is, the toolkit software package. After the installation, configure environment variables as prompted. For details, see.[Configuring Environment Variables](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0040.html)    ). |
| GCC                     | Mandatory.       | 7.3.0 version                                                       | For details, see the[GCC Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000086.html)    Install the GCC compiler. (By default, the GCC 4.8 compiler is used on CentOS 7.6. Therefore, this tool may not be installed. You are advised to update the GCC compiler before installing it.)                                                                                                  |
| Python                  | Mandatory.       | Supports Python3.7.5+, Python3.8.x, Python3.9.x, and Python3.10.x.  | Python 3.7.5 is required if the TensorFlow model precision comparison function is used.                                                                                                                                                                                                                                                                                                                                                             |
| TensorFlow              | Not mandatory.   | -                                                                   | Reference[Installing TensorFlow1.15.0 on CentOS 7.6](https://bbs.huaweicloud.com/blogs/181055)    Install the TensorFlow 1.15.0 environment. (If the TensorFlow precision comparison function is not used, this installation is not required.)                                                                                                                                                                                                         |
| Caffe                   | Optional.        | -                                                                   | Reference[Caffe Installation](http://caffe.berkeleyvision.org/installation.html)    Install the Caffe environment. (If the precision comparison function of the Caffe model is not used, the installation is not required.)                                                                                                                                                                                                                            |
| Clang                   | Optional.        | -                                                                   | Depends on LLVM Clang and needs to be installed.[Clang tool](https://releases.llvm.org/)    .                                                                                                                                                                                                                                                                                                                                                          |

## Installing the MSite

The installation modes include one-click installation of source code and manual installation of different components as required. You can select the installation mode as required.

 * [One-click installation of source code](#one-click-installation-of-source-code)    \: Installs all msit components in one-click mode.
 * [Manually install different components as required.](#manually-install-different-components-as-required)    \: You can select the required msit components as required and install them one by one.

For details about common errors, see.[FAQs About MSite Installation](#faq-qa)    

### Description

 * Install the drivers, firmware, and CANN packages related to the Ascend AI inference in the development running environment. For details, see.[CANN-6.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/601/overview/index.html)    . After the installation, you can set the CANN_PATH environment variable, for example, export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest, to specify the path of the installed CANN version. If this parameter is not set, the tool attempts to obtain the CANN version from the environment variables ASCEND_TOOLKIT_HOME and /usr/local/Ascend/ascend-toolkit/latest by default.

### One-click installation of source code

```shell
git clone https://gitcode.com/Ascend/msit.git
#1. Update the latest code by git pull origin. 
cd msit/msit

#2. Add the execute permission.
chmod u+x install.sh

#3. Select one install.sh script as required.
#a. Install the msit, including the debug, profile, benchmark, and analyze groups.
./install.sh
  
#b. Install the msit, including the debug, profile, benchmark, and analyze components. (The sudo permission is required for installing the system dependency libraries such as clang.)
./install.sh --full
  
#c. Reinstall the msit and its debug, profile, benchmark, and analyze components.
./install.sh --force-reinstall
```

### Manually install different components as required

```shell
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit

#Adding the execute permission
chmod u+x install.sh

#1. Install only the Surgeon component under debug.
./install.sh --surgeon

#2. Install only the compare component under debug. (By default, the benchmark component is installed due to the dependency.)
./install.sh --compare

#3. Install only the benchmark component.
./install.sh --benchmark

#4. Only the analyze component is installed.
./install.sh --analyze

#5. Install only the profile component.
./install.sh --profile

#6. Install only the convert component.
./install.sh --convert
```

# Uninstalling

Note: If the msit tool is downloaded before 2023/08/01, uninstall the msit tool and its subtools again.

```shell
cd msit/msit

chmod u+x install.sh

#1. Query uninstallation one by one
./install.sh --uninstall

#2. Uninstall all the components without query.
./install.sh --uninstall -y

#3. Query uninstallation of a single component (e.g., the Surgeon component)
./install.sh --uninstall --surgeon

#4. Directly uninstall a single component (such as the Surgeon component) without query.
./install.sh --uninstall --surgeon -y
```

# Windows environment

In Windows, only the Surgeon component can be installed.

## Installed

```shell
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit

#1. Install the msit, including the surgeon component.
install.bat

#2. Install only the surgeon component under debug.
install.bat --surgeon
```

## Uninstalling

Note: If the msit tool is downloaded before 2023/08/01, uninstall the msit tool and its subtools again.

```shell
cd msit/msit

#1. Query uninstallation one by one
install.bat --uninstall

#2. Uninstall all the components without query.
install.bat --uninstall -y

#3. Query uninstallation of a single component (e.g., the Surgeon component)
install.bat --uninstall --surgeon

#4. Directly uninstall a single component (such as the Surgeon component) without query.
install.bat --uninstall --surgeon -y
```

# FAQ Q&A

[Reference: FAQs About MSite Installation](./FAQ.md)    
