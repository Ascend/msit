# FAQ

 * [1. Q: What should I do if the installation fails and the message "find no cann path" is displayed?](#1-q-what-should-i-do-if-the-installation-fails-and-the-message-find-no-cann-path-is-displayed)    
 * [2. Q: When the ./install.sh command is used for installation, the -bash: ./install.sh: Permission denied message is displayed.](#2-q-when-the-installsh-command-is-used-for-installation-the--bash-installsh-permission-denied-message-is-displayed)    
 * [3. Q: The common error XXX requires YYY, which is not installed.](#3-q-the-common-error-message-xxx-requires-yyy-which-is-not-installed-is-displayed)    
 * [4. Q: When the ./install.sh command is used, the error message "/usr/bin/env: 'bash\\r': No such file or directory" is displayed.](#4-q-when-the-installsh-command-is-used-the-following-error-message-is-displayed-usrbinenv-bashr-no-such-file-or-directory)    
 * [5. Q: How to obtain the CAN package path?](#5-q-how-do-i-obtain-the-cann-package-path)    
 * [6. Q: If the dependency packages in the environment are damaged by others or other tools, the message "pkg_resources.VersionConflict:XXXXX" is displayed when the msit is used.](#6-q-what-can-i-do-if-the-message-pkg_resourcesversionconflictxxxxx-is-displayed-when-i-use-the-msit-after-the-msit-is-installed-but-the-dependency-packages-in-the-subsequent-environment-are-damaged-by-others-or-other-tools)
 * [7. Q: When the msit is installed, the skl2onnx component fails to be installed.](#7-q-when-the-msit-is-installed-the-skl2onnx-component-fails-to-be-installed)    
 * [8. Q: OpenSSL: error:1408F10B:SSL routines:ssl3_get_record:wrong version number](#8-q-opensslerror1408f10bssl-routinesssl3_get_recordwrong-version-number)
 * [9. Q: If the problem occurs during use`No module named 'acl'`Check whether the environment variables of the CANN package are correct.](#9-q-if-the-problem-occurs-during-useno-module-named-aclcheck-whether-the-environment-variables-of-the-cann-package-are-correct)    
 * [10. Q: If the following message is displayed during the installation, the problem is: WARNING: env ACLTRANSFORMER_HOME_PATH is not set. Dump on demand package cannot be used.](#10-q-if-the-following-message-is-displayed-during-the-installation-the-system-displays-the-following-information-warning-env-acltransformer_home_path-is-not-set-dump-on-demand-package-cannot-be-used)    

## 1. Q: What should I do if the installation fails and the message "find no cann path" is displayed?

Installation error:

![输入图片说明](https://foruda.gitee.com/images/1686801650121824710/b64bf91e_9570626.png)    

**A: After the installation, you can set the CANN_PATH environment variable to specify the path of the CNAN version, for example, export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest/. If this parameter is not set, the tool attempts to obtain the CANN version from the ASCEND_TOOLKIT_HOME and /usr/local/Ascend/ascend-toolkit/latest directories by default.**

The following is a general method for setting the environment variables of the CANN package (assuming that the CANN package installation directory is`ACTUAL_CANN_PATH`):

 * Run the following command:
    
    ```bash
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```

## 2. Q: When the ./install.sh command is used for installation, the -bash: ./install.sh: Permission denied message is displayed

**A: This is because the execute permission is not granted to the install.sh script.**

```bash
#Adding Permissions
chmod u+x install.sh

#Or use
bash install.sh
```

## 3. Q: The common error message XXX requires YYY, which is not installed is displayed

![which is not installed](https://foruda.gitee.com/images/1686645293870003179/234cf67c_8913618.png)    A: This problem is caused by the lack of dependency on the local installation package. It is not an error reported by the msit. Install the software as prompted.

```bash
pip3 install YYY
```

## 4. Q: When the ./install.sh command is used, the following error message is displayed: /usr/bin/env: 'bash\\r': No such file or directory

![No such file or directory](./No_such_file.png)    

**A: This is not a file error. The common cause is that the format of the code is changed by default in the local compiler. In the lower right corner of the Pycharm editor, the format of the .sh file is changed from CRLF to LF.**![CRLF改为LF](https://foruda.gitee.com/images/1686645370968699210/f44f04b3_8913618.png)    

## 5. Q: How do I obtain the CANN package path?

**A: In this command, export \| grep ASCEND_HOME_PATH will output all environment variables and pass the result to the grep command through pipe characters. The grep command looks for the line that contains ASCEND_HOME_PATH and passes the result to the cut command. The cut command extracts the value of the second field, ASCEND_HOME_PATH, separated by equal signs, and outputs the value.**

```bash
echo $ASCEND_HOME_PATH
```

## 6. Q: What can I do if the message "pkg_resources.VersionConflict:XXXXX" is displayed when I use the msit after the msit is installed but the dependency packages in the subsequent environment are damaged by others or other tools

![输入图片说明](./VersionConflict.png)    

**A: This indicates that the version of the dependency package of the msit may be upgraded to an incorrect version. You only need to reinstall the msit. That is, run the following command in the msit/msit directory:**

```bash
./install.sh
```

Or carry out

```bash
pip3 check
```

Check the versions of the python components in the environment and manually install the python components to the corresponding version. For example, the following check result indicates that the protobuf version does not match. In this case, reinstall the corresponding version.

![输入图片说明](https://foruda.gitee.com/images/1686887221107606902/a0872e5b_9570626.png)    

Execution

```bash
pip3 install protobuf==3.20.2
```

## 7. Q: When the msit is installed, the skl2onnx component fails to be installed

![输入图片说明](https://foruda.gitee.com/images/1688461726292472393/721044b8_8277365.png)    A: Solution 1: Replace the pip source and manually install skl2onnx.

```bash
pip3 install skl2onnx==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

Solution 2: Install the wheel package.

Download[skl2onnx](https://pypi.tuna.tsinghua.edu.cn/packages/5e/59/0a47737c195da98d33f32073174b55ba4caca8b271fe85ec887463481f67/skl2onnx-1.14.1-py2.py3-none-any.whl)    Then, in the downloaded directory, run the following command:

```bash
pip3 install skl2onnx-1.14.1-py2.py3-none-any.whl
```

## 8. Q: OpenSSL:error:1408F10B:SSL routines:ssl3_get_record:wrong version number

**A: Solution: This problem is caused by the network proxy. Generally, configure the proxy as a private proxy and reinstall the msit. The proxy format is as follows:**

```bash
export http_proxy="http://User name:password@proxy address"
export https_proxy="http://User name:password@proxy address"
```

Note: The password must be escaped using the URL.

## 9. Q: If the problem occurs during use`No module named 'acl'`Check whether the environment variables of the CANN package are correct

 * **A: Solution:**
    
    > The following is a general method for setting the environment variables of the CANN package (assuming that the CANN package installation directory is`ACTUAL_CANN_PATH`):
    > 
    >* Run the following command:
    
    ```bash
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```
    
    >* Common user`ACTUAL_CANN_PATH`Generally, the`$HOME`. Generally, the root user is`/usr/local`

## 10. Q: If the following message is displayed during the installation, the system displays the following information: WARNING: env ACLTRANSFORMER_HOME_PATH is not set. Dump on demand package cannot be used

**A: If the large model precision comparison function is not used, ignore this alarm.**
