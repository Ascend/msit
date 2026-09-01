# FAQ

- [1. Q: Installation fails with the message "find no cann path". How to handle this?](#1-q-installation-fails-with-the-message-find-no-cann-path-how-to-handle-this)
- [2. Q: Using ./install.sh for installation but getting -bash: ./install.sh: Permission denied](#2-q-using-installsh-for-installation-but-getting--bash-installsh-permission-denied)
- [3. Q: Common error XXX requires YYY, which is not installed.](#3-q-common-error-xxx-requires-yyy-which-is-not-installed)
- [4. Q: Using ./install.sh, getting the error: /usr/bin/env: 'bash\r': No such file or directory.](#4-q-using-installsh-getting-the-error-usrbinenv-bashr-no-such-file-or-directory)
- [5. Q: How to obtain the CANN package path?](#5-q-how-to-obtain-the-cann-package-path)
- [6. Q: msit was working after installation, but later the dependency packages in the environment were damaged by others or other tools. When using msit, the message "pkg_resources.VersionConflict:XXXXX" is displayed. What should I do?](#6-q-msit-was-working-after-installation-but-later-the-dependency-packages-in-the-environment-were-damaged-by-others-or-other-tools-when-using-msit-the-message-pkg_resourcesversionconflictxxxxx-is-displayed-what-should-i-do)
- [7. Q: OpenSSL: error:1408F10B:SSL routines:ssl3_get_record:wrong version number](#7-q-openssl-error1408f10bssl-routinesssl3_get_recordwrong-version-number)
- [8. Q: If `No module named 'acl'` occurs during use, verify whether the CANN package environment variables are correct.](#8-q-if-no-module-named-acl-occurs-during-use-verify-whether-the-cann-package-environment-variables-are-correct)
- [9. Q: If the following message appears during installation: WARNING: env ACLTRANSFORMER_HOME_PATH is not set. Dump on demand package cannot be used.](#9-q-if-the-following-message-appears-during-installation-warning-env-acltransformer_home_path-is-not-set-dump-on-demand-package-cannot-be-used)

## 1. Q: Installation fails with the message "find no cann path". How to handle this?

Installation error:

![Input image description](https://foruda.gitee.com/images/1686801650121824710/b64bf91e_9570626.png "Screenshot")

**A:** After installation, you can specify the installed CANN version path by setting the CANN_PATH environment variable, for example: export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest/. If not set, the tool attempts to obtain the CANN version from the environment variable ASCEND_TOOLKIT_HOME and the path /usr/local/Ascend/ascend-toolkit/latest respectively.

The following is a general method to set the CANN package environment variables (assuming the CANN package installation directory is `ACTUAL_CANN_PATH`):

* Execute the following command:

    ```bash
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```

## 2. Q: Using ./install.sh for installation but getting -bash: ./install.sh: Permission denied

**A:** This is because the execute permission is not added to install.sh.

```bash
# Add permission
chmod u+x install.sh

# Or use
bash install.sh
```

## 3. Q: Common error XXX requires YYY, which is not installed

![which is not installed](https://foruda.gitee.com/images/1686645293870003179/234cf67c_8913618.png "Screenshot")
**A:** This is caused by missing dependencies in the local installation package. It is not an msit error. Install the dependencies as prompted by the command line.

```bash
pip3 install YYY
```

## 4. Q: Using ./install.sh, getting the error: /usr/bin/env: 'bash\r': No such file or directory

![No such file or directory](./No_such_file.png "Screenshot")

**A:** This is not a file error. The common cause is that the file format was changed by default in the local editor. In the PyCharm editor, change the .sh file format from CRLF to LF in the bottom right corner.
![CRLF to LF](https://foruda.gitee.com/images/1686645370968699210/f44f04b3_8913618.png "Screenshot")

## 5. Q: How to obtain the CANN package path?

**A:** In this command, export | grep ASCEND_HOME_PATH outputs all environment variables and passes the result to the grep command through a pipe. The grep command searches for lines containing ASCEND_HOME_PATH and passes the result to the cut command. The cut command uses the equal sign as the delimiter, extracts the second field, which is the value of ASCEND_HOME_PATH, and outputs it.

```bash
echo $ASCEND_HOME_PATH
```

## 6. Q: msit was working after installation, but later the dependency packages in the environment were damaged by others or other tools. When using msit, the message "pkg_resources.VersionConflict:XXXXX" is displayed. What should I do?

![Input image description](./VersionConflict.png "Screenshot")

**A:** This indicates that the dependency package version of msit may have been upgraded to a mismatched version. Reinstalling msit is sufficient, that is, execute the following in the msit/msit directory again:

```bash
./install.sh
```

Or execute

```bash
pip3 check
```

to check which Python components in the environment have version dependency mismatches, and manually install the corresponding versions. For example, the following check result indicates a protobuf version mismatch. Reinstall the corresponding version:

![Input image description](https://foruda.gitee.com/images/1686887221107606902/a0872e5b_9570626.png "Screenshot")

Execute

```bash
pip3 install protobuf==3.20.2
```

## 7. Q: OpenSSL: error:1408F10B:SSL routines:ssl3_get_record:wrong version number

**A:**
Solution: This is a network proxy issue. Generally, configure the proxy as a personal proxy and reinstall msit. The proxy format is as follows:

```bash
export http_proxy="http://username:password@proxy_address"
export https_proxy="http://username:password@proxy_address"
```

Note: The password must be URL-escaped.

## 8. Q: If `No module named 'acl'` occurs during use, verify whether the CANN package environment variables are correct

- **A:** Solution:
    > The following is a general method to set the CANN package environment variables (assuming the CANN package installation directory is `ACTUAL_CANN_PATH`):
    >
    > * Execute the following command:

    ```bash
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```

    > * For a regular user, `ACTUAL_CANN_PATH` is usually `$HOME`. For the root user, it is usually `/usr/local`.

## 9. Q: If the following message appears during installation: WARNING: env ACLTRANSFORMER_HOME_PATH is not set. Dump on demand package cannot be used

**A:** If you do not use the large model precision comparison function, ignore this warning.
