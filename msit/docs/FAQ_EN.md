# FAQ

## 1. The error message "cannot allocate memory in static TLS block" is displayed during torch import

**The error message ImportError: \{site-packages path\}/torch.libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLS block is displayed.**

**Errors are usually caused by insufficient thread local storage (TLS) space, which is common when certain libraries (such as cv2, torch, or libgomp) are used.**

**Solution**

1. Use the LD_PRELOAD environment variable to preload related libraries.

    ```shell
    #Find this file location
    find / -name libgomp-6e1a1d1b.so.1.0.0
    #Add the file path to the LD_PRELOAD environment variable. (The specific path is determined by the command output in the previous step. The following path is for reference only.)
    export LD_PRELOAD=$LD_PRELOAD:/root/anaconda3/envs/test/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
    ```

2. Upgrade the glibc version to 2.32 or later. Ubuntu is used as an example.

    ```shell
    ldd --version #Check the glibc version.
    sudo apt-get update
    sudo apt-get install libc6
    ```
