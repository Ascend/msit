
# FAQ

## 1. Error When Importing torch: cannot allocate memory in static TLS block

**Error Message** ImportError: {site-packages path}/torch.libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLS block

**Cause** This error typically occurs when the Thread-Local Storage (TLS) space is insufficient. This issue is common when using libraries such as cv2, torch, or libgomp.

**Solution**

1. Use the LD_PRELOAD environment variable to preload the related library.

    ```shell
    # Find the file location
    find / -name libgomp-6e1a1d1b.so.1.0.0
    # Add the file path to the LD_PRELOAD environment variable (the actual path depends on the output of the previous command; the path below is for reference only)
    export LD_PRELOAD=$LD_PRELOAD:/root/anaconda3/envs/test/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
    ```

2. Upgrade the glibc version to 2.32 or later. Take the Ubuntu system as an example:

    ```shell
    ldd --version # Check the glibc version
    sudo apt-get update
    sudo apt-get install libc6
    ```
