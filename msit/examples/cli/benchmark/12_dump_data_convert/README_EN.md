# Dump data convert #

## Introduction ##

Automatically converts the .bin file of the dump result to the .npy file.

## Running an Example ##

```
msit benchmark --om-model ./pth_resnet50_bs1.om --output ./output --dump 1 --dump-npy 1
```

An example of the output is as follows:

```
output/
    |-- 2023_01_03-06_35_53
    |-- 2023_01_03-06_35_53_summary.json
    |-- dump/
        |--20230103063551/
        |--20230103063551_npy/
```

In the dump directory, the original 20230103063551 subdirectory stores the bin file, and the converted 20230103063551_npy subdirectory contains the converted npy file.

