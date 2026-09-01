# Dump Data Convert

## Introduction

Automatically convert the dumped bin result files to npy files.

## Running Samples

```bash
msit benchmark --om-model ./pth_resnet50_bs1.om --output ./output --dump 1 --dump-npy 1
```

The output result sample is as follows:

```bash
    output/
    |-- 2023_01_03-06_35_53
    |-- 2023_01_03-06_35_53_summary.json
    |-- dump/
        |--20230103063551/
        |--20230103063551_npy/
```

In the dump directory, in addition to the original 20230103063551 subdirectory that saves bin files, there is also a converted 20230103063551_npy subdirectory that contains the converted npy files.
