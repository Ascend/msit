# MindIE-Torch precision comparison scenario #

## 1. Dependency ##

 *  CANN (8.0RC3 or later)
 *  MindIE (1.0RC3 or later)

## 2. Install the msit. ##

 *  For details about how to install the msit, see.[Installing the msit tool](../../../../../../msit/docs/install/README.md)	

## 3. Dump data ##

 *  For details about MindIE-Torch data dump, see.[MindIE-Torch Scenario-Dumping Operator Data on the Entire Network](../../dump/07_mindie_torch_dump/README.md)	

## 4. Compare precision ##

 *  Execution`msit debug compare --golden-path [/path/to/cpu/dumpdata] --my-path [/path/to/npu/dumpdata] --output [path/to/csv] --ops-json [path/to/json]`. Output the comparison result file in CSV format.

### Parameter Description ###

| Parameter name     | Description                                                                                                         | Mandatory. |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------- |
| --golden-path, -gp | Root path of CPU/GPU dump data                                                                                      | Yes        |
| --my-path, -mp     | NPU Dump Data Root Path                                                                                             | Yes        |
| --ops-json         | Running the`msit debug dump`Path of the operator mapping file generated in, which is usually in the current folder. | Yes        |
| --output, -o       | Output path of the comparison result in the CSV file. The default path is the current path.                         | No.        |

## Attention. ##

 *  Currently, the operator precision comparison in the MindIE scenario supports TorchScript and Torch.export routes.

