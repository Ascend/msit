# FAQ

## 1. If a non-root user uses the /usr/local/Ascend/ascend-toolkit file in the root directory when using the analyze tool, the fast_query shell fails to be invoked

 * Error message:

```text
2023-06-16 09:23:47,490 INFO : convert model to json, please wait...
2023-06-16 09:24:01,852 INFO : convert model to json finished.
2023-06-16 09:24:04,998 INFO : try to convert model to om, please wait...
2023-06-16 09:24:28,326 INFO : try to convert model to om finished.
2023-06-16 09:24:29,190 ERROR : load opp data failed, err:exec fast_query shell failed, err:2023-06-16 09:24:29 [ERROR] The input path may be insecure because it does not belong to you.
.
2023-06-16 09:24:29,247 INFO : analysis result has been written in out/result.csv.
2023-06-16 09:24:29,247 INFO : number of abnormal operators: 13.
2023-06-16 09:24:29,248 INFO : analyze model finished.
```

 * Analysis of the error cause:
    
    Currently, the analyzer invokes the operator quick check tool in the CANN package during model support analysis. The security check of tool files requires that the user who invokes the operator quick check tool script and the owner of the script must be the same person. Therefore, when a non-root user uses a file in the /usr/local/Ascend/ascend-toolkit directory under the root directory, the file security check of the analyze tool cannot be passed. Therefore, the file cannot be invoked.
 * Solution:
    
    A non-root user installs the CANN developer suite package in the /home/userxxx/ directory, configures the environment variables as prompted, and runs the analyze tool.
