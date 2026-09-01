
# FAQ

## 1. Error: Failed to Call the fast_query Shell When a Non-Root User Uses the analyze Tool with Files Under the Root User's /usr/local/Ascend/ascend-toolkit Directory

- Error message:

```bash
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

- Cause analysis:

    During model support analysis, the analyze tool calls the operator quick query tool in the CANN package for verification. The file security check of the tool requires that the user calling the operator quick query script be the same as the owner of the script. When a non-root user uses files under the root user's /usr/local/Ascend/ascend-toolkit directory, the file security check of the analyze tool fails. As a result, the call cannot be executed.

- Solution:

    Install the CANN developer toolkit package in the /home/userxxx/ directory as a non-root user, and configure the related environment variables correctly (follow the prompts after the CANN package installation is complete). Then run the analyze tool.
