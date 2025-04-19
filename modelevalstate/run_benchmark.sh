#!/usr/bin/bash
#  benchmark文件权限可能有问题，需要设置
export RANKTABLEFILE=/data/deepseek/ranktable.json # adapter
mkdir /tmp/ModelEvalState/output
find /usr/local/lib/python3.11/site-packages/mindie*  -name  config.json |xargs chmod -R 640
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"
benchmark --DatasetPath "/benchmark_data" \
  --DatasetType "gsm8k" \
  --ModelName llama3-8b \
  --ModelPath "/data/llama3-8b"  \
  --TestType client \
  --MaxOutputLen 256 \
  --Http http://127.0.0.1:8025 \
  --ManagementHttp http://127.0.0.1:8026 \
  --Concurrency $CONCURRENCY \
  --RequestRate 20 \
  --WarmupSize 1 \
  --Tokenizer True \
  --SaveTokensPath /tmp/ModelEvalState/output
