沟通事项
block sparse工作量 block size是不是会变长
TE分离是不是能引导用local
自研学习框架效果，GPU上开发的 npu上适配了
全局变量交付时间  改什么分支 常用什么分支  问题单

mrope和普通rope有什么区别

rope interleave什么意思

shm

# Copyright 2024 Huawei Technologies Co., Ltd
#!/bin/bash
# Source CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set Logs
mkdir -p ./logs
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0

# Set Perf
# export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export ACLNN_CACHE_LIMIT=100000

# Set Mem
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Set Training Configs
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path ./path_to_your_data \
  --output_path ./output_models \
  --dit_path "./Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 4000 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --dataloader_num_workers 16 \
  --use_gradient_checkpointing | tee logs/train_${logfile}.log 2>&1
chmod 440 logs/train_${logfile}.log
set +x
