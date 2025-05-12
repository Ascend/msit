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
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export ACLNN_CACHE_LIMIT=100000

# Set Mem
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Set Training Configs
GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS examples/wanvideo/train_wan_i2v.py \
  --task train \
  --train_architecture full \
  --dataset_path ./path_to_your_data  \
  --metadata_name metadata.json \
  --output_path ./output_models \
  --dit_path "[
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors\",
        \"./Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors\"
    ]"  \
  --steps_per_epoch 4000 \
  --max_epochs 10 \
  --learning_rate 2e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_worker 16 \
  --training_strategy "deepspeed_stage_3" | tee logs/train_${logfile}.log 2>&1
chmod 440 logs/train_${logfile}.log
set +x
