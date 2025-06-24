# Copyright 2024 Huawei Technologies Co., Ltd
#!/bin/bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1200
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1
export HF_ENDPOINT=https://hf-mirror.com
TIMESTAMP=$(date +"%y%m%d%H%M")

accelerate launch  \
 --mixed_precision bf16 \
 --num_processes 8 \
 --num_cpu_threads_per_process 2 flux_train_network.py \
 --pretrained_model_name_or_path /path/to/flux/flux1-dev.safetensors \
 --clip_l /path/to/flux/clip_l_bf16.safetensors \
 --t5xxl /path/to/flux/t5xxl_fp16.safetensors \
 --ae /path/to/flux/ae.safetensors \
 --mixed_precision bf16 \
 --save_model_as safetensors \
 --output_dir output \
 --output_name flux-lora \
 --sdpa --persistent_data_loader_workers \
 --max_data_loader_n_workers 4 \
 --seed 42 \
 --save_precision bf16 \
 --network_module networks.lora_flux \
 --network_dim 128 \
 --learning_rate 1e-5 \
 --guidance_scale 1.0 \
 --loss_type l2 \
 --timestep_sampling sigmoid \
 --network_train_unet_only \
 --max_train_epochs 400 \
 --save_every_n_epochs 400 \
 --dataset_config demo.toml \
 --highvram \
 --cache_text_encoder_outputs \
 --cache_text_encoder_outputs_to_disk \
 --cache_latents_to_disk \
 --deepspeed \
 --zero_stage 2>&1 | tee ./logs/flux_$TIMESTAMP.log