#!/usr/bin/bash
# 配置服务化环境变量
export MIES_CONTAINER_IP=90.91.105.45
echo "MIES_CONTAINER_IP=$MIES_CONTAINER_IP" # adapter
export MASTER_IP=90.91.105.45 # adapter
echo "MASTER_IP=$MASTER_IP"

# 环境变量相关
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export NETWORK_ADAPTER="enp189s0f0"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=0
export OMP_NUM_THREADS=1

# 关闭确定性计算，使能AIV
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

export RANKTABLEFILE=/data/deepseek/hccl_2s_16p.json # adapter
export MINDIE_LOG_TO_STDOUT=1
export MINDIE_LLM_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=2
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=1

# 关闭确定性计算，使能AIV
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

# 设置显存比
export NPU_MEMORY_FRACTION=0.96

chmod 640 $RANKTABLEFILE
echo "MASTER_IP=$MASTER_IP"
echo "MIES_INSTALL_PATH=$MIES_INSTALL_PATH"
sed -i "s/\"ipAddress\" *: \".*\"/\"ipAddress\": \"$MASTER_IP\"/" $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"managementIpAddress\" *: \".*\"/\"managementIpAddress\": \"$MASTER_IP\"/" $MIES_INSTALL_PATH/conf/config.json
grep ipAddress $MIES_INSTALL_PATH/conf/config.json
grep managementIpAddress $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"httpsEnabled\" *: .*/\"httpsEnabled\": false,/" $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"interCommTLSEnabled\" *: .*/\"interCommTLSEnabled\": false,/" $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"interNodeTLSEnabled\" *: .*/\"interNodeTLSEnabled\": false,/" $MIES_INSTALL_PATH/conf/config.json
grep httpsEnabled $MIES_INSTALL_PATH/conf/config.json
grep interCommTLSEnabled $MIES_INSTALL_PATH/conf/config.json
grep interNodeTLSEnabled $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"multiNodesInferEnabled\" *: .*/\"multiNodesInferEnabled\": true,/" $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"modelName\" *: .*/\"modelName\": \"$MODEL_NAME\",/" $MIES_INSTALL_PATH/conf/config.json
sed -i "s/\"modelWeightPath\" *: .*/\"modelWeightPath\": \"\/data\/deepseek\/$MODEL_NAME\",/" $MIES_INSTALL_PATH/conf/config.json
grep multiNodesInferEnabled $MIES_INSTALL_PATH/conf/config.json
grep modelName $MIES_INSTALL_PATH/conf/config.json
grep modelWeightPath $MIES_INSTALL_PATH/conf/config.json

# 日志的相关
#ASDOPS_LOG_LEVEL=INFO
#ASDOPS_LOG_TO_STDOUT=1
#MINDIE_LLM_PYTHON_LOG_TO_STDOUT=1
#MINDIE_LLM_LOG_TO_STDOUT=1
#MINDIE_LLM_LOG_LEVEL=DEBUG
#MINDIE_LOG_LEVEL=debug
#MINDIE_LOG_TO_STDOUT=1
#MINDIE_LOG_VERBOSE=1
#MIES_CERTS_LOG_TO_STDOUT=1
#MIES_CERTS_LOG_LEVEL=DEBUG

# profiler 采集
# 配置文件内容
#{
# "enable": 1,
# "prof_dir": "./prof_dir/",
# "profiler_level": "INFO"
#}
#export SERVICE_PROF_CONFIG_PATH=/usr/local/Ascend/mindie/latest/mindie-service/ms_service_profiler_config.json # msprofiler

export PYTHONPATH=/data/deepseek/ModelEvalState:$PYTHONPATH #adapter
pkill -9 'mindie'
ps -ef | grep 'python'
ps -ef | grep 'mindie'
npu-smi info
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon