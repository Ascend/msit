export PYTHONPATH=/data/deepseek/ModelEvalState:$PYTHONPATH #adapter
pkill -9 'mindie'
ps -ef | grep 'python'
ps -ef | grep 'mindie'
npu-smi info
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon        