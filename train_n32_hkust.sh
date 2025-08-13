#!/bin/bash
### 启用 IB 通信
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=./conf/dump.xml

export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
### 获取每个节点的 hostname
for i in `scontrol show hostnames`
do
    let k=k+1
    host[$k]=$i
    echo ${host[$k]}
done

config_file=./config/vamoe.yaml
config='vamoe'
run_num='1'

NAME='vit_rebuttal_zquvt5_70_0.0002AdamW_CosLR_trainl2loss0.1_useMoE_ChannelMoE_GradClip_patch2channel768_cl_0523'

LOG_DIR="./logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

checkpoint=""

source export_DDP_vars.sh
source activate env

echo ${host[4]}

###主节点运行 
torchrun --nnodes=4 --node_rank=0 --nproc_per_node=4 --master_addr="${host[1]}" \
            --master_port="29501" --max_restarts=3 \
            train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR --checkpoint=$checkpoint \
            > ${LOG_DIR}train_1.log 2>&1 &


### 使用 srun 运行第二个节点
srun -N 1 --gres=gpu:4 -w ${host[2]} \
torchrun --nnodes=4 --node_rank=1 --nproc_per_node=4 --master_addr="${host[1]}" \
            --master_port="29501" --max_restarts=3 \
            train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR --checkpoint=$checkpoint \
            > ${LOG_DIR}train_2.log 2>&1 &


srun -N 1 --gres=gpu:4 -w ${host[3]} \
torchrun --nnodes=4 --node_rank=2 --nproc_per_node=4 --master_addr="${host[1]}" \
            --master_port="29501" --max_restarts=3 \
            train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR --checkpoint=$checkpoint \
            > ${LOG_DIR}train_3.log 2>&1 &


srun -N 1 --gres=gpu:4 -w ${host[4]} \
torchrun --nnodes=4 --node_rank=3 --nproc_per_node=4 --master_addr="${host[1]}" \
            --master_port="29501" --max_restarts=3 \
            train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR \
            > ${LOG_DIR}train_4.log 2>&1 &

wait


### --checkpoint=$checkpoint