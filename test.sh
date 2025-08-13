config_file=./config/vamoe.yaml
config='vamoe'
run_num='1'

NAME='vit_rebuttal_zquvt5_70_0.0002AdamW_CosLR_trainl2loss0.1_useMoE_ChannelMoE_GradClip_patch2channel768_cl_0523'


LOG_DIR="./logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

WEIGHTS="./logs/${NAME}/vamoe/1/training_checkpoints/ckpt.tar"

CUDA_VISIBLE_DEVICES=0 nohup python test.py \
            --yaml_config=$config_file --config=$config --run_num=$run_num --override_dir=$LOG_DIR \
            --weights=$WEIGHTS > ${LOG_DIR}test.log 2>&1 &
            