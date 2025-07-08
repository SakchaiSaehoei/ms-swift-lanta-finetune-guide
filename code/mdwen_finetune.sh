#!/bin/bash
#SBATCH -p gpu                         
#SBATCH -N 1                            
#SBATCH --ntasks-per-node=1            
#SBATCH -c 32                           
#SBATCH --gpus-per-node=4              
#SBATCH -t 24:00:00                     
#SBATCH -A lt200344                     
#SBATCH -J ms-swift                   
ml Mamba     
mamba activate ms-swift
LOG_DIR="logs"
mkdir -p $LOG_DIR
# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/mdwen-8b_lora_sft_${TIMESTAMP}.log"
# 设置CUDA设备
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#Supervised Fine-Tuning
swift sft \
    --model '/project/lt200344-zhthmt/Y/.cache/modelscope/models/Qwen/Qwen3-8B' \
    --train_type lora \
    --dataset '/project/lt200344-zhthmt/Y/MS-SWIFT/data/ready_to_train.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --split_dataset_ratio 0 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --weight_decay 0.01 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ../swift_output/MDwen3-8B-Lora-SFT \
    --dataloader_num_workers 256 \
    --model_author sakchaisaehoei \
    --model_name MDwen3-8B-Lora-SFT \
    > "$LOG_FILE" 2>&1 
# 打印进程ID和日志文件位置
echo "Training started with PID $!"
echo "Log file: $LOG_FILE"
# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"