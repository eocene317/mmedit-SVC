#!/bin/bash
#SBATCH --job-name="ecnqp32" # 名称
#SBATCH --nodes=1 # 申请节点数
#SBATCH -c 40 # 核数
#SBATCH --output=ch.log # 日志文件
#SBATCH -p gpu2Q # partition
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:2 # gpu数
#SBATCH -A pi_zhanghao

sleep 60