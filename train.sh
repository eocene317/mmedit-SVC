#!/bin/bash
#SBATCH --job-name="ecnqp32" # 名称
#SBATCH --nodes=1 # 申请节点数
#SBATCH -c 20 # 核数
#SBATCH --output=train-ecn-qp32.txt # 日志文件
#SBATCH -p gpu2Q # partition
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH -A pi_zhanghao

export MASTER_PORT=$((12000 + $RANDOM % 20000))
date
# module load /public/software/Modules/modulefiles/compiler/CUDA/10.2
# module load /public/software/Modules/modulefiles/compiler/GNU/gcc-10.1.0
module load CUDA/10.2
module load GNU/gcc-10.1.0
source /public/software/anaconda3/bin/activate mmedit

cd /public/home/hpc204711073/code/mmedit-SVC
PORT=$MASTER_PORT tools/dist_train.sh configs/restorers/ecn/edvrm_x2_g8_600k_reds.py 1