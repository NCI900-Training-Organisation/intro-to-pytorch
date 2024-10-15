#!/bin/bash

# Load shell environment variables
source ~/.bashrc

module load python3/3.11.0  
module load cuda/12.3.2

. /scratch/vp91/Training-Venv/pytorch/bin/activate
 
# Application script
APPLICATION_SCRIPT=/scratch/vp91/jxj900/intro-to-pytorch/src/multinode_torchrun.py
 
# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}
 
# Run PyTorch application
torchrun --nnodes=${1} --nproc_per_node=${2} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${3}:29400 ${APPLICATION_SCRIPT}
