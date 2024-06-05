#!/bin/bash

#SBATCH --job-name=full-finetune_gvqa_b-nlq-llavacap-encoded
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -x ariel-k[1,2],ariel-m1

date +%Y-%m-%d/%H:%M:%S
hostname
echo

script_path='scripts/ltvu/finetune_gvqa_b-nlq-llavacap-encoded-t5_ca.sh'
cat $script_path
. $script_path
