#!/bin/bash

#SBATCH --job-name=sbert-video
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -x ariel-k[1,2],ariel-m1,ariel-v12

date +%Y-%m-%d/%H:%M:%S
hostname
echo


export batch_flag=$(scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])')

# python -B -m pdb run.py \
HYDRA_FULL_ERROR=1 python -B run.py \
    batch_flag=\'$batch_flag\' \
    model.loss_fn_q_cap=multi-pos \
    dataset.max_num_caps=128 \
    trainer.max_epochs=100 \
    optim.optimizer.lr=1e-4 \
