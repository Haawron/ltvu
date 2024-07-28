#!/bin/bash

#SBATCH --job-name=reproduce
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

export batch_flag=$(scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])')
export enable=$([[ $batch_flag == 1 ]] && echo False || echo True)

HYDRA_FULL_ERROR=1 python run.py \
    model=groundvqa_b \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=6 \
    trainer.enable_progress_bar=$enable \
    trainer.gpus=${SLURM_GPUS_ON_NODE:-8} \
    trainer.max_epochs=30 \
    model.enable_infuser=False \
