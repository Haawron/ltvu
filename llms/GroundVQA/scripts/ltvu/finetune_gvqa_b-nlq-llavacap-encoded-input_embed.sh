#!/bin/bash

#SBATCH --job-name=fft-gvqa-llavacap-X-input_embed
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

source scripts/ltvu/node-setup.sh

HYDRA_FULL_ERROR=1 \
python -B run.py \
    batch_flag="'$batch_flag'" \
    model=input_embed \
    dataset=egovlp_internvideo_llavacap \
    dataset.nlq_train_splits='[NLQ_train]' \
    dataset.test_splits='[NLQ_val]' \
    dataset.batch_size=6 \
    dataset.num_workers=8 \
    optim.optimizer.lr=1e-5 \
    trainer.accumulate_grad_batches=4 \
    trainer.max_epochs=20 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    trainer.enable_progress_bar=$enable \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    # optim.lr_scheduler=True \
