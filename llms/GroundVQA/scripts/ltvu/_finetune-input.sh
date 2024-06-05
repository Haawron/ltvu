#!/bin/bash

#SBATCH --job-name=fft-input
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

# Checkpoints
ckpt_path='checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt'
# ckpt_path='checkpoints/GroundVQA_B-NLQ-VLG-val_R1_03=15.5.ckpt'

# Environments
# llava_dir='data/features/llava-v1.6-34b/multi-qa-mpnet-base-dot-v1'
# llava_dir='data/features/LLaVA-NeXT-Video-7B-DPO/multi-qa-mpnet-base-dot-v1'
llava_dir='data/features/00_cheat_env_binary'
# llava_dir='data/features/01_cheat_env_bimodal_plus1'

HYDRA_FULL_ERROR=1 \
python -B run.py \
    batch_flag=\'$batch_flag\' \
    model=input_embed \
    model.model_variant=input_concat \
    model.env_ext_variant=id \
    dataset.nlq_train_splits='[NLQ_train]' \
    dataset=egovlp_internvideo_llavacap_sbert \
    dataset.batch_size=6 \
    dataset.llava.llava_dir=\'$llava_dir\' \
    dataset.llava.feature_aggregation=cross \
    optim.optimizer.lr=1e-5 \
    trainer.max_epochs=20 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.enable_progress_bar=$enable \
    +trainer.checkpoint_path=\'$ckpt_path\' \
    # optim.lr_scheduler=True \
