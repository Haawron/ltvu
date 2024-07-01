#!/bin/bash

#SBATCH --job-name=sbert-ft
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

############################## Captioner = all-mpnet-base-v2 ##############################

# # llava-image / SBert
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.freeze_embeddings=True \
#     model.alpha_loss_q_cap=1.0 \
#     model.alpha_loss_cap_cap=0.5 \
#     model.model_name='sentence-transformers/all-mpnet-base-v2' \
#     dataset.max_num_caps=96 \
#     dataset.max_cap_len=256 \

# # llava-video / SBert
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.freeze_embeddings=False \
#     model.alpha_loss_q_cap=1.0 \
#     model.alpha_loss_cap_cap=0.5 \
#     dataset.caption_pair_relation=adjacency \
#     model.model_name='sentence-transformers/all-mpnet-base-v2' \
#     dataset.max_num_caps=128 \
#     dataset.max_cap_len=256 \
#     dataset.captioner_name='LLaVA-NeXT-Video-7B-DPO' \

# # VideoRecap / SBert
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.freeze_embeddings=True \
#     model.model_name='sentence-transformers/all-mpnet-base-v2' \
#     dataset.max_num_caps=256 \
#     dataset.max_cap_len=64 \
#     dataset.captioner_name='VideoRecap' \

############################## Captioner = 'all-MiniLM-L6-v2' ##############################

# llava-image / SBert
HYDRA_FULL_ERROR=1 python -B run.py \
    batch_flag=\'$batch_flag\' \
    trainer.max_epochs=200 \
    optim.optimizer.lr=3e-4 \
    model.freeze_embeddings=False \
    model.alpha_loss_q_cap=1.0 \
    model.alpha_loss_cap_cap=0.0 \
    model.model_name='sentence-transformers/all-MiniLM-L6-v2' \
    dataset.max_num_caps=96 \
    dataset.max_cap_len=256 \
    others.save_top_k=5 \
    model.loss_fn_q_cap=bcewl \

############################## Captioner = EgoVLP ##############################

# # llava-image / EgoVLP
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.alpha_loss_q_cap=1.0 \
#     model.alpha_loss_cap_cap=0.5 \
#     dataset.caption_pair_relation=adjacency \
#     model.freeze_embeddings=True \
#     model.model_name=egovlp \
#     dataset.max_num_caps=96 \
#     dataset.max_cap_len=256 \

# # llava-video / EgoVLP
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.alpha_loss_q_cap=1.0 \
#     model.alpha_loss_cap_cap=0.5 \
#     model.freeze_embeddings=True \
#     model.model_name=egovlp \
#     dataset.max_num_caps=128 \
#     dataset.max_cap_len=256 \
#     dataset.captioner_name='LLaVA-NeXT-Video-7B-DPO' \

# # VideoRecap / EgoVLP
# HYDRA_FULL_ERROR=1 python -B run.py \
#     batch_flag=\'$batch_flag\' \
#     trainer.max_epochs=100 \
#     optim.optimizer.lr=1e-4 \
#     model.freeze_embeddings=True \
#     model.model_name=egovlp \
#     dataset.max_num_caps=256 \
#     dataset.max_cap_len=64 \
#     dataset.captioner_name='VideoRecap' \
#     +trainer.limit_train_batches=100 \
