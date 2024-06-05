#!/bin/bash

#SBATCH --job-name=GVQA-default
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

python run.py \
    batch_flag=\'$batch_flag\' \
    'model=groundvqa_b' \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    'dataset.batch_size=6' \
    'optim.optimizer.lr=1e-12' \
    trainer.max_epochs=3 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    'trainer.load_nlq_head=True' \
    trainer.enable_progress_bar=$enable \
    +trainer.limit_train_batches=5 \

	# '+model.ignore_decoder=True'
