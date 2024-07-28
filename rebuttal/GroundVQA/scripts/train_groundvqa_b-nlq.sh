#!/bin/bash

#SBATCH --job-name=ours-no_act-finetune-lr_scale
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
export log_every_n_steps=$([[ $batch_flag == 1 ]] && echo 10 || echo 1)

# LR scaling
export base_lr=0.0001
export base_bsz=$(( $SLURM_GPUS_ON_NODE * 16 ))
export bsz_per_gpu=6
export bsz=$(( $SLURM_GPUS_ON_NODE * $bsz_per_gpu ))
export lr=$(python -c "print(f'{""$base_lr / $base_bsz * $bsz"":.5e}')")
echo "Base LR: $base_lr, Base BSZ: $base_bsz, LR: $lr, BSZ: $bsz"

ckpt='/data/gunsbrother/prjs/ltvu/llms/GroundVQA/GroundVQA/GroundVQA_B-NLQ-VLG-val_R1_03=15.5.ckpt'
HYDRA_FULL_ERROR=1 python run.py \
    model=groundvqa_b \
    dataset.nlq_train_splits='[NLQ_train]' \
    dataset.test_splits='[NLQ_val]' \
    dataset.batch_size=$bsz_per_gpu \
    trainer.log_every_n_steps=$log_every_n_steps \
    trainer.gpus=${SLURM_GPUS_ON_NODE:-8} \
    trainer.max_epochs=30 \
    +trainer.checkpoint_path="'$ckpt'" \
    optim.optimizer.lr="$lr" \
