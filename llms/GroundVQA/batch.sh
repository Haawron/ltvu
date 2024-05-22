#!/bin/bash

#SBATCH --job-name=GVQA-llava-v1.6-34B
#SBATCH --output=logs/slurm/%A-%a--%x.log
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -x ariel-g[1-5],ariel-v[7,10,13],ariel-k[1,2],ariel-m1

date +%Y-%m-%d/%H:%M:%S
hostname
echo

# bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
#     /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar \
#     /local_datasets/ego4d_data/v2/

. scripts/finetune_groundvqa_b-nlq.sh

exit 0
