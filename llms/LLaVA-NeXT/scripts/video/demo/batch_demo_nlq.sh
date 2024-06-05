#!/bin/bash

#SBATCH --job-name=EgoNLQ-LLaVA-NeXT-Video-7B-DPO
#SBATCH --array=0-31
#SBATCH --output=logs/slurm/%A-%a--%x.log
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH -x ariel-k[1,2],ariel-m1,ariel-v[6-13],ariel-g[1,3]

date +%Y-%m-%d/%H:%M:%S
hostname
echo

RANK=$SLURM_ARRAY_TASK_ID
WORLDSIZE=$SLURM_ARRAY_TASK_COUNT

sleep $RANK

HF_HUB_CACHE="/data/shared_cache/hf/hub" \
python playground/demo/video_demo.py --rank $RANK --world-size $WORLDSIZE
