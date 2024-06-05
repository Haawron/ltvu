#!/bin/bash

#SBATCH --job-name=llava-v1.6-34B-extract-nlq-local
#SBATCH --array=0-7
#SBATCH --output=logs/slurm/%A-%a--%x.log
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH -x ariel-g[1-5],ariel-v[7,10,13],ariel-k[1,2],ariel-m1

date +%Y-%m-%d/%H:%M:%S
hostname
echo

WORLDSIZE=$SLURM_ARRAY_TASK_COUNT
RANK=$SLURM_ARRAY_TASK_ID

# source .env
cache_dir='/data/shared_cache/hf/hub/'
local_cache_dir='/data2/local_datasets/shared_cache/hf/hub'
target_model_dirname='models--liuhaotian--llava-v1.6-34b'

lock_file=$local_cache_dir/.lock-"${SLURM_ARRAY_JOB_ID}"
target_dir="$local_cache_dir/$target_model_dirname"

if [ ! -d "$target_dir" ]; then
    mkdir -p $local_cache_dir
    if [ ! -f "$lock_file" ]; then
        touch "$lock_file"
        echo "Copying model from $cache_dir/$target_model_dirname to $local_cache_dir/"
        cp -r "$cache_dir/$target_model_dirname" "$local_cache_dir/"
        rm "$lock_file"
    else
        while [ -f "$lock_file" ]; do
            echo "Waiting for lock to be released..."
            sleep 10
        done
    fi
fi

HF_HUB_CACHE=/data2/local_datasets/shared_cache/hf/hub
# HF_HOME=

python quickstart2.py --rank $RANK --world-size $WORLDSIZE
