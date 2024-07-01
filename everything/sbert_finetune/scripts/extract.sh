#!/bin/bash

#SBATCH --job-name=sbert-extract
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

target_jobids=(
    102720
    103204
    103250
    103248
    103336
)
notnow=(
    103329
    103327
    103345
)
for target_jobid in "${target_jobids[@]}"; do
    ckpt=$(find outputs -name "$target_jobid" -type d -exec find {} -name '*.ckpt' \;)
    hydra_dir=$(find outputs -name "$target_jobid" -type d -exec bash -c 'echo $(dirname $(dirname {}))/.hydra' \;)
    echo "Checkpoint: $(realpath $ckpt)"
    echo "Hydra dir: $(realpath $hydra_dir)"

    HYDRA_FULL_ERROR=1 python -B run.py \
        --config-path $(realpath $hydra_dir) --config-name config.yaml \
        batch_flag=\'$batch_flag\' \
        +ckpt_path=$(realpath $ckpt | sed 's/=/\\=/g') \

    if [ $? -ne 0 ]; then
        echo "Failed to extract features for $target_jobid"
        break
    fi

    sleep 1
done
