#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

if [ -z "$SLURM_JOB_ID" ]; then
    echo "SLURM_JOB_ID is not set."
    exit 1
fi

python=/home/ltvu/anaconda3/envs/ltvu/bin/python
modulename='ltvu.models.without_rgb.lightning_modules'
random_unused_port=$(shuf -i 10000-65535 -n 1)

if [ -z "$SLURM_CPUS_ON_NODE" ]; then
    SLURM_CPUS_ON_NODE=1
fi
if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    python -Bm ${modulename} "${@:1}"
else
    OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE torchrun \
    --nnodes=1 \
    --master-port=$random_unused_port \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
        -m ${modulename} "${@:1}"
fi
