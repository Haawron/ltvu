#!/bin/bash

#SBATCH --job-name=ltvu
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=batch_grad
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:8
#SBATCH --time=1-0
#SBATCH --mem-per-gpu=42G


./run.sh $@
