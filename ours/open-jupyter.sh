#!/bin/bash

srun -x 'ariel-k[1,2],ariel-m1' \
    --gres=gpu:1 \
    --cpus-per-gpu=8 \
    --mem-per-gpu=52G \
    -p debug_grad \
    -t 4:00:00 \
    --pty \
    /data/gunsbrother/prjs/ltvu/ours/open-jupyter-remote.sh
