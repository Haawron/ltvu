srun -x 'ariel-g[1-5],ariel-k[1-2]' \
    --gres=gpu:1 \
    --cpus-per-gpu=8 \
    --mem-per-gpu=29G \
    -p debug_grad -t 4:00:00 \
    --pty $(which python) slurm/scripts/generate_short_term_captions.py
