srun \
    -x 'ariel-v[10,13],ariel-k[1,2],ariel-m1' \
    --gres=gpu:4 \
    --cpus-per-gpu=8 \
    --mem-per-gpu=29G \
    -p debug_grad \
    -t 4:00:00 \
    --pty \
    bash scripts/finetune_groundvqa_b-nlq-llavacap.sh
