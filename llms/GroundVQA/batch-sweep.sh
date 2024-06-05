#!/bin/bash

#SBATCH --job-name=GVQA-llava-v1.6-34B-sweeps
#SBATCH --output=logs/slurm/%A-%a--%x.log
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --array=0-14%2
#SBATCH --begin=now+5hour
#SBATCH -x ariel-v[10,13],ariel-k[1,2],ariel-m1

date +%Y-%m-%d/%H:%M:%S
hostname
echo

RANK=$SLURM_ARRAY_TASK_ID
sweep0=('init' 'answer' 'all')
sweep1=('[6]' '[11]' '[0,1,2,3,4,5,6,7,8,9,10,11]' '[6,7,8,9,10,11]' '[9,10,11]')
sweep0=${sweep0[$((RANK % 3))]}
sweep1=${sweep1[$((RANK / 3))]}

echo "sweep0: $sweep0"
echo "sweep1: $sweep1"

bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
    /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar \
    /local_datasets/ego4d_data/v2/

python run.py \
    model=groundvqa_b \
    dataset=egovlp_internvideo_llavacap \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.llava.feature_aggregation="$sweep0" \
    dataset.batch_size=4 \
    dataset.num_workers=16 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    trainer.enable_progress_bar=False \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    '+model.model_variant=t5_ca' \
    '+model.t5_ca_layer_idxs='"$sweep1" \
    '+model.freeze_all_but_ca=True' \


exit 0
