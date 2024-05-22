# tar -xvf /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar -C /local_datasets/ego4d_data/v2/

# enable = True if 'batch' in $SLURM_JOB_PARTITION else False
if [[ $SLURM_JOB_PARTITION == *"batch"* ]]; then
    enable=False
else
    enable=True
fi

bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
    /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar \
    /local_datasets/ego4d_data/v2/

# HYDRA_FULL_ERROR=1 python run.py \
python run.py \
    model=groundvqa_b \
    dataset=egovlp_internvideo_llavacap \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=4 \
    dataset.num_workers=16 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    trainer.enable_progress_bar=$enable \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    '+model.model_variant=t5_ca' \
    '+model.t5_ca_layer_idxs=[4]' \
