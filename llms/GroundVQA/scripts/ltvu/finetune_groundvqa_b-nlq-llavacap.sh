
source scripts/ltvu/node-setup.sh

HYDRA_FULL_ERROR=1 \
python -B run.py \
    model=inter_ca \
    dataset=egovlp_internvideo_llavacap \
    batch_flag="'$batch_flag'" \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.llava.feature_aggregation=all \
    dataset.batch_size=6 \
    dataset.num_workers=16 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    trainer.enable_progress_bar=$enable \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
