
source scripts/ltvu/node-setup.sh

HYDRA_FULL_ERROR=1 \
python -B run.py \
    batch_flag="'$batch_flag'" \
    model=groundvqa_b \
    dataset=egovlp_internvideo_llavacap \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.llava.feature_aggregation=init \
    dataset.batch_size=6 \
    dataset.num_workers=16 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.enable_progress_bar=$enable \
    '+model.model_variant=t5_ca' \
    '+model.t5_ca_layer_idxs='"[6,7,8,9,10,11]" \

    # trainer.load_nlq_head=False \
    # '+model.freeze_all_but_ca=True' \
    # '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
