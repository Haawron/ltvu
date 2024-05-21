HYDRA_FULL_ERROR=1 python run.py \
    model=groundvqa_b \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=4 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    dataset=egovlp_internvideo_llavacap \
    '+model.model_variant=t5_ca'
