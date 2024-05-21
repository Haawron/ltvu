python run.py \
    'model=groundvqa_b' \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    'dataset.batch_size=6' \
    'optim.optimizer.lr=1e-5' \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    'trainer.load_nlq_head=True' \

	# '+model.ignore_decoder=True'
