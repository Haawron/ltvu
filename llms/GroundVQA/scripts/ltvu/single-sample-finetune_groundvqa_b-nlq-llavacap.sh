# trainer.overfit_batches는 쓰면 안 됨
# Dataloader를 통째로 딥카피하는데 hdf5 오브젝트가 피클링이 안 돼서 프로세스 간 복사가 안 됨

HYDRA_FULL_ERROR=1 python run.py \
    model=groundvqa_b \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_train]' \
    batch_flag="'$batch_flag'" \
    dataset.batch_size=1 \
    optim.optimizer.lr=1e-5 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.load_nlq_head=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    '+trainer.num_sanity_val_steps=0' \
    dataset=egovlp_internvideo_llavacap \
    '+model.model_variant=t5_ca'


    # 원래 코드 1-sample check
    # '+trainer.limit_train_batches=1' \
    # '+trainer.val_check_interval=1.' \
    # '+trainer.check_val_every_n_epoch=1024' \
    # 'trainer.max_epochs=1024' \
