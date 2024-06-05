
source scripts/ltvu/node-setup.sh

HYDRA_FULL_ERROR=1 \
python -B run.py \
    batch_flag="'$batch_flag'" \
    model=groundvqa_b \
    dataset=egovlp_internvideo \
    'dataset.nlq_train_splits=[NaQ,NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=6 \
    dataset.num_workers=16 \
    optim.optimizer.lr=$lr \
    trainer.max_epochs=6 \
    trainer.gpus=$SLURM_GPUS_ON_NODE \
    trainer.enable_progress_bar=$enable \
    optim.lr_scheduler=True
