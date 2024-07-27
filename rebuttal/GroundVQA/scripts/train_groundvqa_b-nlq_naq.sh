python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=8
