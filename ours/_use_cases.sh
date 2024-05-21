python -Bm ltvu.models.without_rgb.lightning_modules

./run.sh \
--batch-size 16 \
--ckpt-path ''

./run.sh \
--model-version 1 \
--batch-size 32 \
--ckpt-path 'results/without_rgb/mips-trained/google--flan-t5-base/version_57/checkpoints/epoch=1-iogt>=0.3 R@05=0.6129.ckpt'

./run.sh \
--head_name af --batch-size 4 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 150 \
--proposal_width_sec_train 30 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1}'


# 그나마 잘 됨
./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": True, "train_cls_label_distance_smoothing_kernel": "rect"}'

./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": True, "train_cls_label_distance_smoothing_kernel": "rect", "train_cls_label_distance_smoothing_kernel_size": 5}'

./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 150 \
--proposal_width_sec_train 30 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": True, "train_cls_label_distance_smoothing_kernel": "gauss"}'

./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_compensate": True}'

##################################

./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--max_ctx_len 480 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": True, "train_cls_label_distance_smoothing_kernel": "rect"}' \
--ds-kws '{"caption_stride_sec": 1, "max_t": 480}'

./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--max_ctx_len 448 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": False}' \
--ds-kws '{"caption_stride_sec": 1, "max_t": 448}'

./run.sh \
--head_name af --batch-size 3 \
--language-model-name 'google/flan-t5-base' \
--proposal_width_sec 30 \
--proposal_width_sec_train 30 \
--max_ctx_len 128 \
--caption_stride_sec 2 \
--gather_consecutive_captions_factor 2 \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": False}' \
--ds-kws '{"max_t": 128}'


./run.sh \
--head_name af --batch-size 1 \
--language-model-name 'google/flan-t5-base' \
--proposal-mode False \
--lm-kws '{"enable_time_embed": False}' \
--head-kws '{"enable_input_linear": True,  "num_tx_layers": 1, "train_cls_label_distance_smoothing": True, "train_cls_label_distance_smoothing_kernel": "rect"}'
