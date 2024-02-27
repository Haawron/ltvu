ckpt_path=$2
eval_split_name=test
device_id=$1
eval_id=$3
eval_path=data/mad_data/test.jsonl
echo ${eval_path}
CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python cone/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--eval_id ${eval_id} \
--num_workers 8 \
${@:4}
