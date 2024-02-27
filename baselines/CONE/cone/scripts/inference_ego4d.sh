device_id=$1
ckpt_path=$2
eval_id=$3
eval_split_name=val
eval_path=data/ego4d_data/val.jsonl
echo ${eval_path}
CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python cone/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--eval_id ${eval_id} \
${@:4}
