# tar -xvf /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar -C /local_datasets/ego4d_data/v2/
if [ ! -d '/local_datasets/ego4d_data/v2/' ] && [ ! -L '/local_datasets/ego4d_data/v2/' ]; then
    mkdir -p /data2/local_datasets/ego4d_data/v2/
    ln -s /data2/local_datasets/ego4d_data /local_datasets/
fi

# bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
#     /data/datasets/tarfiles/egonlq-llava-v1.6-34b-global-only.tar \
#     /local_datasets/ego4d_data/v2/features/

# bash /data/gunsbrother/helpful-scripts/exclusive_copy.sh \
#     data/unified/egovlp_internvideo.hdf5 \
#     /local_datasets/ego4d_data/v2/features/egovlp_internvideo.hdf5

# bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
#     /data/datasets/tarfiles/egonlq-egovlp-internvideo-clip_feats.tar \
#     /local_datasets/ego4d_data/v2/features/

bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
    /data/datasets/tarfiles/egonlq-llava-v1.6-34b-cross-encoded.tar \
    /local_datasets/ego4d_data/v2/features/

bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
    /data/datasets/tarfiles/egonlq-egoenv.tar \
    /local_datasets/ego4d_data/v2/features/

bash /data/gunsbrother/helpful-scripts/exclusive_untar.sh \
    /data/datasets/tarfiles/egonlq-llava-v1.6-34b-sbert-encoded.tar \
    /local_datasets/ego4d_data/v2/features/

export batch_flag=$(scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])')
export enable=$([[ $batch_flag == 1 ]] && echo False || echo True)

# LR scaling
# export base_lr=${1:-'1e-5'}
# export base_bsz=$(( 16 * 8 ))
# export bsz=$(( $SLURM_GPUS_ON_NODE * 6 ))
# export lr=$(python -c "print(f'{""$base_lr / $base_bsz * $bsz"":.5e}')")
# echo "Base LR: $base_lr, Base BSZ: $base_bsz, LR: $lr, BSZ: $bsz"
