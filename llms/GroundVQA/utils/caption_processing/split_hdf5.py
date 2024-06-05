from pathlib import Path
import h5py
import json
import torch
import multiprocessing as mp
from tqdm import tqdm


def worker(clip_uid: str):
    p_hdf5 = '/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/unified/egovlp_internvideo.hdf5'
    p_pt_target_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/egovlp_intervideo')
    video_features = h5py.File(p_hdf5, 'r')
    video_feature = torch.from_numpy(video_features[clip_uid][:])
    p_pt_target = p_pt_target_dir / f'{clip_uid}.pt'
    torch.save(video_feature, p_pt_target)


def main():
    p_jsons_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/unified')
    jobs = []
    for split in ['train', 'val', 'test_unannotated']:
        p_json = p_jsons_dir / f'annotations.NLQ_{split}.json'
        data = json.load(p_json.open())
        jobs_split = set(entry['video_id'] for entry in data)
        print(f'{split}: {len(jobs_split)}')
        jobs.extend(list(jobs_split))
    print(len(jobs))
    # worker(jobs[0])
    pbar = tqdm(total=len(jobs))
    with mp.Pool(64) as pool:
        for _ in pool.imap_unordered(worker, jobs):
            pbar.update(1)


if __name__ == '__main__':
    main()
