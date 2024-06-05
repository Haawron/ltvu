import math
import json
from pathlib import Path
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

from pytorch_lightning import seed_everything


seed_everything(42)


METHODS = [
    'binary',
    'bimodal_plus1'
]


def worker(i):
    stride, fps = 16, 30  # sec, frames/sec
    D = 768
    sample = ann[i]
    sample_id = sample['sample_id']
    p_out = p_out_dir / f'{sample_id}.pt'
    clip_duration = sample['clip_duration']
    num_features = int(clip_duration * fps / stride)
    s, e = sample['clip_start_sec'], sample['clip_end_sec']
    s_idx = int(s * fps / stride)
    e_idx = int(np.clip(math.ceil(e * fps / stride), s_idx+1, num_features-1))
    if method == 'binary':
        z_env = torch.zeros(num_features, D, dtype=torch.float32)
        z_env[s_idx:e_idx] = 1.
    elif method == 'bimodal_plus1':
        z_env = torch.randn(num_features, D, dtype=torch.float32)
        z_env[s_idx:e_idx] += 1.
    else:
        raise ValueError(method)
    env_pack = [(i*stride, z_env[i]) for i in range(num_features)]
    torch.save(env_pack, p_out)


def make_cheat_env():
    pass
    stride, fps = 16, 30  # sec, frames/sec
    D = 768
    for sample in tqdm(ann):
        sample_id = sample['sample_id']
        p_out = p_out_dir / f'{sample_id}.pt'
        # if p_out.exists():
        #     continue
        clip_duration = sample['clip_duration']
        num_features = int(clip_duration * fps / stride)
        s, e = sample['clip_start_sec'], sample['clip_end_sec']
        s_idx = int(s * fps / stride)
        e_idx = int(np.clip(math.ceil(e * fps / stride), s_idx+1, num_features-1))
        if method == 'binary':
            z_env = torch.zeros(num_features, D, dtype=torch.float32)
            z_env[s_idx:e_idx] = 1.
        elif method == 'bimodal':
            z_env = torch.randn(num_features, D, dtype=torch.float32)
            z_env[s_idx:e_idx] += 1.
        env_pack = [(i*stride, z_env[i]) for i in range(num_features)]
        torch.save(env_pack, p_out)


def main():
    num_samples = len(ann)
    pbar = trange(num_samples, dynamic_ncols=True, position=0, leave=True)
    num_workers = min(cpu_count(), num_samples)
    print(f'Found {num_samples} clips. Using {num_workers} workers.')
    print(f'Save dir: {p_out_dir}')
    print('\n')
    with Pool(num_workers) as pool:
        for _ in pool.imap_unordered(worker, range(num_samples)):
            pbar.update(1)


if __name__ == '__main__':
    method_idx = 1
    method = METHODS[method_idx]
    p_out_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/{method_idx:02d}_cheat_env_{method}')
    p_out_dir.mkdir(parents=True, exist_ok=True)
    ann = sum([
        json.load(open(f'data/unified/annotations.NLQ_{split}.json'))
        for split in ['train', 'val']
    ], [])
    main()
