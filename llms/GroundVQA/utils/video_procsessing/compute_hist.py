import os
import sys
from pathlib import Path
from tqdm import tqdm, trange
from multiprocessing import Pool, Queue

import torch
import decord
import numpy as np
from einops import rearrange


def compute_color_hist(frame, bins_per_color=16):
    H, W, C = frame.shape
    frame = rearrange(frame, 'h w c -> c h w')
    hists = []
    for channel in range(C):
        hist, _ = np.histogram(frame[channel].ravel(), bins=bins_per_color, range=(0, 255))
        hists.append(hist)
    hists = np.concatenate(hists, axis=-1)  # (C*bins)
    hists = hists / hists.sum()
    return hists


def worker(idx):
    rank = q_empty.get()
    p_clip = p_clips[idx]
    clip_uid = p_clip.stem
    p_hist_out = p_hist_outdir / f'{clip_uid}.pt'
    if p_hist_out.exists():
        tqdm.write(f'{p_hist_out.name} already exists. Skipping.')

    else:
        vr = decord.VideoReader(str(p_clip))
        frame_hists = []
        pbar = trange(len(vr), dynamic_ncols=True, leave=False, position=rank+1, desc=f'Rank {rank}')
        for i in pbar:
            frame = vr[i].asnumpy()
            frame_hist = compute_color_hist(frame)
            frame_hists.append(frame_hist)
        frame_hists = np.stack(frame_hists, axis=0)  # [T, C*bins]
        tensor = torch.from_numpy(frame_hists).float()
        torch.save(tensor, p_hist_out)

    q_empty.put(rank)
    return 1


def main():
    pbar = tqdm(p_clips, total=len(p_clips), dynamic_ncols=True, position=0, leave=True)
    num_clips = len(p_clips)
    num_workers = min(WORLD_SIZE, num_clips)
    print(f'Found {num_clips} clips. Using {num_workers} workers.')
    print(f'Save dir: {p_hist_outdir}')
    print('\n')
    with Pool(num_workers) as pool:
        for _ in pool.imap_unordered(worker, range(num_clips)):
            pbar.update(1)


if __name__ == '__main__':
    p_clips = sorted(Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/').glob('*.mp4'))
    p_hist_outdir = Path('/data/gunsbrother/prjs/ltvu/ours/data/Ego4D-processed/hists/bins-16')
    p_hist_outdir.mkdir(parents=True, exist_ok=True)
    WORLD_SIZE = os.cpu_count()
    q_empty = Queue()
    for i in range(WORLD_SIZE):
        q_empty.put(i)
    main()
