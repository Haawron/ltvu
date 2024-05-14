import os
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

import cv2
from decord import VideoReader, cpu


def get_table():
    dfs = [
        pd.read_csv('EgoNLQ/csvs/nlq_train_v2.csv'),
        pd.read_csv('EgoNLQ/csvs/nlq_val_v2.csv')
    ]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def process_clip(clip_uid, i, num_clips):
    df = get_table()
    df_clip = df[df['clip_uid']==clip_uid]
    clip_uids = set(df_clip['clip_uid'].tolist())
    assert len(clip_uids) == 1, clip_uids
    df_clip = df_clip.iloc[0]
    video_uid = df_clip['video_uid']
    p_video = p_videos_dir / f'{video_uid}.mp4'

    vr = VideoReader(str(p_video), ctx=cpu(0))
    fps = vr.get_avg_fps()
    h, w, _ = vr[0].shape

    s_ind, e_ind = df_clip['video_start_frame'], df_clip['video_end_frame']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    p_clip = p_clips_dir / f'{clip_uid}.mp4'
    print(f'[{i:3d}/{num_clips}] Processing {str(p_clip)} ...')
    out = cv2.VideoWriter(str(p_clip), fourcc, fps, (w, h))
    for frame_idx in range(s_ind, e_ind+1):
        frame = vr[frame_idx].asnumpy()
        frame = frame[..., ::-1]
        out.write(frame)

    out.release()


def parallel_process(not_done_clip_uids, num_workers):
    num_clips = len(not_done_clip_uids)
    with Pool(num_workers) as pool:
        pool.starmap(process_clip, [(clip_uid, i, num_clips) for i, clip_uid in enumerate(not_done_clip_uids)])


if __name__ == '__main__':
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/')
    p_videos_dir = Path('/data/datasets/ego4d_data/v2/video_320p/')
    df = get_table()

    clip_uids = set(df['clip_uid'].tolist())
    done_clip_uids = set(p_clip.stem for p_clip in p_clips_dir.glob('*.mp4'))
    not_done_clip_uids = clip_uids - done_clip_uids
    print(len(clip_uids), len(done_clip_uids), len(not_done_clip_uids))
    num_workers = os.cpu_count()
    parallel_process(not_done_clip_uids, num_workers)
