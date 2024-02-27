import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Union

import torch
import numpy as np
from PIL import Image
from decord import VideoReader

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, batched_predict as predict


def load_annotations(p_completed_dir: Path = Path('egonlq/results/completed')):
    p_ann_dir = Path('/data/datasets/ego4d_data/v2/annotations')
    p_json_train, p_ann_val = p_ann_dir / 'nlq_train.json', p_ann_dir / 'nlq_val.json'
    ann_train, ann_val = json.load(p_json_train.open()), json.load(p_ann_val.open())
    ann_videos = ann_train['videos'] + ann_val['videos']
    # 생김새: ann_videos[0]['clips'][0]['annotations'][0]['language_queries'][0]

    # list of {'clip_uid': clip_uid, 'queries': list of query dicts}
    ann_clips: List[Dict[str, Union[str,List[dict]]]] = []
    clip_uid_to_idx: Dict[str, int] = {}
    clip_idx = 0
    for ann_video in ann_videos:
        for ann_clip in ann_video['clips']:
            clip_uid: str = ann_clip['clip_uid']
            all_clip_language_queries: List[List[dict]] = [
                annotation['language_queries']
                for annotation in ann_clip['annotations']]
            all_clip_language_queries: List[dict] = sum(
                all_clip_language_queries, start=[])
            ann_clips.append({'clip_uid': clip_uid, 'queries': all_clip_language_queries})
            clip_uid_to_idx[clip_uid] = clip_idx
            clip_idx += 1

    for p_completed in p_completed_dir.glob('**/*.pkl'):
        clip_uid_completed = p_completed.stem
        del clip_uid_to_idx[clip_uid_completed]

    n_clips = len(ann_clips)
    n_queries = sum(map(lambda ann_clip: len(ann_clip['queries']), ann_clips))
    print(f'\n[ All Train & Valid ] # clips: {n_clips:4d}, # queries: {n_queries:5d}')  # 1686 (v1은 1326), 18403 (v1은 11291 + 3874 = 15165)

    n_clips = len(clip_uid_to_idx)
    n_queries = sum(map(
        lambda ann_clip: len(ann_clip['queries']) if ann_clip['clip_uid'] in clip_uid_to_idx else 0,
        ann_clips))
    print(f'[  To be processed  ] # clips: {n_clips:4d}, # queries: {n_queries:5d}')

    return ann_clips, clip_uid_to_idx


def preprocess_frame(frame: np.ndarray, transform) -> Tuple[np.ndarray, torch.Tensor]:
    frame = Image.fromarray(frame)
    frame_transformed, _ = transform(frame, None)
    return frame, frame_transformed


def preprocess_video(frames: np.ndarray, transform) -> Tuple[np.ndarray, torch.Tensor]:
    frames_transformed = []
    for frame in frames:
        frame, frame_transformed = preprocess_frame(frame, transform)
        frames_transformed.append(frame_transformed)
    frames_transformed = torch.stack(frames_transformed)
    return frames, frames_transformed


def fmts2hms(seconds) -> str:
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f'{hours:2d} h {minutes:2d} m {seconds:2d} s'


class Worker:
    def __init__(self, p_clips_dir, rank, bsz=32):
        self.p_clips_dir = p_clips_dir
        self.rank = rank
        self.bsz = bsz
        self.chunk_size = 1024
        self.prefetch_factor = self.chunk_size // bsz
        assert self.chunk_size % self.prefetch_factor == 0

        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25

        self.model = load_model(
            "groundingdino/config/GroundingDINO_SwinB_cfg.py",
            "weights/groundingdino_swinb_cogcoor.pth")
        self.ann_clips, self.clip_uid_to_idx = load_annotations()
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),  # deterministic
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def do_work(self, clip_uid: str):
        p_clip = self.p_clips_dir / f'{clip_uid}.mp4'
        vr = VideoReader(str(p_clip))
        ann_clip = self.ann_clips[self.clip_uid_to_idx[clip_uid]]
        queries = ann_clip['queries']
        words = list(set(word.lower() for query in queries if (word:=query.get('slot_x', None)) is not None))  # slot_x can be null or there may not be slot_x
        text_prompt = ' . '.join(words) + ' .'  # 프롬프트는 클립 안에서는 동일함

        p_result = Path(f'./egonlq/results/buffer/{clip_uid}.pkl')
        print(f'\nClip: {clip_uid}, length: {len(vr)}\n')
        print(text_prompt)
        print()
        results = []
        t0 = time.time()
        for chunk_idx, chunk_frame_offset in enumerate(range(0, len(vr), self.chunk_size)):
            chunk: np.ndarray = vr[chunk_frame_offset : chunk_frame_offset + self.chunk_size].asnumpy()[..., ::-1]
            for batch_idx, batch_frame_offset in enumerate(range(0, chunk.shape[0], self.bsz)):
                batch_global_frame_offset = chunk_frame_offset + batch_frame_offset
                frames = chunk[batch_frame_offset : batch_frame_offset + self.bsz]
                frames_source, frames = preprocess_video(frames, self.transform)
                batch_results = predict(
                    model=self.model,
                    images=frames,
                    caption=text_prompt,
                    box_threshold=self.BOX_THRESHOLD,
                    text_threshold=self.TEXT_THRESHOLD
                )
                results += batch_results

            t1 = time.time()
            frame_count = chunk_frame_offset + self.chunk_size
            print(f'[{frame_count:5d}/{len(vr):5d}] Ellapsed: {fmts2hms(t1-t0)} | ETA: {fmts2hms((t1-t0)/frame_count*len(vr))}')

        with p_result.open('wb') as f:
            pickle.dump(results, f)

        print(f'Clip UID {clip_uid} done.')


def main(rank, world_size):
    import traceback
    import logging
    assert 0 <= rank < world_size
    p_nlqv1_clips_dir = Path(f'/data/datasets/ego4d_data/v2/clips_320p-non_official')
    clip_uids: List[str] = [p_clip.stem for p_clip in p_nlqv1_clips_dir.glob('*.mp4')]
    jobs: List[str] = sorted(clip_uids)
    njobs = len(jobs)  # 1657
    job_clip_indices: List[str] = np.arange(rank, njobs, world_size)

    # clip_uid = '000eba33-8d14-446a-b016-19bd50e9a3b9'
    worker = Worker(p_nlqv1_clips_dir, rank, bsz=16)
    p_not_proceesed_dir = Path('egonlq/not_processed')
    p_not_proceesed_dir.mkdir(exist_ok=True, parents=True)
    f_not_processed = open(p_not_proceesed_dir / f'{rank}.log', 'w')
    for jobid in job_clip_indices:
        clip_uid = jobs[jobid]
        try:
            worker.do_work(clip_uid)
        except Exception as e:
            # clip_uid KeyError => test dataset
            # .join TypeError => slot_x null
            print(clip_uid, file=f_not_processed, flush=True)
            logging.error(traceback.format_exc())
            logging.error(clip_uid + '\n\n')


if __name__ == '__main__':
    # without submitit, for test
    main(0, 1)
