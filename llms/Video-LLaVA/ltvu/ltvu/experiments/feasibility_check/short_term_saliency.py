import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ltvu.utils import *
from ltvu.objects import *
from ltvu.constants.prompts import SHORT_TERM_PROMPTS
prompt_type, prompts = SHORT_TERM_PROMPTS[-1]['type'], SHORT_TERM_PROMPTS[-1]['prompts']


logger = logging.getLogger(__name__)


class ShortTermCaptionGenerator:
    def __init__(
        self,
        p_output_dir: Path = Path('/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu/captions/test/step1')
    ):
        self.p_output_dir = p_output_dir

        self.model = None
        self.tokenizer = None
        self.video_processor = None
        self.assistant = None

        self.initialized = False

    def initialize(self):
        logger.info(f'Initializing {self.__class__.__name__}...')
        from ltvu.utils.load_model import load_model
        self.model, self.tokenizer, self.video_processor = load_model(return_wrapper=True)
        self.assistant = Assistant(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.initialized = True
        logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def dump_caption(q_caption, p_output_json: Path):
        with open(p_output_json, 'w') as f:
            json.dump(q_caption, f, default=Interval.to_dict)

    def caption_single(self,
        p_video: Path,
        s: TimestampType,
        e: TimestampType,
        query: str,
        p_output_json: Path = None
    ):
        assert p_video.exists(), f'{p_video} does not exist.'
        itvl = Interval(s, e)
        s, e = itvl.s, itvl.e
        logger.debug(f'caption_single: {p_video} {s.sec:.1f} ~ {e.sec:.1f}')
        p_trimmed = trim_video(p_video=p_video, s=s, e=e)
        tensor = self.video_processor(p_trimmed)
        captions: list[str] = []
        for prompt in prompts:
            prompt = prompt.format(query=query)
            logger.debug(f'prompt: {prompt}')
            output = self.assistant.converse(
                prompt=prompt,
                tensor=tensor,
                append_outputs=True,
                begin_with=None)
            logger.debug(f'output: {output}')
            captions.append(output)
        q_caption = {
            'interval': itvl,
            'captions': captions}
        if p_output_json is not None:
            self.dump_caption(q_caption, p_output_json)
        return q_caption

    def caption_with_sliding(self,
        p_video: Path,
        query: str,
        stride_sec: float = 3.,
        p_output_json: Path = None,
    ):
        duration_sec = get_video_length(p_video)
        timestamps = np.arange(0., duration_sec, stride_sec)  # drop last
        q_captions = []
        for s, e in zip(timestamps[:-1], timestamps[1:]):
            q_caption = self.caption_single(p_video=p_video, s=s, e=e, query=query)
            q_captions.append(q_caption)
            self.assistant.reset_conv()
        if p_output_json is not None:
            self.dump_caption(q_captions, p_output_json)
        return q_captions


def caption_sample(q_uid=None):
    if q_uid is None:
        from ltvu.experiments.feasibility_check.constants import SAMPLE_CLIP_UIDS
        clip_uid = SAMPLE_CLIP_UIDS[1]
        q_sample = df_nlq[df_nlq['clip_uid'] == clip_uid].iloc[0]
    else:
        q_sample = df_nlq[df_nlq['q_uid']==q_uid]
        clip_uid = q_sample['clip_uid'].values[0]

    itvl_gt = Interval(q_sample['q_clip_start_sec'], q_sample['q_clip_end_sec'])
    logger.debug(f'clip_uid: {clip_uid}')
    logger.debug(f'GT: {itvl_gt}')
    p_clip = p_video_dir / f'{clip_uid}.mp4'
    itvl_clip = Interval(10., 15.)
    query = q_sample['query']
    q_uid = q_sample['q_uid']
    p_output_json = p_output_dir / f'{clip_uid}/{q_uid}.json'
    p_output_json.parent.mkdir(parents=True, exist_ok=True)
    gen = ShortTermCaptionGenerator()
    gen.initialize()
    gen.caption_single(
        p_video=p_clip,
        s=itvl_clip.s,
        e=itvl_clip.e,
        query=query,
        p_output_json=p_output_json)
    logger.info(f'Captions for {str(p_clip)} saved in {str(p_output_json)}.')


def caption_sample_with_sliding(idx_clip=0, idx_query=0, q_uid=None):
    if q_uid is None:
        from ltvu.experiments.feasibility_check.constants import SAMPLE_CLIP_UIDS
        clip_uid = SAMPLE_CLIP_UIDS[idx_clip]
        q_sample = df_nlq[df_nlq['clip_uid']==clip_uid].iloc[idx_query]
    else:
        q_sample = df_nlq[df_nlq['q_uid']==q_uid].iloc[0]
        clip_uid = q_sample['clip_uid']

    itvl_gt = Interval(q_sample['q_clip_start_sec'], q_sample['q_clip_end_sec'])
    logger.debug(f'clip_uid: {clip_uid}')
    logger.debug(f'GT: {itvl_gt}')
    p_clip = p_video_dir / f'{clip_uid}.mp4'
    query = q_sample['query']
    q_uid = q_sample['q_uid']
    p_output_json = p_output_dir / f'{clip_uid}/{q_uid}.json'
    p_output_json.parent.mkdir(parents=True, exist_ok=True)
    gen = ShortTermCaptionGenerator()
    gen.initialize()
    output = {
        'gt': itvl_gt,
        'query': query,
        'prompts': prompts}
    q_captions: list[dict] = gen.caption_with_sliding(
        p_video=p_clip,
        query=query)
    output['captions'] = q_captions
    gen.dump_caption(output, p_output_json)
    logger.info(f'Captions for {str(p_clip)} saved in {str(p_output_json)}')


def caption_samples():
    from ltvu.experiments.feasibility_check.constants import SAMPLE_CLIP_UIDS
    from ltvu.slurm.submitter import Submitter, wait
    from tqdm import tqdm
    import shutil
    import time

    Path('/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu/slurm/logs').mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, clip_uid in enumerate(SAMPLE_CLIP_UIDS):
        nqueries = df_nlq[df_nlq['clip_uid'] == clip_uid].shape[0]
        for j in range(nqueries):
            jobs.append({'idx_clip': i, 'idx_query': j})
    njobs = len(jobs)
    world_size = ngpus = min(16, njobs)
    ranks = list(range(world_size))
    job_chunks = [jobs[rank::world_size] for rank in ranks]

    p_tmp_check_dir = Path('/tmp/video-llava/check-done')
    if p_tmp_check_dir.exists():
        shutil.rmtree(p_tmp_check_dir)
    p_tmp_check_dir.mkdir(parents=True, exist_ok=False)

    def _worker(rank: int):
        _jobs: list[dict] = job_chunks[rank]
        for job in _jobs:
            caption_sample_with_sliding(**job)
            p_done = p_tmp_check_dir / f'{rank:03d}-{job["idx_clip"]:03d}-{job["idx_query"]:03d}.done'
            p_done.touch()
            pbars[rank].update(1)
            pbars[-1].update(1)

    submitter = Submitter(slurm_array_parallelism=ngpus)
    sjobs = submitter.submit_jobs(_worker, ranks)

    pbars = [
        tqdm(
            total=len(job_chunks[rank]),
            desc=f'Rank {rank:03d}',
            position=world_size-rank, leave=True)
        for rank in ranks]
    pbars.append(tqdm(total=njobs, desc=f'Total', position=0, leave=True))

    while True:
        if len(list(p_tmp_check_dir.glob('*.done'))) == njobs:
            break
        pbars[-1].refresh()
        time.sleep(1.)
    tqdm._instances.clear()

    print('All jobs are done.')


if __name__ == '__main__':
    p = str(Path(__file__).relative_to(Path.cwd())).replace('/', '.').replace('.py', '')
    logger = logging.getLogger(p)
    logger.debug('test ShortTermCaptionGenerator')

    p_video_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/')
    p_nlq_csv = Path('/data/gunsbrother/prjs/ltvu/tasks/Ego4D/EgoNLQ/csvs/nlq_val_v1.csv')
    df_nlq = pd.read_csv(p_nlq_csv, header=0)
    p_output_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu/captions/test/step1/{prompt_type}')

    # caption_sample()
    # caption_sample_with_sliding(2)
    caption_sample_with_sliding(q_uid='701e0e4c-ef5d-514e-8a65-3336c42ac03a')
    # logger.setLevel(logging.INFO)
    # caption_samples()
