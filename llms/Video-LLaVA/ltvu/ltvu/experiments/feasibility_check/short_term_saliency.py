import os
import json
import logging
from pathlib import Path

import pyslurm
import numpy as np
import pandas as pd

from ltvu.utils import *
from ltvu.objects import *
from ltvu.constants.prompts import SHORT_TERM_PROMPTS


class ShortTermCaptionGenerator:
    def __init__(
        self,
    ):
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
        for ntry in range(3):
            try:
                p_trimmed = trim_video(
                    p_video=p_video, s=s, e=e,
                    pass_if_exists=(ntry == 0))
                tensor = self.video_processor(p_trimmed)
            except RuntimeError as err:
                logger.error(f'Got an error while running try {ntry+1}', err)
            else:
                break
        else:
            raise err
        captions: list[str] = []
        for prompt in prompts:
            prompt = prompt.format(query=query)
            logger.info(f'prompt: {prompt}')
            output = self.assistant.converse(
                prompt=prompt,
                tensor=tensor,
                append_outputs=True,
                begin_with=None)
            logger.info(f'output: {output}')
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
        nsteps = len(timestamps) - 1
        q_captions = []
        for i, (s_sec, e_sec) in enumerate(zip(timestamps[:-1], timestamps[1:])):
            logger.info(f'[{i:3d}/{nsteps:3d}] {s_sec:.1f} ~ {e_sec:.1f}')
            q_caption = self.caption_single(p_video=p_video, s=s_sec, e=e_sec, query=query)
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


def caption_sample_with_sliding(
    idx_clip: int = 0,
    idx_query: int = 0,
    stride_sec: float = 3.,
    q_uid: None|str = None,
    gen: None|ShortTermCaptionGenerator = None,
):
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
    if gen is None:
        gen = ShortTermCaptionGenerator()
        gen.initialize()
    output = {
        'gt': itvl_gt,
        'query': query,
        'prompts': prompts}
    q_captions: list[dict] = gen.caption_with_sliding(
        p_video=p_clip,
        stride_sec=stride_sec,
        query=query)
    output['captions'] = q_captions
    gen.dump_caption(output, p_output_json)
    logger.info(f'Captions for {str(p_clip)} saved in {str(p_output_json)}')


def get_allocated_nodes(job_id_query: str|int) -> str|None:
    """Returns the allocated nodes for a given SLURM job ID using pyslurm."""
    psjobs = pyslurm.job().get()
    if isinstance(job_id_query, str) and '_' in job_id_query:
        mother_job_id, task_id = list(map(int, job_id_query.split('_')))
        for _job_id, job_info in psjobs.items():
            if job_info.get('array_job_id') == mother_job_id \
                and job_info.get('array_task_id') == task_id:
                job_id_query = int(_job_id)
                break
    else:
        job_id_query = int(job_id_query)
    return psjobs.get(job_id_query, {}).get('nodes')


def caption_samples(**kwargs):
    from ltvu.experiments.feasibility_check.constants import SAMPLE_CLIP_UIDS
    from ltvu.slurm.submitter import Submitter
    from tqdm import tqdm
    import time

    logging.getLogger('ltvu').setLevel(logging.INFO)
    logger.debug(f'Loglevel test')
    p_slurm_logdir = p_prj.joinpath('slurm/logs')
    p_slurm_logdir.mkdir(parents=True, exist_ok=True)
    logger.info(f'SLURM logdir: {p_slurm_logdir}')

    jobs = []
    for i, clip_uid in enumerate(SAMPLE_CLIP_UIDS):
        nqueries = df_nlq[df_nlq['clip_uid'] == clip_uid].shape[0]
        for j in range(nqueries):
            jobs.append({'idx_clip': i, 'idx_query': j})
    njobs = len(jobs)
    world_size = ngpus = min(16, njobs)
    ranks = list(range(world_size))
    job_chunks = [jobs[rank::world_size] for rank in ranks]

    # use pid as multiple extraction workloads with different params can be running simultaneously
    p_tmp_done_check_dir = p_prj / f'tmp/video-llava/pid/{os.getpid():08d}/check-done'
    p_tmp_done_check_dir.mkdir(parents=True, exist_ok=False)

    def _worker(rank: int):  # this closure(?) will be pickled by cloudpickle, a dependency of submitit
        p_done_rank_dir = p_tmp_done_check_dir / f'{rank:d}'
        p_done_rank_dir.mkdir(parents=True, exist_ok=True)
        _jobs: list[dict] = job_chunks[rank]
        gen = ShortTermCaptionGenerator()
        gen.initialize()
        for job in _jobs:
            try:
                caption_sample_with_sliding(**job, gen=gen, **kwargs)
            except Exception as err:
                logger.error(err, exc_info=1, stack_info=True)
            p_done = p_done_rank_dir / f'{job["idx_clip"]:03d}-{job["idx_query"]:03d}.done'
            p_done.touch()

    submitter = Submitter(slurm_array_parallelism=ngpus)
    sjobs = submitter.submit_jobs(_worker, ranks)
    for sjob in sjobs:
        sjob.cancel_at_deletion()

    pbars = [
        tqdm(
            total=len(job_chunks[rank]),
            desc=f'{sjobs[rank].job_id:20s}',
            position=rank+1, leave=False)
        for rank in ranks]
    pbars.append(tqdm(total=njobs, desc=f'{" Total ":#^20s}', position=0, leave=False))

    done_sjobs = ['not done' for _ in sjobs]
    allocated_nodes = [None for _ in sjobs]
    while True:
        n_samples_done = 0
        for sjob in sjobs:
            rank = int(sjob.job_id.split('_')[-1])  # array idx
            # check if any job done with an error
            if done_sjobs[rank] == 'not done' and sjob.done():  # why not .result? ==> it waits synchronously
                try:
                    _ = sjob.result()
                except Exception as err:
                    msg = f'Error: {sjob.job_id}, {sjob.paths.stderr}\n{err}'
                    done_sjobs[rank] = 'done with error'
                else:
                    msg = f'Done: {sjob.job_id}'
                    done_sjobs[rank] = 'done'
                finally:
                    tqdm.write(msg)
            # if the job is firstly allocated
            if allocated_nodes[rank] is None and (node := get_allocated_nodes(sjob.job_id)) is not None:
                pbars[rank].set_description(f'{sjob.job_id:8s} [{node:9s}]')
                allocated_nodes[rank] = node
            p_done_check_rank_dir = p_tmp_done_check_dir / f'{rank:d}'
            n_samples_done_rank = len(list(p_done_check_rank_dir.glob('*.done')))
            n_samples_done += n_samples_done_rank
            delta = n_samples_done_rank - pbars[rank].n
            pbars[rank].update(delta)
            pbars[rank].refresh()
        pbars[-1].update(n_samples_done - pbars[-1].n)
        pbars[-1].refresh()
        if all([done_sjob in ['done', 'done with error'] for done_sjob in done_sjobs]):
            break
        time.sleep(1.)
    tqdm._instances.clear()

    logger.info('')
    logger.info(f'All jobs done [prompt_type: {prompt_type}, stride_sec: {stride_sec:.1f}s]')


if __name__ == '__main__':
    import argparse
    from functools import partial

    p = str(Path(__file__).relative_to(Path.cwd())).replace('/', '.').replace('.py', '')
    logger = logging.getLogger(p)
    logger.info('Run experiment ShortTermCaptionGenerator')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=-1)
    parser.add_argument('--stride-sec', type=float, default=3.)
    parser.add_argument('--prompt-type-id', type=int, default=-1)
    args = parser.parse_args()

    prompt_selected = SHORT_TERM_PROMPTS[args.prompt_type_id]
    prompt_type, prompts = prompt_selected['type'], prompt_selected['prompts']
    p_prj = Path('/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu')
    stride_sec = args.stride_sec
    p_output_dir = p_prj / f'captions/test/step1/{prompt_type}/{stride_sec:.1f}s'

    p_video_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/')
    p_nlq_csv = Path('/data/gunsbrother/prjs/ltvu/tasks/Ego4D/EgoNLQ/csvs/nlq_val_v1.csv')
    df_nlq = pd.read_csv(p_nlq_csv, header=0)

    workload = [
        caption_sample,
        partial(caption_sample_with_sliding, idx_clip=2, idx_query=0),
        partial(caption_sample_with_sliding, q_uid='701e0e4c-ef5d-514e-8a65-3336c42ac03a'),
        partial(caption_sample_with_sliding, q_uid='e97d53eb-4e38-56d1-853d-30c1b0a1747e'),
        partial(caption_samples, stride_sec=stride_sec),
    ][args.mode]

    logger.info(f'prompt_type: {prompt_type}, stride_sec: {stride_sec:.1f}s')
    workload()

else:
    logger = logging.getLogger(__name__)
    prompt_type, prompts = SHORT_TERM_PROMPTS[-1]['type'], SHORT_TERM_PROMPTS[-1]['prompts']
    stride_sec = 3.
    logger.info(f'prompt_type: {prompt_type}, stride_sec: {stride_sec:.1f}s')
