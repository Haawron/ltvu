import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()

import json
import shutil
import textwrap
import subprocess
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def wrapper(
    world_size,
    prompts = ['What would be key objects in this video?'],
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/'),
    p_captions_dir = Path('./ltvu/captions'),
    caption_duration_sec = 3.,
):
    p_nlq_csv = Path('/data/gunsbrother/prjs/ltvu/tasks/Ego4D/EgoNLQ/csvs/nlq_val_v1.csv')
    df_nlq = pd.read_csv(p_nlq_csv, header=0)
    clip_uids = list(df_nlq['clip_uid'].unique())
    p_clips: list[Path] = sorted(p_clip for p_clip in p_clips_dir.glob('*.mp4') if p_clip.stem in clip_uids)
    p_captions_dir.mkdir(exist_ok=True)

    def worker(rank):
        assert 0 <= rank < world_size
        p_clips_rank = p_clips[rank::world_size]
        captioner = VideoLLaVACaptionGenerator(
            video_paths=p_clips_rank,
            caption_duration_sec=caption_duration_sec,
            prompts=prompts,
            p_nlq_csv=p_nlq_csv)
        for video_path in p_clips_rank:
            caption = captioner.generate_captions_for_single_video(video_path)
            clip_uid = video_path.stem
            p_caption = p_captions_dir / f'{clip_uid}.json'
            with p_caption.open('w') as f:
                json.dump(caption, f)

    return worker


class VideoLLaVACaptionGenerator:
    def __init__(
        self,
        video_paths: str|Path|list[str]|list[Path]|None = None,
        caption_duration_sec: int|float = 3.,
        prompts = ['What would be key objects in this video?'],
        p_nlq_csv = Path('/data/gunsbrother/prjs/ltvu/tasks/Ego4D/EgoNLQ/csvs/nlq_val_v1.csv'),
        args = None,
    ):
        # original options
        self.model_path = 'LanguageBind/Video-LLaVA-7B'
        self.model_base = None
        self.load_4bit = True
        self.load_8bit = False
        self.device = 'cuda:0'
        self.conv_mode = 'llava_v1'
        self.cache_dir = '/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/cache_dir'

        # ego4d options
        if video_paths is not None:
            if isinstance(video_paths, list):
                video_paths = [Path(video_path) for video_path in video_paths]
            elif isinstance(video_paths, str):
                video_paths = Path(video_paths)
        self.video_paths: Path|list[Path]|None = video_paths
        self.caption_duration_sec = caption_duration_sec  # the video will be cut every this seconds
        self.p_tmp_dir = Path('/tmp/video_llava/')
        self.prompts = prompts
        self.need_query = any('{query}' in prompt for prompt in self.prompts)
        self.df_nlq = pd.read_csv(p_nlq_csv, header=0)
        self.args = args

        self._load_model()

    def generate_captions_for_videos(self):
        """
        for each video
            for each splitted video
                for each query instance
                    for each prompt
                        caption = model(splitted video, prompt)
        """
        captions = {}
        for video_path in self.video_paths:
            clip_uid = video_path.stem
            captions[clip_uid] = self.generate_captions_for_single_video(video_path)
        return captions

    def generate_captions_for_single_video(self, video_path: Path) -> list[list[str]]:
        clip_uid = video_path.stem
        print(f'======= Generating captions for {clip_uid} ... =======')  # TODO: 남은 시간 표시
        p_splitted_videos = self._split_video(video_path)

        if self.need_query:
            captions_video = {
                'clip_uid': clip_uid,
                'q_instances': {},  # dict of dicts {query: x, gt_start: s, ..., captions: (start sec, end sec, caption1, caption2, ...)}
            }
            df_clip = self.df_nlq[self.df_nlq['clip_uid']==clip_uid]
            annotation_uids = df_clip['annotation_uid'].unique()
            for annotation_uid in annotation_uids:
                df_clip_ann = df_clip[df_clip['annotation_uid']==annotation_uid].reset_index()
                for query_idx, row in df_clip_ann.iterrows():
                    q_instance = {
                        'annotation_uid': annotation_uid,
                        'query_idx': query_idx,
                        'gt_start_sec': row['q_clip_start_sec'],
                        'gt_end_sec': row['q_clip_end_sec'],
                        'query': row['query'],
                        'prompts': [prompt.replace('{query}', row['query']) for prompt in self.prompts],
                        'captions': [],
                    }
                    captions_video['q_instances'][row['q_uid']] = q_instance
            all_quids = list(captions_video['q_instances'].keys())
            print(f'\n\nThis clip instance has {len(all_quids)} queries in total.\n')
        else:
            captions_video = {
                'clip_uid': clip_uid,
                'captions': [],  # list of tuples (start sec, end sec, caption1, caption2, ...)
            }

        N = len(p_splitted_videos)
        pbar = tqdm(total=N, ncols=45 if not self.args.debug else None)
        every_percent = 20
        logging_steps = N // every_percent or 1  # can be 0 when a video too short
        for i, p_splitted_video in enumerate(p_splitted_videos):
            if self.args.debug:
                if i % logging_steps != 0:
                    continue
            else:
                if i > 0 and i % logging_steps == 0:  # every 10%
                    pbar.update(logging_steps)
                    print()
            video_tensor = self.video_processor(str(p_splitted_video), return_tensors='pt')['pixel_values']
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)
            start_sec = i * self.caption_duration_sec
            end_sec = start_sec + self.caption_duration_sec

            def converse(prompts):
                captions_split = [start_sec, end_sec, ]  # start, end, caption, caption, ...
                conv = conv_templates[self.conv_mode].copy()  # conversation
                for i, prompt in enumerate(prompts):
                    if i == 0:
                        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + prompt
                    else:
                        inp = prompt
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids,
                            images=tensor,
                            do_sample=True,
                            temperature=0.1,
                            max_new_tokens=1024,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])
                    caption = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    conv.messages[-1][-1] = caption
                    captions_split.append(caption)
                return captions_split

            if self.need_query:
                textwrapper = textwrap.TextWrapper(
                    width=100,
                    initial_indent='', subsequent_indent=' ' * len('Assistant: '),
                    break_long_words=False, replace_whitespace=False)
                for q_uid in all_quids:
                    q_instance = captions_video['q_instances'][q_uid]
                    prompts = q_instance['prompts']
                    captions_split = converse(prompts)
                    captions_video['q_instances'][q_uid]['captions'].append(captions_split)
                    if self.args.debug:
                        _st_caps_seq = q_instance['captions']
                        _s, _e = q_instance['gt_start_sec'], q_instance['gt_end_sec']
                        for (start_sec, end_sec, *_st_caps) in _st_caps_seq:
                            print(f'\n[{start_sec:5.1f} s ~ {end_sec:5.1f} s, GT: {_s:.1f} ~ {_e:.1f}]')
                            for _st_cap, prompt in zip(_st_caps, prompts):
                                print(textwrapper.fill(f'User     : {prompt}'))
                                print(textwrapper.fill(f'Assistant: {_st_cap}'))
                        else:
                            print('\n-------\n')
            else:
                captions_split = converse(self.prompts)
                captions_video['captions'].append(captions_split)

        print('======= Done captioning =======\n')
        return captions_video

    def _load_model(self):
        print('\nLoading models ...')
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, self.model_name,
            self.load_8bit, self.load_4bit,
            device=self.device, cache_dir=self.cache_dir
        )
        self.video_processor = self.processor['video']
        print('Done loading\n')

    def _split_video(self, video_path: Path) -> list[Path]:
        clip_uid = video_path.stem
        p_tmp_splitted_dir = self.p_tmp_dir / 'splitted_videos' / clip_uid
        if p_tmp_splitted_dir.exists():
            shutil.rmtree(p_tmp_splitted_dir)
        p_tmp_splitted_dir.mkdir(parents=True)
        ext = video_path.suffix  # ex) '.mp4'
        duration = self._get_video_duration(video_path)
        num_subclips = int(duration / self.caption_duration_sec)
        print(f'Splitting {video_path.name} by {self.caption_duration_sec:.1f} secs, into {num_subclips} clips, {duration:.0f} secs in total.')

        start_sec = 0
        p_splitted_videos = []
        while (end_sec := start_sec + self.caption_duration_sec) < duration:
            p_splitted_video = p_tmp_splitted_dir / (f'{start_sec:06.1f}_{end_sec:06.1f}' + ext)
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_sec),
                '-t', str(self.caption_duration_sec),
                '-c', 'copy', '-avoid_negative_ts', '1',
                str(p_splitted_video)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p_splitted_videos.append(p_splitted_video)
            start_sec += self.caption_duration_sec

        print(f'Done splitting, saved in {str(p_tmp_splitted_dir)}\n')
        return p_splitted_videos

    @staticmethod
    def _get_video_duration(video_path: Path) -> float:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        try:
            duration = float(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
        except ValueError:
            raise ValueError(f'Cannot retrieve video duration from {str(video_path)}.')
        return duration


def gather_jsons(p_jsons_dir):
    print(f'\n\nAggregating json files in {str(p_jsons_dir)} ...')
    gathered = {}
    for p_json in sorted(p_jsons_dir.glob('*.json')):
        with p_json.open() as f:
            data = json.load(f)
            gathered.update(data)
    p_json_gathered = p_jsons_dir / 'gathered.json'
    with p_json_gathered.open('w') as f:
        json.dump(gathered, f)
    print(f'Saved the gathered json in {str(p_json_gathered)}')
    return gathered


SAMPLE_CLIP_UIDS = [
    '0aca0078-b6ab-41fb-9dc5-a70b8ad137b2',
    '0b20e242-a496-4662-a3e7-645bcecdbe55',
    '0ca4506c-962d-4cf1-aa6d-f8222f53dee6',
    '00d9a297-d967-4d28-8e5a-6b891814ec65',
    '0ea03f96-b531-444d-9734-ccdc066d0cf2',
    '1c597fc1-7bd0-4325-abbc-645e3ec71866',
    '1d019e7e-e300-4fbc-a7b3-4edf317c1798',
    '2c2bda8d-69a3-4a90-9ad6-f6715bc99f39',
    '3cc0550b-666e-42b7-833a-47f8f9b686ae',
    '4a37144f-63cd-4729-809c-f05ec1839036',
]


if __name__ == '__main__':
    from constants.prompts import SHORT_TERM_PROMPTS
    import argparse, re
    import pandas as pd

    parser = argparse.ArgumentParser(description='Short-term caption generator')
    parser.add_argument('--expname', type=str, default='test')
    parser.add_argument('--prompt-type', type=int, default=-1)
    parser.add_argument('--caption-duration-sec', type=float, default=3.0)
    parser.add_argument('--submitit', action='store_true')
    parser.add_argument('--just-gather-jsons', action='store_true')
    parser.add_argument('--p-captions-dir', type=Path, help='valid only when just_gather_jsons.')
    parser.add_argument('--only-subsets', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    clip_uids = SAMPLE_CLIP_UIDS if args.only_subsets else pd.read_csv(
        '/data/gunsbrother/prjs/ltvu/tasks/Ego4D/EgoNLQ/csvs/nlq_val_v1.csv'
    )['clip_uid'].unique().tolist()
    nclips = len(clip_uids)

    print(Path().absolute())
    print(f'Total clips: {nclips}')
    print('\n\n')
    p_captions_dir = Path(f'./ltvu/captions/{args.expname}/step1/per_{args.caption_duration_sec:.1f}s/')
    p_captions_dir.mkdir(exist_ok=True, parents=True)
    idx_prompt_type = args.prompt_type
    idx_prompt_type = len(SHORT_TERM_PROMPTS) - 1 if idx_prompt_type == -1 else idx_prompt_type
    prompt_type_name = SHORT_TERM_PROMPTS[idx_prompt_type]['type']
    prompts = SHORT_TERM_PROMPTS[idx_prompt_type]['prompts']
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')

    if args.submitit:
        from slurm.submitter import Submitter, get_logger, wait
        Path('./ltvu/slurm/logs').mkdir(parents=True, exist_ok=True)
        ngpus = min(16, nclips)
        logger = get_logger()
        submitter = Submitter(slurm_array_parallelism=ngpus)
        ranks = list(range(ngpus))
        p_caption_tmpdir = p_captions_dir / f'{idx_prompt_type:02d}_{prompt_type_name}'
        p_caption_tmpdir.mkdir(parents=True, exist_ok=True)

        def simple_worker(rank, world_size=ngpus):
            print(Path().absolute())
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            logging.set_verbosity_error()
            p_clips = [
                p_clips_dir / f'{clip_uids[i]}.mp4'
                for i in range(rank, nclips, world_size)]
            print(f'Assigned clips: {len(p_clips)}')
            captioner = VideoLLaVACaptionGenerator(
                video_paths = p_clips,
                caption_duration_sec = args.caption_duration_sec,
                prompts = prompts,
                args = args,
            )
            captions = captioner.generate_captions_for_videos()  # core line
            p_caption = p_caption_tmpdir / f'{rank:02d}_{world_size:02d}.json'
            with p_caption.open('w') as f:
                json.dump(captions, f)
            print(f'Captions are saved in {str(p_caption)}\n')

        jobs = submitter.submit_jobs(simple_worker, ranks)
        if no_error := wait(jobs):
            gather_jsons(p_caption_tmpdir)

    elif args.just_gather_jsons:
        gathered = gather_jsons(args.p_captions_dir)

    else:
        p_caption = p_captions_dir / f'{idx_prompt_type:02d}_{prompt_type_name}.json'
        if not args.debug:
            assert not p_caption.exists()
        else:
            if m := re.findall(r'\(\d+\)', p_caption.stem):
                idx = int(m[0]) + 1
            else:
                idx = 1
            p_caption = p_caption.with_name(p_caption.stem + f'({idx})' + p_caption.suffix)
        captioner = VideoLLaVACaptionGenerator(
            video_paths = [p_clips_dir / f'{clip_uid}.mp4' for clip_uid in clip_uids][:1],
            caption_duration_sec = args.caption_duration_sec,
            prompts=prompts,
            args=args,
        )
        captions = captioner.generate_captions_for_videos()
        with p_caption.open('w') as f:
            json.dump(captions, f)
        print(f'Captions are saved in {str(p_caption)}\n')
