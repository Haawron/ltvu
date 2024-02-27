import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()

import sys
import subprocess
from pathlib import Path
import re
import json

import numpy as np
import torch
from torch import Tensor  # for type-hints
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

SEP = '</s>'
REPLACE_PATTERNS = [  # the order matters
    {'replaced_with': '#C',
    'patterns': [
        r'[Tt]he (man|person)(?!(?:\s+in))',  # not followed by in
        r'[Tt]he (man|person) in (this|the) video',
        r'(?!\w)[Hh]e(?=\s)',
    ]},
    {'replaced_with': '',
    'patterns': [
        r'( [Tt]hese objects[^.]* |, which[^.]*| that[^.]*| to )?help(s|ing)?[^.]*answer[^.]*[\'\"].*[\'\"]',
        r'(?<=(, and ))\S+ is also using ',
    ]},
    {'replaced_with': '. And ',
    'patterns': [
        r'. \S+ is also using '
    ]},
    {'replaced_with': ' himself ',
    'patterns': [
        r' him '
    ]},
    {'replaced_with': 'A',
    'patterns': [
        r'[Ii]n this video, we can see a',
    ]},
    # {'replaced_with': 'the key objects would be',
    # 'patterns': [
    #     r'[Tt]he objects are',
    # ]}
]


def trim_video(video_path, clip_start_sec=0, clip_duration_sec=60):
    p_splitted_video_dir = Path('/tmp/video-llava')
    p_splitted_video_dir.mkdir(exist_ok=True, parents=True)
    clip_end_sec = clip_start_sec + clip_duration_sec
    p_splitted_video = p_splitted_video_dir / f'{video_path.stem}_{clip_start_sec:.1f}_{clip_end_sec:.1f}{video_path.suffix}'
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-ss', str(clip_start_sec),
        '-t', str(clip_duration_sec),
        '-c', 'copy', '-avoid_negative_ts', '1', '-y',
        str(p_splitted_video)
    ]
    print('\n' + ' '.join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(str(p_splitted_video))
    return p_splitted_video


def converse(prompt, conv, model, tokenizer, tensor, begin_with=None):
    if conv.messages:
        inp = prompt
    else:
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], begin_with)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    prompt = conv.get_prompt()
    if begin_with is not None:
        assert prompt.endswith(stop_str), prompt
        prompt = prompt[:-len(stop_str)]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs


def get_video_length(p_video):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(p_video)]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout)
    except ValueError:
        raise ValueError("Could not retrieve video duration.")
    return duration


class Captions:
    def __init__(self, caption_sequence):
        self.raw_caption_sequence = caption_sequence  # list of (start_sec, end_sec, caption)

    def captions2prompt(self, start_sec, duration_sec, ret_list = False):
        lines = ['[Captions]']
        caption_end_sec = self.raw_caption_sequence[-1][1]
        assert start_sec < caption_end_sec, \
            f'Out of bounds: start_sec = {start_sec} >= caption_end_sec = {caption_end_sec}'
        for s, e, *_caps in self.raw_caption_sequence:
            if s < start_sec: continue
            if e > start_sec + duration_sec: break
            _cap = _caps[-1]
            _cap = self.polish(_cap)
            _cap = f'{s}s, {_cap}'
            lines.append(_cap)
        if ret_list:
            return lines
        else:
            return '\n'.join(lines)

    @staticmethod
    def polish(caption: str):
        caption = caption.strip().replace(SEP, "")
        for pattern_dict in REPLACE_PATTERNS:
            repl = pattern_dict['replaced_with']
            for pattern in pattern_dict['patterns']:
                caption = re.sub(pattern, repl, caption)
        caption = '. '.join([line.capitalize() for line in caption.split('. ')])
        if not caption.endswith('.'): caption = caption + '.'
        return caption


class Debator:
    def __init__(self,
        p_video,
        start_sec,
        duration_sec,
        model,
        tokenizer,
        video_processor,
        name: str = '',
    ):
        self.p_video = p_video
        self.start_sec = start_sec
        self.duration_sec = duration_sec
        self.end_sec = start_sec + duration_sec
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.name = name
        self.log_prompt = f'{name}: ' if name else ''

        self.conv = conv_templates[conv_mode].copy()
        self.tensor = self.get_trimmed_tensor(
            self.p_video, self.start_sec, self.duration_sec, self.video_processor,
            self.model.device)
        self.log(f'Tensor {self.start_sec:.1f}s ~ {self.end_sec:.1f}s loading done.')

    def converse(self,
        prompt: str,
        begin_with: str|None = None,
        append_outputs: bool = False
    ):
        outputs = converse(
            prompt,
            self.conv,
            self.model,
            self.tokenizer,
            self.tensor,
            begin_with=begin_with,
        )
        if append_outputs:
            self.conv.messages[-1][-1] = outputs
        return outputs

    def dict(self):
        return {
            'p_video': str(self.p_video),
            'start_sec': self.start_sec,
            'duration_sec': self.duration_sec,
            'end_sec': self.end_sec,
            'name': self.name,
            'conv': self.conv.dict(),
        }

    def log(self, msg=''):
        print(f'{self.log_prompt}{msg}')

    def initialize_conv(self):
        init_conv = conv_templates[conv_mode].copy()
        self.conv = init_conv
        return init_conv

    @staticmethod
    def get_trimmed_tensor(
        p_source_video: Path,
        start_sec: float,
        duration_sec: float,
        video_processor,
        device = 'cuda'
    ) -> Tensor:
        p_trimmed = trim_video(p_source_video, start_sec, duration_sec)
        video_tensor: Tensor = video_processor(
            str(p_trimmed)
            , return_tensors='pt')['pixel_values']
        tensor: Tensor = video_tensor.to(device, dtype=torch.float16)
        return tensor


class DebateManager:
    def __init__(self,
        p_video: Path,
        model,
        tokenizer,
        video_processor,
        args=None,
    ):
        self.p_video = p_video
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.args = args

        self.num_debators = 8
        self.video_duration_sec = get_video_length(self.p_video)
        self.debator_duration_sec = self.video_duration_sec / self.num_debators
        self.debator_start_secs = np.linspace(
            0, self.video_duration_sec, self.num_debators,
            endpoint=False)
        self.debators = [
            Debator(
                self.p_video, start_sec, self.debator_duration_sec,
                self.model, self.tokenizer, self.video_processor, name=f'Assistant#{idx_debator}')
            for idx_debator, start_sec in enumerate(self.debator_start_secs)]
        print()

        # cursors
        self.current_round = 0

        # results
        self.results_round1: list[dict] = None

    def first_round(self, captions: Captions, query: str) -> list[dict]:
        """Give opinions simultaneously."""
        if self.current_round != 0:
            print(f'Round {self.current_round} != 0. Resetting convs ...')
            for debator in self.debators:
                debator.initialize_conv()
                debator.log('Conv initialization done.')
            print()

        prompt_templates = [
            '{caption_prompt}\n\n'
            'Hi, {debator_name}! Above is a sequence of short-term captions of this video generated by an video LLM '
            'after watching each short clip to later help another long-term LLM in answering the question "{query}". '
            'Each caption starts with its corresponding timestamp \'${{sec}}s, \'. '
            'What do you think is the most salient moment within this video for answering the question? And explain why.',

            'What temporal window within this video does correspond to that moment? '
            'Tell me the start, and the end of that moment in seconds. '
            'Answer briefly.',

            'The IoU score between your answer and the GT will be...?'
        ]
        self.current_round = 1
        results = []
        for idx_debator, debator in enumerate(self.debators):
            debator.log()
            caption_prompt = captions.captions2prompt(
                debator.start_sec,
                debator.duration_sec)
            for prompt in prompt_templates:
                prompt = prompt.format(
                    debator_name=debator.name,
                    caption_prompt=caption_prompt,
                    query=query)
                outputs = debator.converse(prompt, append_outputs=True)
                print(f'[PROMPT]\n{prompt}')
                print(f'[OUTPUT]\n{outputs}')
                print()
            answers = [answer for role, answer in debator.conv.messages[1::2]]
            debator_record = {
                'debator': debator,
                'explaination': answers[0].replace(SEP, ''),
                'prediction': self.find_floats(answers[1]),
                'confidence': self.find_floats(answers[2]) or 0.,  # 0% when saying "I'm sorry, ..."
            }
            results.append(debator_record)
            if self.args.debug:
                print(debator_record)
                print('\n\n')
        results = sorted(results, key=lambda result: result['debator'].start_sec)
        self.results_round1 = results
        return results

    def debate_type1(self, proceed_round=True, captions=None):
        assert self.current_round > 0
        current_round = self.current_round + 1
        if proceed_round:
            self.current_round = current_round
        self._reset_convs()

        # gather opinions
        opinions = []
        for result in self.results_round1:
            debator = result['debator']
            start_sec, end_sec = result['prediction']
            conf = result['confidence']
            exp = result['explaination']
            opinion: str = (
                f"{debator.name}: "
                f"assigned section={debator.start_sec:.1f}s~{debator.end_sec:.1f}s,"
                f"prediction={start_sec:.1f}s~{end_sec:.1f}s,"
                f"confidence={conf},"
                f"explanation=\"{exp}\"")
            opinions.append(opinion)

        opinion_prompt = '[Opinions]' + debator.conv.sep2.join(opinions)
        if self.args.debug:
            print(opinion_prompt.replace(debator.conv.sep2, '\n'))
        prompt_template = (
            '{opinion_prompt}\n'
            'Above are predictions of all debators who seen different sections of this video. '
            'Do you agree with their opinions? Do you want to modify yours? '
            '{caption_prompt}\n'
            'What do you think is the temporal window of the most salient moment '
            'for answering the question now? '
            'Give me the start and the end seconds of your prediction.'
        )
        begin_with = (
            'After reviewing those opinions of other debators, '
            'I think the start and the end seconds of the most salient moment for answering the question '
            'are'
        )
        debator_records = []
        for idx_debator, debator in enumerate(self.debators):
            debator.log()
            caption_prompt = captions.captions2prompt(
                debator.start_sec,
                debator.duration_sec)
            prompt = prompt_template.format(
                caption_prompt=caption_prompt,
                opinion_prompt=opinion_prompt)
            prompt = prompt.replace(debator.name, f'{debator.name}(YOU)')
            outputs = debator.converse(
                prompt,
                begin_with=begin_with,
                append_outputs=proceed_round)
            debator_record = {
                'debator': debator,
                'answer': outputs}
            debator_records.append(debator_record)
            if self.args.debug:
                print(f'[PROMPT]\n{prompt_template.format(opinion_prompt="...omitted", caption_prompt=caption_prompt)}')
                print(f'[OUTPUT]\n{begin_with} {outputs}')
                print()
        return debator_records

    def _reset_convs(self):
        for debator in self.debators:
            debator.initialize_conv()

    @staticmethod
    def find_floats(s: str) -> float|tuple[float]:
        floats = re.findall(r'\d*\.?\d+', s)
        try:
            floats = list(map(float, floats))
        except ValueError as e:
            print(e)
        if len(floats) == 1:
            return floats[0]
        else:
            return floats

    @staticmethod
    def compute_iou_1d(interval, interval_others):
        interval, interval_others = np.array(interval), np.array(interval_others)
        ref_s, ref_e = interval
        s, e = interval_others[:, 0], interval_others[:, 1]
        inter_s = np.maximum(ref_s, s)
        inter_e = np.minimum(ref_e, e)
        inters = np.maximum(0, inter_e - inter_s)
        l_ref = ref_e - ref_s
        l_others = e - s
        unions = l_ref + l_others - inters
        ious = inters / unions
        return ious


def load_model():
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    return model, tokenizer, video_processor


def worker_single_q_instance(
    model, tokenizer, video_processor,
    p_captions_dir, step1_captions_json, clip_uid, q_uid
):
    p_clip = p_clips_dir / f'{clip_uid}.mp4'
    q_inst = step1_captions_json[clip_uid]['q_instances'][q_uid]
    q_query = q_inst['query']
    q_captions = q_inst['captions']
    p_caption: Path = p_captions_dir / f'{clip_uid}/{q_uid}.json'
    p_caption.parent.mkdir(parents=True, exist_ok=True)

    q_captions = Captions(q_captions)
    manager = DebateManager(p_clip, model, tokenizer, video_processor, args)
    results1 = manager.first_round(q_captions, q_query)
    q_GT = q_inst['gt_start_sec'], q_inst['gt_end_sec']
    intervals = [result['prediction'] for result in manager.results_round1]
    ious = manager.compute_iou_1d(q_GT, intervals)

    print(str(p_caption))
    print(q_GT, intervals)
    print(ious)
    print()

    caption = {k: v for k, v in q_inst.items() if k not in ['prompts', 'captions']}
    debator_records = []
    for idx_debator, debator_record in enumerate(results1):
        debator = debator_record['debator']
        serializable_record = {'name': debator.name}
        serializable_record.update({k: v for k, v in debator_record.items() if k != 'debator'})
        serializable_record['iou'] = ious[idx_debator]
        serializable_record['debator'] = debator.dict()
        debator_records.append(serializable_record)
    caption['step2'] = {'debators': debator_records}

    with p_caption.open('w') as f:  # dump here just in case of an exception below
        json.dump(caption, f)

    results2 = manager.debate_type1(captions=q_captions)
    debator_records = []
    for idx_debator, debator_record in enumerate(results2):
        debator = debator_record['debator']
        serializable_record = {'name': debator.name}
        serializable_record.update({k: v for k, v in debator_record.items() if k != 'debator'})
        # serializable_record['iou'] = ious[idx_debator]
        # omit duplicated opinion prompts for tidyness
        prompt = debator.conv.messages[0][-1]
        s_ind, e_ind = prompt.find('[Opinions]'), prompt.find('[Captions]')
        opinion_prompt_debator = prompt[s_ind:e_ind]
        debator.conv.messages[0][-1] = prompt[:s_ind] + '[Opinions]...omitted\n' + prompt[e_ind:]
        serializable_record['debator'] = debator.dict()

        debator_records.append(serializable_record)
    caption['step3'] = {
        'opinion_prompt': opinion_prompt_debator.replace('(YOU)', ''),
        'debators': debator_records,
    }

    with p_caption.open('w') as f:
        json.dump(caption, f)


if __name__ == '__main__':
    import argparse, re
    parser = argparse.ArgumentParser(description='Mid-term Solvers')
    parser.add_argument('--expname', type=str, default='test')
    parser.add_argument('--single-sample', action='store_true')
    parser.add_argument('--only-subsets', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')
    p_captions_dir = Path(f'./ltvu/captions/{args.expname}/step2-3/')
    p_captions_dir.mkdir(exist_ok=True, parents=True)
    conv_mode = "llava_v1"

    p_sample_captions_json = Path("/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu/captions/test/per_3.0s/03_20240219v0/gathered.json")
    if args.single_sample:
        with p_sample_captions_json.open() as f:
            step1_captions_json = json.load(f)
        clip_uid = '0b20e242-a496-4662-a3e7-645bcecdbe55'  # '0aca0078-b6ab-41fb-9dc5-a70b8ad137b2'
        q_uid = 'cbdc37c7-820a-5bb3-a597-53ca31a13a6f' # '9e5cd376-1b29-5861-8115-be750272d0a9'
        model, tokenizer, video_processor = load_model()
        worker_single_q_instance(
            model, tokenizer, video_processor,
            p_captions_dir, step1_captions_json, clip_uid, q_uid)

    elif args.only_subsets:
        from slurm.scripts.submitter import Submitter, get_logger, wait
        with p_sample_captions_json.open() as f:
            step1_captions_json = json.load(f)
        clip_uids = list(step1_captions_json.keys())
        nclips = len(clip_uids)
        print(Path().absolute())
        print(f'Total clips: {nclips}')
        print('\n\n')
        Path('./ltvu/slurm/logs').mkdir(parents=True, exist_ok=True)
        ngpus = min(16, nclips)
        logger = get_logger()
        submitter = Submitter(
            model_name='Video-LLAVA-captioning-step2-3',
            slurm_array_parallelism=ngpus)
        ranks = list(range(ngpus))

        def simple_worker(rank, world_size=ngpus):
            print(Path().absolute())
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            logging.set_verbosity_error()

            _clip_uids = clip_uids[rank::world_size]
            print(f'Assigned clips: {len(_clip_uids)}\n')
            print('Loading the model ...')
            model, tokenizer, video_processor = load_model()
            for clip_uid in _clip_uids:
                for q_uid in step1_captions_json[clip_uid]['q_instances'].keys():
                    try:
                        worker_single_q_instance(
                            model, tokenizer, video_processor,
                            p_captions_dir, step1_captions_json, clip_uid, q_uid)
                    except Exception as e:
                        print(e, file=sys.stderr)

        jobs = submitter.submit_jobs(simple_worker, ranks)
        if no_error := wait(jobs):
            pass
