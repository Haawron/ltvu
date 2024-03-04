import sys
sys.path.append('/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu')

import json
from pathlib import Path

from generate_mid_term_captions import load_model, Captions, Debator, get_video_length
from utils.data_types import (
    TimestampType, IntervalType,
    Timestamp, Interval, get_near_gt_interval)


class Windowing:
    def __init__(self,
        p_video: Path,
        itvl: IntervalType,
        model,
        tokenizer,
        video_processor,
    ):
        self.p_video = p_video
        self.itvl = itvl
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        self.debator = Debator(
            p_video=p_video, start_sec=itvl.s.sec, duration_sec=itvl.l.sec,
            model=model, tokenizer=tokenizer, video_processor=video_processor, name='Assistant'
        )

    def run(self,
        captions: Captions,
        query: str,
        itvl_gt: IntervalType
    ) -> list[dict]:
        pass


if __name__ == '__main__':
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')
    p_captions_dir = Path(f'./ltvu/captions/cheat/step2-3/')
    p_captions_dir.mkdir(exist_ok=True, parents=True)
    conv_mode = "llava_v1"

    p_sample_captions_json = Path("/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu/captions/test/per_3.0s/03_20240219v0/gathered.json")
    with p_sample_captions_json.open() as f:
        step1_captions_json = json.load(f)
    clip_uid = '0b20e242-a496-4662-a3e7-645bcecdbe55'  # '0aca0078-b6ab-41fb-9dc5-a70b8ad137b2'
    q_uid = 'cbdc37c7-820a-5bb3-a597-53ca31a13a6f' # '9e5cd376-1b29-5861-8115-be750272d0a9'
    model, tokenizer, video_processor = load_model()

    p_clip = p_clips_dir / f'{clip_uid}.mp4'
    duration_sec = get_video_length(p_clip)
    q_inst = step1_captions_json[clip_uid]['q_instances'][q_uid]
    q_query = q_inst['query']
    q_gt = Interval(float(q_inst['gt_start_sec']), float(q_inst['gt_end_sec']))
    q_captions = q_inst['captions']
    p_caption: Path = p_captions_dir / f'{clip_uid}/{q_uid}.json'
    p_caption.parent.mkdir(parents=True, exist_ok=True)

    itvl_window = get_near_gt_interval(Interval(0., duration_sec), q_gt)

    exp = Windowing(
        p_video=p_clips_dir / f'{clip_uid}.mp4',
        itvl=itvl_window,
        model=model,
        tokenizer=tokenizer,
        video_processor=video_processor
    )
    exp.run(
        captions=Captions(p_caption),
        query=q_query,
        itvl_gt=q_gt
    )
