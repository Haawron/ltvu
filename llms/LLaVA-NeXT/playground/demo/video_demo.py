from pathlib import Path
import argparse
import torch

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

import transformers
from transformers import AutoConfig

import time

import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # parser.add_argument("--video_path", help="Path to the video files.", required=True)
    # parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    # parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--clip-chunk-sec", type=float, default=2.)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    # fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def load_video_by_stride(video_path, args):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    clip_chunk_frame = fps * args.clip_chunk_sec
    total_frame_num = len(vr)
    total_chunks = int(total_frame_num / clip_chunk_frame)
    chunks = []
    for i in range(total_chunks):
        start_frame = i * clip_chunk_frame
        end_frame = (i + 1) * clip_chunk_frame
        frame_idxs = np.linspace(start_frame, end_frame, args.for_get_frames_num, dtype=int)
        frame_idxs = np.clip(frame_idxs, 0, total_frame_num - 1)
        chunk = vr.get_batch(frame_idxs.tolist()).asnumpy()
        chunks.append(chunk)
    return np.stack(chunks), fps  # (total_chunks, frames, height, width, channels)


def load_model(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["patchify_video_feature"] = False

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)


        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
        else:
            least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

        scaling_factor = math.ceil(least_token_number/4096)
        # import pdb;pdb.set_trace()

        if scaling_factor >= 2:
            if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                print(float(scaling_factor))
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    return tokenizer, model, image_processor, context_len


def build_output_json(question, prompt, frame_idxs, answers):
    assert len(frame_idxs) == len(answers)
    scopes = ['global'] * len(frame_idxs)  # for compatibility with legacy
    output_json = {
        'prompts': {
            'global': {
                'raw_query': question,
                'prompt': prompt,
            }
        },
        'answers': list(zip(frame_idxs, scopes, answers))
    }
    return output_json


def run_inference(args):
    # p_sample_clip = '/data/datasets/ego4d_data/v2/clips_320p-non_official/00030ae8-e5c4-41ec-a8ea-63548a7665d6.mp4'

    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')
    p_clips = sorted(p_clips_dir.glob('*.mp4'))
    if args.world_size > 1:
        p_clips = p_clips[args.rank::args.world_size]
        print(f"[Rank {args.rank} out of {args.world_size:2d}] Total clips: {len(p_clips)}")
    else:
        print(f"Total clips: {len(p_clips)}")

    p_outdir = Path('work_dirs/LLaVA-NeXT-Video-7B-DPO/global')
    p_outdir.mkdir(parents=True, exist_ok=True)
    transformers.logging.set_verbosity_error()  # ignore huggingface transformers' warnings

    question = "What can you see?"
    tokenizer, model, image_processor, context_len = load_model(args)

    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    @torch.inference_mode()
    def run_for_single_clip(p_clip):
        frame_idxs = []
        answers = []
        videos, fps = load_video_by_stride(p_clip, args)
        for i, video in enumerate(videos):
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            model.update_prompt([[question]])
            output_ids = model.generate(
                inputs=input_ids, images=[video], attention_mask=attention_masks, modalities="video",
                do_sample=True, temperature=0.2, max_new_tokens=512,
                use_cache=False,
                stopping_criteria=[stopping_criteria])

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            start_frame, end_frame = int(i * fps * args.clip_chunk_sec), int((i + 1) * fps * args.clip_chunk_sec)
            frame_idxs.append((start_frame, end_frame))
            answers.append(outputs)

        output_json = build_output_json(question, prompt, frame_idxs, answers)
        p_out = p_outdir / f"{p_clip.stem}.json"
        json.dump(output_json, p_out.open('w'))

    # run_for_single_clip(p_sample_clip)
    for p_clip in tqdm(p_clips):
        try:
            run_for_single_clip(p_clip)
        except Exception as e:
            print(f"Error processing {p_clip}: {e}")
            Path(f'work_dirs/errors/{args.rank:02d}-{args.world_size:2d}.txt').write_text(
                f"{p_clip}\n{e}\n")
            continue


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
