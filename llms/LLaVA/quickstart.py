import dotenv
dotenv.load_dotenv()

import re
import json
import time
from typing import Literal
from pathlib import Path

from tqdm import tqdm
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.eval.run_llava import load_images
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def _eval_model(
    debug=False,
    load_8bit: bool = False, load_4bit: bool = False,
    local_or_global: Literal['local', 'global'] = 'global',
    rank: int = 0, world_size: int = 1,
):
    print(f'Loading the model in 8bit: {load_8bit}, 4bit: {load_4bit}')
    print(f'Rank: {rank}, World size: {world_size}')

    model_path = "liuhaotian/llava-v1.6-34b"
    if local_or_global == 'global':
        raw_query = "What can you see?"
    else:
        raw_query = "What can you see? Answer only with the names of objects."
    sep = ','

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "raw_query": raw_query,
        "conv_mode": "chatml_direct" if 'v1.6-34b' in model_path else "llava_v1",
        "sep": sep,
        "temperature": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "max_new_tokens": 1024
    })()

    # Model
    disable_torch_init()

    for n_tries, cache_dir in enumerate([
        # '/local_datasets/shared_cache/hf/hub',
        '/data/shared_cache/hf/hub/'
    ]):
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, args.model_base, args.model_name,
                load_8bit=load_8bit, load_4bit=load_4bit, local_files_only=True, cache_dir=cache_dir,
            )
        except OSError as e:
            print(f'Error: {e}')
        else:
            break
    else:
        raise e

    print(tokenizer)
    print(model)

    def process_raw_query(raw_query: str, model_config):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in raw_query:
            if model_config.mm_use_im_start_end:
                query = re.sub(IMAGE_PLACEHOLDER, image_token_se, raw_query)
            else:
                query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, raw_query)
        else:
            if model_config.mm_use_im_start_end:
                query = image_token_se + "\n" + raw_query
            else:
                query = DEFAULT_IMAGE_TOKEN + "\n" + raw_query
        return query

    query = process_raw_query(raw_query, model.config)

    def reset_conv_and_get_prompt(conv_mode, query):
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def get_prompt_input_ids(conv_mode, tokenizer, query):
        prompt = reset_conv_and_get_prompt(conv_mode, query)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        return input_ids

    def get_image_tensors(model, image_files, image_processor, model_config):
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model_config
        ).to(model.device, dtype=torch.float16)
        return images_tensor, image_sizes

    @torch.inference_mode()
    def do_answer(p_img, model: LlavaLlamaForCausalLM, tokenizer, query, args):
        images_tensor, image_sizes = get_image_tensors(model, [p_img], image_processor, model.config)
        output_ids = model.generate(
            x:=get_prompt_input_ids(args.conv_mode, tokenizer, query),
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature or None,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            # use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True)
        hidden_states = output_ids.hidden_states
        num_passes = len(hidden_states)  # 1 + # output tokens, 1 for the prompt (init state )
        z_lasts = [hidden_states[i][-1] for i in range(num_passes)]
        z_last_init = z_lasts[0]                       # [B=1, #prompt_tokens=1390,   D=7168]
        z_last_output = torch.cat(z_lasts[1:], dim=1)  # [B=1, #output_tokens=20~200, D=7168]
        return (
            tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip(),
            (z_last_init, z_last_output)
        )

    p_output_dir = Path(f'results/egonlq-sample/{local_or_global}')
    # p_output_dir = Path(f'results/egonlq/{model_path.split("/")[-1]}')
    if not p_output_dir.exists() and rank == 0:
        p_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        time.sleep(1)
    assert p_output_dir.exists()
    print(f'Output dir: {p_output_dir}')

    p_rawframe_root_dir = Path('/data/datasets/ego4d_data/v2/nlq_clip_rawframes')
    p_rawframe_clips = sorted(p_rawframe_root_dir.iterdir())
    if world_size > 1:
        num_clips_prev = len(p_rawframe_clips)
        p_rawframe_clips = p_rawframe_clips[rank::world_size]
        num_clips = len(p_rawframe_clips)
        print(f'[Rank {rank} / {world_size}]: Got [{num_clips} / {num_clips_prev}] clips.')
    p_rawframe_clips = tqdm(p_rawframe_clips)
    for p_rawframe_clip_dir in p_rawframe_clips:
        if debug and p_rawframe_clip_dir.name != 'f06d1935-550f-4caa-909c-b2db4c28f599':
            continue
        print(f'Processing {p_rawframe_clip_dir}')
        clip_uid = p_rawframe_clip_dir.name
        p_output_clip_json = p_output_dir / f"{clip_uid}.json"
        p_output_clip_pt = p_output_dir / f"{clip_uid}.pt"
        answers, tensors = [raw_query], []
        p_rawframes_list = sorted(p_rawframe_clip_dir.glob('*.jpg'))
        if debug:
            p_rawframes_list = tqdm(p_rawframes_list)
        for p_img in p_rawframes_list:
            frame_idx = int(p_img.stem)
            print(frame_idx, end=' ')
            t0 = time.time()
            outputs, (z0, z) = do_answer(str(p_img), model, tokenizer,
            query, args)
            dt = time.time() - t0
            print(dt)
            answers.append((frame_idx, outputs))
            num_tokens = z0.shape[1], z.shape[1]
            tensors.append((
                frame_idx,
                num_tokens,
                z0.mean(dim=1, keepdim=True),
                z.mean(dim=1, keepdim=True),
            ))
            break
        print(answers)
        print()
        # with open(p_output_clip_json, 'w') as f:
        #     json.dump(answers, f)
        # torch.save(tensors, p_output_clip_pt)
        # print(f'Done for {clip_uid}.')


if __name__ == "__main__":
    import fire
    fire.Fire(_eval_model)


class SampleRun:
    model_path = "liuhaotian/llava-v1.6-34b"
    query = "What can you see?"
    p_img = "/data/gunsbrother/prjs/ltvu/tasks/samples/00d9a297-d967-4d28-8e5a-6b891814ec65/000000.jpg"

    # full-A5000x4
    """In the image, I see a kitchen counter with various cooking ingredients and utensils. There are bowls containing what appears to be chopped ingredients, possibly for a stir-fry or similar dish. There are also bottles that might contain sauces or condiments, and a cup that could be for measuring ingredients. On the stove, there's a pan with what looks like a sauce or soup being cooked. The person in the image is holding a pair of chopsticks, suggesting they might be preparing an Asian-style meal. The kitchen has a modern look with a white countertop and a black stove. There's a clock on the wall indicating the time, and the overall setting suggests someone is in the process of cooking a meal."""
    # full-3090x4
    """In the image, I see a kitchen counter with various items on it. There are two bowls, one containing what appears to be raw meat and the other containing a white substance, possibly flour or a similar ingredient. There are also several bottles, which could be cooking oils, sauces, or condiments. A pair of chopsticks is being used to mix the contents of the bowl with the white substance. In the background, there's a stove with a pot on it, suggesting that cooking is in progress. The kitchen looks well-equipped and is in use for meal preparation."""

    # 8bit-A5000x4
    """In the image, I see a kitchen counter with various items on it. There are several bottles, which could be condiments or cooking ingredients, a bowl containing what appears to be chopped ingredients, possibly for a dish, and a pan with a sauce or liquid in it. There's also a pair of chopsticks being used, suggesting that the person might be preparing an Asian-style meal. The kitchen has a modern look with a white countertop and a black stove. In the background, there's a glimpse of a living area with a couch and a television. The image is taken from a first-person perspective, likely from the person cooking."""
    # 8bit-3090x4
    """In the image, I see a kitchen counter with various items on it. There are several bottles, which could be condiments or cooking ingredients, a bowl containing what appears to be chopped ingredients, possibly for a dish, and a pan with a sauce or liquid in it. There's also a pair of chopsticks being used, suggesting that the person might be preparing an Asian-style meal. In the background, there's a kitchen sink and a window with a view of the outside, where it seems to be daytime. The overall scene suggests someone is in the process of cooking or preparing a meal."""

    # 4bit-A5000x4
    """In the image, I see a kitchen counter with various items on it. There are two bowls containing what appears to be food, possibly ingredients for cooking. There are also several bottles, which could be condiments or cooking oils. A pair of hands is visible, holding what looks like a wooden spoon or spatula, suggesting someone is in the process of cooking or preparing a meal. In the background, there's a kitchen sink and a window with curtains, allowing natural light into the room. The overall scene suggests a home cooking environment."""
    # 4bit-3090x4
    """In the image, I see a kitchen counter with various items on it. There are two bowls containing what appears to be food, possibly ingredients for cooking. There are also several bottles, which could be condiments or cooking oils. A pair of hands is visible, holding what looks like a wooden spoon or spatula, suggesting someone is in the process of cooking or preparing a meal. In the background, there's a kitchen sink and a window with curtains, allowing natural light into the room. The overall scene suggests a home cooking environment."""
