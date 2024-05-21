import re
import json
import time
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


def process_raw_query(raw_query: str, mm_use_im_start_end):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in raw_query:
        if mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, raw_query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, raw_query)
    else:
        if mm_use_im_start_end:
            query = image_token_se + "\n" + raw_query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + raw_query
    return query


def reset_conv_and_get_prompt(conv_mode, processed_query):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], processed_query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def get_prompt_input_ids(prompt, tokenizer):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids


def get_image_tensors(model, image_file, image_processor, model_config):
    images = load_images([image_file])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model_config
    ).to(model.device, dtype=torch.float16)
    return images_tensor, image_sizes


@torch.inference_mode()
def do_answer(
    input_ids, images_tensor, image_sizes,
    model: LlavaLlamaForCausalLM, tokenizer,
    temperature = 0,
    top_p = 1.0,
    num_beams = 1,
    max_new_tokens = 256,
):
    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=True if temperature > 0 else False,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
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


def eval_model(
    debug = False,
    load_8bit: bool = False, load_4bit: bool = False,
    rank: int = 0, world_size: int = 1,
):
    print(f'Loading the model in 8bit: {load_8bit}, 4bit: {load_4bit}')
    print(f'Rank: {rank}, World size: {world_size}')

    model_path = "liuhaotian/llava-v1.6-34b"
    model_base = None
    model_name = get_model_name_from_path(model_path)
    raw_queries = {
        # 'local':  'What can you see? Answer only with the names of objects.',
        'global': 'What can you see?',
    }
    conv_mode = "chatml_direct" if 'v1.6-34b' in model_path else "llava_v1"

    # Model
    disable_torch_init()

    for n_tries, cache_dir in enumerate([
        '/local_datasets/shared_cache/hf/hub',
        '/data/shared_cache/hf/hub/'
    ]):
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, model_base, model_name,
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

    p_output_dir = Path(f'results/egonlq/{model_path.split("/")[-1]}')
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

    prompts = {
        query_type: {
            'raw_query': raw_query,
            'prompt': reset_conv_and_get_prompt(
                conv_mode,
                process_raw_query(raw_query, model.config.mm_use_im_start_end),
            )}
        for query_type, raw_query in raw_queries.items()}

    for p_rawframe_clip_dir in p_rawframe_clips:
        if debug and p_rawframe_clip_dir.name != 'f06d1935-550f-4caa-909c-b2db4c28f599':
            continue
        print(f'Processing {p_rawframe_clip_dir}')
        clip_uid = p_rawframe_clip_dir.name
        p_output_clip_json = p_output_dir / f"{clip_uid}.json"
        p_output_clip_pt = p_output_dir / f"{clip_uid}.pt"
        p_tmp_json = p_output_dir / f"-{clip_uid}.json"
        p_tmp_pt = p_output_dir / f"-{clip_uid}.pt"

        answers = {'prompts': prompts, 'answers': []}
        tensors = []

        p_rawframes = sorted(p_rawframe_clip_dir.glob('*.jpg'))
        if debug:
            p_rawframes = p_rawframes[:5]
        p_rawframes = tqdm(p_rawframes, desc=f'Frames for {clip_uid[:8]}')
        for p_img in p_rawframes:
            frame_idx = int(p_img.stem)
            print(frame_idx, end=' ')

            for query_type, prompt_dict in prompts.items():
                input_ids = get_prompt_input_ids(prompt_dict['prompt'], tokenizer)
                images_tensor, image_sizes = get_image_tensors(model, str(p_img), image_processor, model.config)
                outputs, (z0, z) = do_answer(
                    input_ids, images_tensor, image_sizes,
                    model, tokenizer)
                answers['answers'].append((
                    frame_idx,
                    query_type,
                    outputs)
                )
                num_tokens = z0.shape[1], z.shape[1]
                tensors.append((
                    frame_idx,
                    query_type,
                    num_tokens,
                    z0.mean(dim=1, keepdim=True),
                    z.mean(dim=1, keepdim=True))
                )

            with open(p_tmp_json, 'w') as f:
                json.dump(answers, f)
            torch.save(tensors, p_tmp_pt)
            print()

        p_tmp_json.rename(p_output_clip_json)
        p_tmp_pt.rename(p_output_clip_pt)
        print(f'\nDone for {clip_uid}.')


if __name__ == "__main__":
    import fire
    fire.Fire(eval_model)
