import dotenv
dotenv.load_dotenv()

import re
import fire
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
    p_image: str = "/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/sample_images/image copy 2.png",
    load_8bit: bool = False, load_4bit: bool = False,
):
    print(f'Loading the model in 8bit: {load_8bit}, 4bit: {load_4bit}')

    model_path = "liuhaotian/llava-v1.6-34b"
    raw_query = "What can you see?"
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

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name,
        load_8bit=load_8bit, load_4bit=load_4bit, local_files_only=True,
    )

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

    outputs, (z0, z) = do_answer(str(p_image), model, tokenizer, query, args)
    print(outputs)


if __name__ == '__main__':
    fire.Fire(_eval_model)
