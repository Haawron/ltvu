import torch
import logging
from pathlib import Path

from videollava.utils import disable_torch_init
from videollava.mm_utils import get_model_name_from_path
from videollava.model.builder import load_pretrained_model


def load_model(return_wrapper=False):
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = '/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']

    def wrapper(p_video: str|Path, device='cuda', return_tensors='pt'):
        video_tensor = video_processor(
            str(p_video), return_tensors=return_tensors)['pixel_values']
        tensor: torch.Tensor = video_tensor.to(device, dtype=torch.float16)
        return tensor

    return model, tokenizer, wrapper if return_wrapper else video_processor


if __name__ == '__main__':
    logger = logging.getLogger('LTVU')

    logger.debug('test load_model')
    model, tokenizer, video_processor = load_model()
    print(model)
    print(tokenizer)
    print(video_processor)
    print('test passed')
