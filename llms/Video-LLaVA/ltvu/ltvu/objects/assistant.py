import torch

from ltvu import VideoLLaVA
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


def converse(prompt, conv, model: VideoLLaVA, tokenizer, tensor, begin_with=None):
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


class Assistant:
    def __init__(
        self,
        model,
        tokenizer,
        conv_mode = "llava_v1",
    ):
        """\
        # An AI assistant for video conversations.

        # Differences
        - vs. Debator in ltvu/ltvu/utils/data_types/debator.py:
            - Assistant has less properties.
        - vs. Conversation in videollava/conversation.py:
            - Assistant has a model and a tokenizer loaded.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode

        self.conv = self.reset_conv()

    def converse(self,
        prompt: str,
        tensor: torch.Tensor,
        begin_with: str|None = None,
        append_outputs: bool = True
    ) -> str:
        outputs = converse(
            prompt=prompt,
            tensor=tensor,
            conv=self.conv,
            model=self.model,
            tokenizer=self.tokenizer,
            begin_with=begin_with
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

    def reset_conv(self):
        init_conv = conv_templates[self.conv_mode].copy()
        self.conv = init_conv
        return init_conv
