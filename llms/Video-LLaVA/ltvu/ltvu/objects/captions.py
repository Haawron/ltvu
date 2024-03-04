import re

from ltvu.constants import SEP, REPLACE_PATTERNS


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
