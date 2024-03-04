# import sys
# sys.path.append('/data/gunsbrother/prjs/ltvu/llms/Video-LLaVA/ltvu')

import json
from pathlib import Path

from feasibility_check.constants.clip_uids import SAMPLE_CLIP_UIDS
from generate_mid_term_captions import load_model, Captions, Debator, get_video_length
import utils
# class SaliencyFromKeyObjectDetection:
#     def __init__(self):
#         self.caption_generator = VideoLLaVACaptionGenerator(

#         )
