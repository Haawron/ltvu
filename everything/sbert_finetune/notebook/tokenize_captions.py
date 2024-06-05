import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from model.dataset import get_captions


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')


def tokenize(texts_or_tokens, max_length=256):
    return tokenizer(
        texts_or_tokens,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


# captioner_name = 'LLaVA-NeXT-Video-7B-DPO'
# captioner_name = 'llava-v1.6-34b'
captioner_name = 'VideoRecap'
p_caps_root_dir = Path('data/captions') / captioner_name
assert p_caps_root_dir.exists()
p_out_pts_dir = Path('data/tokenized') / captioner_name
p_out_pts_dir.mkdir(parents=True, exist_ok=True)
p_caps = list(p_caps_root_dir.glob('**/*.json'))
pbar = tqdm(p_caps, dynamic_ncols=True)
all_time_longest = 0
for i, p_cap_json in enumerate(pbar):
    clip_uid = p_cap_json.stem
    pbar.set_description(clip_uid)
    pr_cap_json = p_cap_json.relative_to(p_caps_root_dir)
    p_out_pt = p_out_pts_dir / pr_cap_json.with_suffix('.pt')
    p_out_pt.parent.mkdir(parents=True, exist_ok=True)
    frame_idxs, caps = get_captions(captioner_name, p_cap_json)
    tokens = tokenize(caps, None)
    longest_idx = tokens.attention_mask.sum(dim=1).argmax().item()
    longest_cap = caps[longest_idx]
    longest_length = tokens.input_ids.shape[-1]
    all_time_longest = max(all_time_longest, longest_length)
    pbar.set_postfix(current_longest=all_time_longest)
    torch.save((frame_idxs, tokens), p_out_pt)
    if i % 100 == 0:
        pbar.write(f'[{clip_uid}]\nLongest caption over {len(caps)} caps with {longest_length} tokens:\n{longest_cap}\n\n')
    break
pbar.close()

print(f'All Done!')
frame_idxs, tokens = torch.load(p_out_pt)
print(frame_idxs)
print(tokens)
trunc_tokens = tokenizer.truncate_sequences(tokens.input_ids, num_tokens_to_remove=tokens.input_ids.shape[-1] - 128)
print(trunc_tokens)
