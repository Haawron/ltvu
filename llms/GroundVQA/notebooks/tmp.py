from pathlib import Path

import torch
from tqdm import tqdm


if __name__ == '__main__':
    p_dir = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/llava-v1.6-34b-uniner-extracted/version-2-2-3-1')
    for p_pt in tqdm(sorted(p_dir.glob('*.pt'))):
        tqdm.write(str(p_pt))
        data = torch.load(p_pt)
        tensors = torch.stack([entry[-1] for entry in data])
