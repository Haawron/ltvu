from pathlib import Path
import torch
import multiprocessing as mp
from tqdm import tqdm


def worker(p_pt: str):
    data = torch.load(p_pt, map_location='cpu')
    for idx, v in data:
        v.requires_grad = False
    p_pt = Path(p_pt)
    p_pts_target = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/llava-v1.6-34b-cross-encoded')
    # (p_pts_target / p_pt.parent).mkdir(exist_ok=True, parents=True)
    p_pt_target = p_pts_target / p_pt.parent.name / p_pt.name
    torch.save(data, p_pt_target)


def main():
    p_pts_source = Path('/data/soyeonhong/GroundVQA/cross_encoding/')
    jobs = list(str(p) for p in p_pts_source.glob('**/*.pt'))
    pbar = tqdm(total=len(jobs))
    with mp.Pool(64) as pool:
        for _ in pool.imap_unordered(worker, jobs):
            pbar.update(1)


if __name__ == '__main__':
    main()
