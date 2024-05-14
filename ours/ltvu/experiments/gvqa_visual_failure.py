import logging
import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os

import torch
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import lightning as L


GET_FRAME_SUCCESS = 0
GET_FRAME_FAIL_NOT_OPENED = 1
GET_FRAME_FAIL_UNKNOWN = 999
p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')


def get_frames(
    p_mp4: str = str(p_clips_dir / '4aac2bee-312f-4609-bfc1-2148d264064f.mp4'),
    times: list[float] = [0]
):
    def get_random_frame():
        return Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))

    cap = cv2.VideoCapture(str(p_mp4))
    if cap.isOpened():  # video header sanity check
        rets, frames = [], []
        duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        for time in times:
            if time > duration_sec:
                frame = get_random_frame()
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, time*1000)
                ret, frame = cap.read()
                if not ret:
                    frame = get_random_frame()
                else:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rets.append(ret)
            frames.append(frame)
        cap.release()
        ret = GET_FRAME_SUCCESS if all(rets) else GET_FRAME_FAIL_NOT_OPENED
    else:
        ret = GET_FRAME_FAIL_NOT_OPENED
        frames = [get_random_frame() for _ in times]
    return ret, frames


def get_color_hist(frame):
    hist = cv2.calcHist([np.array(frame)], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])  # [4, 4, 4]
    hist = cv2.normalize(hist, hist).flatten()  # [4**3]
    return hist


def main():
    for name, logger in logging.root.manager.loggerDict.items():
        if 'transformers' in name and isinstance(logger, logging.Logger):
            logger.setLevel(logging.ERROR)

    model_name = 'clip-ViT-B-32'
    p_json = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/analysis/VLG_OpenQA.json')
    p_out_root_dir = Path('results/gvqa_failure')
    p_out_dir = p_out_root_dir / model_name
    p_out_tmp_dir = p_out_dir / 'tmp'
    p_out_tmp_dir.mkdir(parents=True, exist_ok=True)
    p_report = p_out_dir / 'report.txt'
    p_output = p_out_dir / 'outputs.pt'

    model = SentenceTransformer(model_name, device='cuda')

    data = json.load(p_json.open())
    outputs = []
    pbar = tqdm(data)
    for i, sample in enumerate(pbar):
        clip_uid = sample['clip_uid']
        query_id = sample['query_id']
        query = sample['gt_window']['query']
        pbar.set_description(f'{clip_uid}--{query_id}')
        p_clip = p_clips_dir / f'{clip_uid}.mp4'
        gt_s, gt_e = sample['gt_window']['clip_start_sec'], sample['gt_window']['clip_end_sec']
        gt_t = (gt_s + gt_e) / 2
        ts = [gt_t]
        ts += [(s + e) / 2 for s, e, _ in sample['pred_window']]
        ret, frames = get_frames(p_clip, ts)
        if ret != GET_FRAME_SUCCESS:
            msg = f'{clip_uid}--{query_id}--{ret}'
            p_report.write_text(msg + '\n')
            pbar.write(msg)
        embs = model.encode(frames)
        hists = np.stack([get_color_hist(frame) for frame in frames])  # [1+5, 4**3]
        output = {
            'clip_uid': clip_uid,
            'embs': embs,
            'hists': hists,
            'sims': util.cos_sim(embs[0], embs[1:]).squeeze(0).cpu().numpy(),
            'sims_hist': util.cos_sim(hists[0], hists[1:]).squeeze(0).cpu().numpy(),
            'query': query,
            'gt_t': ts[0],
            'preds_t': ts[1],
            'gt': (gt_s, gt_e),
            'preds': np.array([pred[:-1] for pred in sample['pred_window']]),
            'ious': np.array([pred[-1] for pred in sample['pred_window']]),
            'exit_code': ret,
        }
        outputs.append(output)
        p_out = p_out_tmp_dir / f'{i:04d}--{clip_uid}--{query_id}.pt'
        torch.save(output, p_out)
        if i == pbar.total // 2:
            print('Saving outputs as half way...')
            torch.save(outputs, p_output)
        # if i == 50:
        #     break
    print('Saving outputs...')
    torch.save(outputs, p_output)


if __name__ == '__main__':
    import fire
    fire.Fire()
