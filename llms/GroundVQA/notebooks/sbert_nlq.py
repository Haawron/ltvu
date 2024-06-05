import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from sentence_transformers import SentenceTransformer


def get_captions(captioner_name, clip_uid):
    p_cap_data = p_caps_dir / f'{clip_uid}.json'
    if not p_cap_data.exists():
        tqdm.write(f'{p_cap_data} not found')
        return np.arange(10), ['hi'] * 10

    if captioner_name == 'VideoRecap':
        cap_data = json.load(p_cap_data.open())
        caps = cap_data['captions']['text']
        intervals = np.stack([cap_data['captions']['start'], cap_data['captions']['end']], axis=-1)
        frame_idxs = (intervals.mean(axis=-1) * FPS).astype(int)

    elif 'llava' in captioner_name.lower():
        frame_idxs, caps = [], []
        for c in json.load(p_cap_data.open())['answers']:
            if isinstance(c[0], int):  # timestamp in frame index
                frame_idxs.append(c[0])
            else:  # interval in frame indices
                frame_idxs.append((c[0][0] + c[0][1]) / 2)
            caps.append(c[-1])
        frame_idxs = np.array(frame_idxs)

    else:
        raise ValueError(f'captioner_name={captioner_name}')

    return frame_idxs, caps


def main():
    model = SentenceTransformer(model_name)

    nlq_data = json.load(p_nlq_val_json.open())
    clip_uids = sorted(set([q['video_id'] for q in nlq_data]))

    correct_records = []
    pbar = tqdm(clip_uids, dynamic_ncols=True)
    for clip_uid in pbar:
        pbar.set_description(clip_uid)

        # questions and GTs
        queries, gts = [], []
        clip_samples = list(filter(lambda x: x['video_id'] == clip_uid, nlq_data))
        for q in clip_samples:
            queries.append(q['question'])
            gts.append((q['clip_start_sec'], q['clip_end_sec']))
        gts = np.array(gts)

        # captions
        frame_idxs, caps = get_captions(captioner_name, clip_uid)

        # encode and similarity
        embs_qs = model.encode(queries)
        embs_caps = model.encode(caps)
        sim = model.similarity(embs_qs, embs_caps).cpu().numpy()

        # evaluate
        correct_record = evaluate(sim, frame_idxs, gts)
        correct_records.append(correct_record)

    # aggregate and print
    output_dict = {
        'captioner_name': captioner_name,
        'model_name': model_name,
    }
    for metric_name in correct_records[0].keys():
        values = [record[metric_name] for record in correct_records]
        values = np.concatenate(values)
        output_dict[metric_name] = f'{100*np.mean(values):5.2f}'
    print(pd.Series(output_dict))


def evaluate(sim, frame_idxs, gts):
    # preds from similarity
    pred_cap_idxs = sim.argsort(axis=1)[:, :5]
    pred_frame_idxs = frame_idxs[pred_cap_idxs]
    preds = np.stack([pred_frame_idxs / FPS - pred_width_sec / 2, pred_frame_idxs / FPS + pred_width_sec / 2], axis=-1)

    # compute iou
    gts_ = gts[:, None, :]
    inters = np.maximum(0, np.minimum(preds[..., 1], gts_[..., 1]) - np.maximum(preds[..., 0], gts_[..., 0]))
    unions = (preds[..., 1] - preds[..., 0]) + (gts_[..., 1] - gts_[..., 0]) - inters
    ious = inters / unions

    # compute recalls thresholded by iou
    record = {}
    for k in ks:
        for th in iou_threses:
            mask = ious >= th
            corrects = mask[:, :k].any(axis=1)   # [N,]
            metric_name = f'R{k}@{th:.1f}'  # e.g., R1@0.3
            record[metric_name] = corrects.tolist()

    # compute AUC
    N_q, N_cap = sim.shape
    gt_mask = np.zeros((N_q, N_cap), dtype=bool)
    for i, (s, e) in enumerate(gts):
        s_idx, e_idx = int(s * FPS), int(e * FPS)
        s_ord = np.argmin(np.abs(frame_idxs - s_idx))
        e_ord = np.argmin(np.abs(frame_idxs - e_idx))
        gt_mask[i, s_ord:e_ord+1] = True
    record['AUC'] = [roc_auc_score(gt_mask.ravel(), sim.ravel())] * N_q

    return record


if __name__ == '__main__':
    FPS = 30
    ks, iou_threses = [1, 5], [0.3, 0.5]
    pred_width_sec = 30.

    captioner_name = 'VideoRecap'
    # captioner_name = 'llava-v1.6-34b'
    # captioner_name = 'LLaVA-NeXT-Video-7B-DPO'

    # model_name = 'all-mpnet-base-v2'
    model_name = 'multi-qa-mpnet-base-dot-v1'

    p_caps_dir_collection = {
        'VideoRecap': Path('/data/gunsbrother/prjs/ltvu/ours/data/Ego4D-processed/captions/VideoRecap/caption_2s/val/'),
        'llava-v1.6-34b': Path('/data/gunsbrother/prjs/ltvu/llms/LLaVA/results/egonlq/llava-v1.6-34b/global'),
        'LLaVA-NeXT-Video-7B-DPO': Path('/data/gunsbrother/prjs/ltvu/llms/LLaVA-NeXT/work_dirs/LLaVA-NeXT-Video-7B-DPO/global'),
    }
    p_caps_dir = p_caps_dir_collection[captioner_name]
    p_nlq_val_json = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/unified/annotations.NLQ_val.json')

    main()


"""
                            R1@0.3  R1@0.5  R5@0.3  R5@0.5  AUROC
all-mpnet-base-v2             1.41    0.48    5.71    1.96  56.78
multi-qa-mpnet-base-dot-v1    0.97    0.31    5.58    1.91  55.78
"""

"""
captioner_name                    VideoRecap
model_name        multi-qa-mpnet-base-dot-v1
R1@0.3                                  1.08
R1@0.5                                  0.33
R5@0.3                                  4.22
R5@0.5                                  1.30
AUC                                    59.96

VideoRecap
all-mpnet-base-v2
R1@0.3:  1.03
R1@0.5:  0.24
R5@0.3:  4.20
R5@0.5:  1.54
AUC   : 60.93

llava-v1.6-34b
all-mpnet-base-v2
R1@0.3:  1.41
R1@0.5:  0.48
R5@0.3:  4.79
R5@0.5:  1.67
AUC   : 56.78

LLaVA-NeXT-Video-7B-DPO
all-mpnet-base-v2
R1@0.3:  0.92
R1@0.5:  0.35
R5@0.3:  3.38
R5@0.5:  1.23
AUC   : 63.01
"""
