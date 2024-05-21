import os
import re
from io import StringIO

import torch
import numpy as np
import pandas as pd


def merge_intervals(ivs: list[tuple[int, int]], tol_sec=0., eps_sec=1e-3) -> list[tuple[int, int]]:
    ivs = sorted(ivs)
    merged = [ivs[0]]
    tol = tol_sec + eps_sec
    for (si, ei) in ivs[1:]:
        s, e = merged[-1]
        if si - tol <= e:
            merged[-1] = (s, max(ei, e))  # update e
        else:
            merged.append((si, ei))
    return merged


def nms_1d_centers(
    centers: np.ndarray,  # [T]
    scores: np.ndarray,  # [T]
    width_sec=30.,
    iou_thres=.7
):
    if centers.shape[0] == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep_idxs = []
    while len(idxs) > 0:
        i = idxs[0]
        keep_idxs.append(i)
        ds = np.abs(centers[i] - centers[idxs[1:]])
        ious = np.maximum(1. - 2 * ds / (ds + width_sec), 0.)
        idxs = idxs[1:][ious <= iou_thres]
    keep_idxs = np.array(keep_idxs)
    return centers[keep_idxs], keep_idxs  # [<=T], N.B. not k


def get_gpu_stats() -> dict[str, np.ndarray]:
    # total mem, used mem, util, temp from nvidia-smi
    cmd = 'nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used,temperature.gpu --format=csv'
    info = os.popen(cmd).read()
    df = pd.read_csv(StringIO(info), header=0)
    df_new = {}
    for col in df.columns:
        new_col = col.strip().split(' ')[0]
        if df[col].dtype == 'object':
            unit = re.search(r'\[(.*)\]', col).group(1)
            new_sr = df[col].str.replace(rf'\s*{unit}', '', regex=True).astype(float)
        else:
            new_sr = df[col]
        df_new[new_col] = new_sr.to_numpy()
    # df_new = pd.DataFrame(df_new)
    return df_new


@torch.no_grad()
def evaluate_topk_recalls(
    preds: np.array,  # [B, max k, 2]
    gt_segment: np.array,  # [B, 2]
    ks: list[int] = [1, 5],
    recall_threses: list[float] = [.3, .5],
):
    gt_segment = gt_segment[:, None]  # [B, 1, 2]
    maxs = np.maximum(preds, gt_segment)
    mins = np.minimum(preds, gt_segment)
    inters = np.maximum(0., mins[..., 1] - maxs[..., 0])
    unions = np.maximum(1e-6, maxs[..., 1] - mins[..., 0])
    iogts = inters / (gt_segment[..., 1] - gt_segment[..., 0])
    ious = inters / unions
    results = {}
    for k in ks:
        for thres in recall_threses:
            results[f'iogt>={thres} R@{k:02d}'] = (iogts[:, :k] > thres).any(axis=1).mean()
            results[f'iou>={thres} R@{k:02d}'] = (ious[:, :k] > thres).any(axis=1).mean()
    return results, {
        'iogts': iogts,
        'ious': ious,
        'preds': preds,
        'gt_segment': gt_segment[:, 0],
    }


@torch.no_grad()
def evaluate_topk_span_recalls(
    logits: list[torch.Tensor],
    caps_starts: list[torch.Tensor],
    caps_ends: list[torch.Tensor],
    gt_start_secs: torch.Tensor,
    gt_end_secs: torch.Tensor,
    dfs_caps: list[pd.DataFrame] = None,
    width_sec: float = 30.,
    ks: list[int] = [1, 5],
    recall_threses: list[float] = [.3, .5],
):
    preds = []
    max_k = max(ks)
    for logit, caps_start, caps_end in zip(logits, caps_starts, caps_ends):
        caps_c = (caps_start + caps_end) / 2  # [T_i]
        pred_c, _ = nms_1d_centers(
            caps_c.cpu().numpy(),
            logit.float().cpu().numpy(),
            width_sec=width_sec, iou_thres=.2
        )
        pred_ivs = [(c-width_sec/2, c+width_sec/2) for c in pred_c][:max_k]
        # pred_ivs = merge_intervals(pred_ivs)
        if len(pred_ivs) < max_k:
            pred_ivs += [(0., 0.)] * (max_k - len(pred_ivs))
        preds.append(pred_ivs)  # [max k, 2]
    gts: np.ndarray = torch.stack([gt_start_secs, gt_end_secs], dim=-1).cpu().numpy()[:, None]  # [B, 1, 2]
    preds = np.stack(preds, axis=0)  # [B, max k, 2]
    maxs = np.maximum(preds, gts)  # [B, max k, 2]
    mins = np.minimum(preds, gts)  # [B, max k, 2]
    inters = np.maximum(0., mins[..., 1] - maxs[..., 0])  # [B, max k]
    unions = np.maximum(1e-6, maxs[..., 1] - mins[..., 0])  # [B, max k]
    iogts = inters / (gts[..., 1] - gts[..., 0])  # [B, max k]
    ious = inters / unions  # [B, max k]
    scores_dict = {}
    for k in ks:
        for thres in recall_threses:
            scores_dict[f'iogt>={thres} R@{k:02d}'] = (iogts[:, :k] > thres).any(axis=1).mean().round(4)
            scores_dict[f'iou>={thres} R@{k:02d}'] = (ious[:, :k] > thres).any(axis=1).mean().round(4)
    return scores_dict, {
        'iogts': iogts,
        'ious': ious,
        'preds': preds,
        'gt_segment': gts[:, 0],
    }
