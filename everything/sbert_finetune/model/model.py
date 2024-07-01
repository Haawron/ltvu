
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from sklearn.metrics import roc_auc_score

from transformers import AutoModel

import lightning as L


FPS = 30


def get_model(model_name: str):
    if model_name == 'egovlp':
        state_dict = torch.load('data/checkpoints/egovlp-config-removed.pth', map_location='cpu')
        new_state = {}
        for k, v in state_dict['state_dict'].items():
            if k.startswith('module.text_model.'):
                new_state[k.replace('module.text_model.', '')] = v
        model = AutoModel.from_pretrained('distilbert-base-uncased', local_files_only=False, state_dict=new_state)

    else:
        model = AutoModel.from_pretrained(model_name, local_files_only=False)

    return model


class SentenceGroundingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.model
        self._verify_config(model_config)
        self.model_name = model_config.model_name
        self.max_num_caps = model_config.max_num_caps
        self.freeze_embeddings = model_config.get('freeze_embeddings', False)

        self.model = get_model(model_config.model_name)
        self.model.pooler = None
        # self.pooling_mode = 'mean' if 'all-mpnet-base' in model_config.model_name else 'cls'
        self.pooling_mode = 'mean' if ('all-mpnet-base' in model_config.model_name) or ('all-MiniLM' in model_config.model_name) else 'cls'

        self.alpha_loss_q_cap: float = model_config.alpha_loss_q_cap
        self.alpha_loss_cap_cap: float = model_config.alpha_loss_cap_cap
        self.loss_fn_q_cap: str = model_config.loss_fn_q_cap
        self.loss_fn_cap_cap: str = model_config.loss_fn_cap_cap

        if self.freeze_embeddings:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

    def _verify_config(self, model_config):
        assert model_config.alpha_loss_q_cap > 0 or model_config.alpha_loss_cap_cap > 0, \
            'At least one of alpha_loss_q_cap or alpha_loss_cap_cap should be > 0' \
            f'but got {model_config.alpha_loss_q_cap} and {model_config.alpha_loss_cap_cap}'
        loss_names = ['multi-pos', 'bcewl', 'bce']
        assert model_config.loss_fn_q_cap in loss_names
        assert model_config.loss_fn_cap_cap in loss_names

    def encode(self, tokens):
        z = self.model(**tokens).last_hidden_state
        if self.pooling_mode == 'cls':
            z = z[:, 0]
        elif self.pooling_mode == 'mean':
            z = masked_mean(z, tokens['attention_mask'])
        z = F.normalize(z, p=2, dim=-1)
        return z

    def compute_loss(self, sim, gt_mat: torch.Tensor, loss_fn='multi-pos'):
        # sim: [N_q, N_cap]
        # gt_mat: [N_q, N_cap], binary
        if loss_fn == 'multi-pos':
            loss = multiple_positive_loss(sim, gt_mat)
        elif loss_fn == 'bcewl':
            loss = F.binary_cross_entropy_with_logits(5*sim, gt_mat, reduction='none')
        elif loss_fn == 'bce':
            sim_ = torch.clamp(sim, min=1e-7, max=1-1e-7)
            loss = gt_mat * -torch.log(sim_) + (1. - gt_mat) * -torch.log(1. - sim_)
        else:
            raise NotImplementedError
        return loss

    def forward(
        self,
        # for training
        cap_tokens,  # [N_cap, L_cap]
        q_tokens,  # [N_q, L_q]
        gt_mat = None,  # [N_q, N_cap]
        cap_mat = None,  # [N_cap, N_cap]

        # for evaluation
        segments = None,  # [N_q, 2]
        caption_frame_idxs = None,  # [N_cap]

        **kwargs,
    ):
        z_cap = self.encode(cap_tokens)  # [N_cap, D]
        z_q = self.encode(q_tokens)  # [N_q, D]

        # losses
        loss_total, loss_dict = 0, {}
        if self.alpha_loss_q_cap > 0:
            assert gt_mat is not None
            sim_q_cap = z_q @ z_cap.t()
            loss_q_cap = self.compute_loss(sim_q_cap, gt_mat, self.loss_fn_q_cap)
            alpha = self.alpha_loss_q_cap if self.alpha_loss_cap_cap > 0 else 1
            loss_q_cap = loss_q_cap.mean()
            loss_dict['loss/q_cap'] = loss_q_cap
            loss_q_cap = alpha * loss_q_cap
            loss_total = loss_total + loss_q_cap

        if self.alpha_loss_cap_cap > 0:
            assert cap_mat is not None
            sim_cap_cap = z_cap @ z_cap.t()
            loss_cap_cap = self.compute_loss(sim_cap_cap, cap_mat, self.loss_fn_cap_cap)
            alpha = self.alpha_loss_cap_cap if self.alpha_loss_q_cap > 0 else 1
            loss_cap_cap = loss_cap_cap.mean()
            loss_dict['loss/cap_cap'] = loss_cap_cap
            loss_cap_cap = alpha * loss_cap_cap
            loss_total = loss_total + loss_cap_cap

        loss_dict['loss/total'] = loss_total

        # scores
        score_dict: dict[str, float] = {}
        if not self.training:
            if gt_mat is not None:
                nlq_score = compute_nlq_score_from_simmat(
                    sim_q_cap.detach().float().cpu().numpy(),
                    gt_mat=gt_mat.cpu().numpy(),
                    segments=segments.cpu().numpy(),
                    frame_idxs=caption_frame_idxs.cpu().numpy(),
                )
                for k, v in nlq_score.items():
                    score_dict[f'nlq/{k}'] = v

            if cap_mat is not None:
                pass

        # gather and return
        output_dict = {
            'loss': loss_total,
            'loss_dict': {k: l.detach() for k, l in loss_dict.items()},
            'score_dict': score_dict,
        }
        return output_dict


def masked_mean(token_embeddings, attention_mask):
    # token_embeddings: [B, L, D]
    # attention_mask: [B, L]
    token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # [B, L, D]
    numer = token_embeddings.sum(dim=1)  # [B, D]
    denom = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
    return numer / torch.clamp(denom, min=1e-9)  # [B, D]


def multiple_positive_loss(sim, gt_mat: torch.Tensor):
    MIN_PROB = 1e-7
    probs = F.softmax(sim, dim=-1)  # [N_q, N_cap]
    probs_pos = (probs * gt_mat).sum(dim=-1)  # [N_q]
    return -torch.log(probs_pos.clamp(min=MIN_PROB, max=1))  # [N_q]


def compute_nlq_score_from_simmat(
    sim: np.array,
    frame_idxs: np.array,
    segments: np.array,
    gt_mat: np.array,
    ks: list[int] = [1, 5],
    iou_threses: list[float] = [.3, .5],
    pred_width_sec: float = 30.,
):
    # preds from similarity
    pred_cap_idxs = sim.argsort(axis=1)[:, :5]
    pred_frame_idxs = frame_idxs[pred_cap_idxs]
    preds = np.stack([
        pred_frame_idxs / FPS - pred_width_sec / 2,
        pred_frame_idxs / FPS + pred_width_sec / 2], axis=-1)

    # compute iou
    segs_ = segments[:, None, :]
    inters = np.maximum(0, np.minimum(preds[..., 1], segs_[..., 1]) - np.maximum(preds[..., 0], segs_[..., 0]))
    unions = (preds[..., 1] - preds[..., 0]) + (segs_[..., 1] - segs_[..., 0]) - inters
    ious = inters / unions

    # compute recalls thresholded by iou
    scores = {}
    for k in ks:
        for th in iou_threses:
            mask = ious >= th
            corrects = mask[:, :k].any(axis=1)   # [N,]
            metric_name = f'R{k}@{th:.1f}'  # e.g., R1@0.3
            scores[metric_name] = 100*corrects.mean()

    # compute AUC
    if set(gt_mat.astype(int).ravel()) == {0, 1}:
        scores['AUC'] = 100*roc_auc_score(gt_mat.ravel(), sim.ravel())
    else:
        scores['AUC'] = 100

    return scores
