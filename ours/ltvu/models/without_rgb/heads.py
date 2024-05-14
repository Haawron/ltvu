from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer, PreTrainedTokenizerFast,
    DistilBertModel,
)

from sentence_transformers import SentenceTransformer, util as sbert_util
from ltvu.models.heads.nlq_head import (
    NLQHead, MaskedConv1D,
    ctr_diou_loss_1d, sigmoid_focal_loss
)
from ltvu.utils import nms_1d_centers


def build_bert_encoder(num_layers=3) -> DistilBertModel:
    model: DistilBertModel = AutoModel.from_pretrained('distilbert-base-uncased')
    model.embeddings.requires_grad_(False)
    model.transformer.layer = model.transformer.layer[:num_layers]
    return model


def build_valid_masked_conv1d(in_dim, out_dim, kernel_size, bias=True):
    return MaskedConv1D(
        in_dim, out_dim, kernel_size, bias=bias,
        stride=1, padding=kernel_size // 2
    )


class Conv1DBlock(nn.Module):
    def __init__(
            self,
            input_dim,
            feat_dim,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=True,
    ):
        super().__init__()
        self.act = act_layer()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            with_ln_here = with_ln if idx < num_layers - 2 else False
            self.head.append(build_valid_masked_conv1d(
                in_dim, out_dim, kernel_size, bias=(not with_ln_here)))
            self.norm.append(
                nn.LayerNorm(out_dim) if with_ln_here else nn.Identity())

    def forward(self, x):
        cur_out = x  # [B, D, T]
        mask = torch.ones_like(x[:, :1, :], dtype=bool)  # [B, 1, T]
        for idx, (head, norm) in enumerate(zip(self.head, self.norm)):
            cur_out, _ = head(cur_out, mask)
            cur_out = norm(cur_out)
            if idx < len(self.head) - 1:
                cur_out = self.act(cur_out)
        return cur_out


class FlatNLQHead(NLQHead):
    _NLQHeadOutput = namedtuple('_NLQHeadOutput', [
        'segments',  # [k=5, 2], intervals in seconds???  FIXME: seconds???
        'scores',  # [k=5] probs at each point selected as top-k
        'labels',  # [k=5] 0 or 1, 1 if the point is a positive sample
    ])
    def forward(
        self,
        z_ctx,  # [B, D, T]
        m_ctx,  # [B, T]
        gt_segment = None,  # [B, 2]
        return_loss = True,
        return_preds = True
    ):
        assert return_loss or return_preds, \
            'At least one of return_loss or return_preds should be True'
        if return_loss:
            assert gt_segment is not None

        B, D, T = z_ctx.size()

        # fpn_feats: F=1 x [B, D, T_f=T]
        # fpn_masks: F=1 x [B, 1, T_f=T]
        fpn_mask = torch.ones_like(z_ctx, dtype=torch.bool, device=z_ctx.device)[:, :1, :]
        # fpn_mask = m_ctx.to(bool).unsqueeze(1)  # fpn_mask as an attention_mask
        fpn_feats, fpn_masks = self.neck([z_ctx], [fpn_mask])

        points = self.point_generator(fpn_feats)  # F=1 x [ΣT_f=T, 4], t-th row: [t, reg_range_left=0, reg_range_right=10000, fpn_stride=1]
        out_cls_logits = [self.cls_head(fpn_feats, fpn_masks)[0].permute(0, 2, 1)]  # F=1 x [B, T_f=T, K=1]
        out_offsets = [self.reg_head(fpn_feats, fpn_masks)[0].permute(0, 2, 1)]  # F=1 x [B, T_f=T, 2K=2]
        fpn_masks = [fpn_masks[0].squeeze(1)]  # F=1 x [B, T_f=T]

        result_dict = {'logits': out_cls_logits[0].detach().squeeze(2)}  # [B, T]
        if return_loss:
            gt_segment = gt_segment.unsqueeze(1)
            gt_labels = torch.ones((B, 1, 1), dtype=torch.int64, device=z_ctx.device)  # one-hots for k=0 --> all ones
            gt_cls_labels, gt_offsets = self.label_points(
                points,       # [ΣT_f=T, 4]
                gt_segment,   # [B, N_b=1, 2], basically B x N_b x 2
                gt_labels,    # [B, N_b=1, K=1], all ones, basically B x N_b x K
                1             # K=1
            )  # B x [ΣT_f=T, K=1], B x [ΣT_f=T, 2], point-wise labels and offsets
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets)
            result_dict['loss_dict'] = losses
            result_dict['loss'] = losses['final_loss']

        if return_preds:
            results: list[FlatNLQHead._NLQHeadOutput] = self.inference(
                points, fpn_masks, out_cls_logits, out_offsets, 1)
            result_dict['preds'] = results

        return result_dict


class TXActionFormerHead(nn.Module):
    def __init__(
        self,
        in_dim,
        max_ctx_len = 256,
        feature_grid_stride_sec = 2,
        enable_v_emb = False,
    ):
        super().__init__()
        # self.neck = Conv1DBlock(in_dim, in_dim, num_layers=3)
        self.max_ctx_len = max_ctx_len
        self.cross_encoder = build_bert_encoder(num_layers=1)
        self.flat_actionformer_head = FlatNLQHead(
            in_dim, max_ctx_len, feature_grid_stride_sec=feature_grid_stride_sec)
        self.v_emb = nn.Parameter(torch.randn((1, 1, in_dim))) if enable_v_emb else 0

    def forward(
        self,
        z_ctx,  # [B, T, D]
        m_ctx,  # [B, T]
        z_q,  # [B, D]
        gt_segment=None,  # [B, 2]
        **kwargs
    ):
        z_comb = torch.cat(
            [z_q.unsqueeze(1), z_ctx+self.v_emb], dim=1)  # [B, 1+T, D]
        m_comb = torch.cat([m_ctx[:, :1], m_ctx], dim=1)  # [B, 1+T]
        zz_comb = self.cross_encoder.forward(
            inputs_embeds=z_comb,
            attention_mask=m_comb
        ).last_hidden_state  # [B, 1+T, D]
        zz_ctx = zz_comb[:, 1:]  # [B, T, D]
        zz_ctx = rearrange(zz_ctx, 'b t d -> b d t')

        if self.training:
            assert gt_segment is not None
            result_dict = self.flat_actionformer_head.forward(
                zz_ctx, m_ctx, gt_segment=gt_segment)
        else:
            result_dict = self.flat_actionformer_head.forward(
                zz_ctx, m_ctx, return_loss=False)

        return result_dict  # should support d['loss'], d['preds'][bid]['segments']


class ActionFormerHead(nn.Module):
    def __init__(
        self,
        in_dim,
        max_ctx_len = 256,
        feature_grid_stride_sec = 2,
        enable_v_emb = False,
    ):
        super().__init__()
        self.max_ctx_len = max_ctx_len
        self.flat_actionformer_head = FlatNLQHead(
            in_dim, max_ctx_len, feature_grid_stride_sec=feature_grid_stride_sec)
        self.v_emb = nn.Parameter(torch.randn((1, 1, in_dim))) if enable_v_emb else 0

    def forward(
        self,
        z_ctx,  # [B, T, D]
        m_ctx,  # [B, T]
        z_q,  # [B, D]
        gt_segment=None,  # [B, 2]
        **kwargs
    ):
        zz_ctx = rearrange(z_ctx, 'b t d -> b d t')

        if self.training:
            assert gt_segment is not None
            result_dict = self.flat_actionformer_head.forward(
                zz_ctx, m_ctx, gt_segment=gt_segment)
        else:
            result_dict = self.flat_actionformer_head.forward(
                zz_ctx, m_ctx, return_loss=False)

        return result_dict  # should support d['loss'], d['preds'][bid]['segments']


class SimilarityHead(nn.Module):
    @staticmethod
    def test_me():
        from pprint import pprint
        bsz, t, dim = 4, 256, 768
        p = dict(requires_grad=True)
        head = SimilarityHead(dim, max_ctx_len=t)
        z_ctx: torch.Tensor = torch.randn(bsz, t, dim, **p)
        z_q: torch.Tensor = torch.randn(bsz, dim, **p)
        gt_seg = torch.arange(bsz*2, dtype=torch.float).reshape(bsz, 2)
        output = head.forward(z_ctx=z_ctx, m_ctx=None, z_q=z_q, gt_segment=gt_seg)
        pprint(output)
        output['loss'].backward()
        print('grad norms:')
        print('ctx', z_ctx.grad.norm(dim=(1, 2)).tolist())
        print('query', z_q.grad.norm(dim=1).tolist())
        assert all(k in output for k in ['loss', 'preds'])
        assert all(k in output['preds'][0] for k in ['segments', 'scores'])

    def __init__(
        self,
        in_dim,
        max_ctx_len = 256,
        feature_grid_stride_sec = 2,
        pred_width_sec = 30.,
    ):
        super().__init__()
        self.max_ctx_len = max_ctx_len
        self.feature_grid_stride_sec = feature_grid_stride_sec
        self.pred_width_sec = pred_width_sec
        self.max_ctx_duration_sec = max_ctx_len * feature_grid_stride_sec

    def forward(
        self,
        z_ctx,  # [B, T, D]
        m_ctx,  # [B, T]
        z_q,  # [B, D]
        gt_segment=None,  # [B, 2], in effective indices
        **kwargs
    ):
        logits = torch.einsum('btd,bd->bt', z_ctx, z_q)
        result_dict = {}
        if gt_segment is not None:
            gt_mask = torch.zeros_like(logits, dtype=torch.float)
            for bid, seg in enumerate(gt_segment):
                s, e = seg.round().int().tolist()
                gt_mask[bid, s:e] = 1.
            loss = sigmoid_focal_loss(logits, gt_mask, reduction='mean')
            result_dict['loss'] = loss

        result_dict['preds'] = []
        w, s = self.pred_width_sec, self.feature_grid_stride_sec
        T, T_sec = self.max_ctx_len, self.max_ctx_duration_sec
        grid_secs = s * np.arange(0, T, dtype=float)
        for bid, logit in enumerate(logits.float().detach().cpu().numpy()):
            idxs = np.random.randint(T, size=(5,))
            idxs_ = nms_1d_centers(grid_secs, logit, width_sec=w)[1][:5]
            idxs[:len(idxs_)] = idxs_
            centers = grid_secs[idxs]
            scores = np.zeros_like(centers)
            scores[:len(idxs_)] = logit[idxs_]
            segs_sec = np.stack([centers - w / 2, centers + w / 2], axis=1)
            result_dict['preds'].append({
                'segments': segs_sec.clip(0, T_sec).round(4).tolist(),
                'scores': scores.round(4).tolist(),
            })

        return result_dict


if __name__ == '__main__':
    SimilarityHead.test_me()
