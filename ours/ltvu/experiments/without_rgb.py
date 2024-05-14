import logging
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import lightning as L
from lightning import seed_everything

from ltvu.models.base import BasicTransformerEncoder
from ltvu.models.heads.nlq_head import NLQHead
from ltvu.data_loader.egonlq import EgoNLQDataset
from ltvu.models.without_rgb.caption_histogram_nlq import NLQModel5

import os


class CaptionHistogramNLQModel(nn.Module):
    def __init__(self, ctx_len=256, d_model=256, d_hist=16):
        super().__init__()
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.ctx_mode_emb = nn.Parameter(torch.randn((1, 1, d_model)))  # mode embedding
        self.caption_encoder = SentenceTransformer('all-mpnet-base-v2')
        hidden_size = self.caption_encoder.get_sentence_embedding_dimension()
        self.caption_proj = nn.Linear(hidden_size, d_model-d_hist)

        self.query_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.query_proj = nn.Linear(hidden_size, d_model)

        self.vl_encoder = BasicTransformerEncoder(
            d_input=d_model, d_output=d_model, d_model=d_model, dim_feedforward=2*d_model,
            nhead=8, num_layers=1, dropout=.0, droppath=.0,
            activation='gelu', prenorm=True, max_len=ctx_len
        )
        self.norm_vl = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self.nlq_head = NLQHead(d_model, max_v_len=ctx_len)

    @property
    def device(self):
        return self.ctx_mode_emb.device

    @property
    def tokenizer(self):
        return self.caption_encoder.tokenizer

    def forward(self,
        cap_tokens, cap_masks, hists,
        q_tokens, q_masks,
        gt_segments, **kwargs
    ):
        """
        Args:
            cap_tokens: [B, T, L]
            cap_masks: [B, T, L]
            hists: [B, T, d_hist]
            q_tokens: [B, L]
            q_masks: [B, L]
            gt_segments: [B, 1, 2]
        """
        cap_tokens, cap_masks, hists, q_tokens, q_masks, gt_segments = map(
            lambda x: x.to(self.device), [cap_tokens, cap_masks, hists, q_tokens, q_masks, gt_segments])
        B = cap_tokens.shape[0]
        z_cap = self.caption_encoder.forward({
            'input_ids': rearrange(cap_tokens, 'b t l -> (b t) l'),
            'attention_mask': rearrange(cap_masks, 'b t l -> (b t) l')
        })['sentence_embedding']
        z_cap = self.caption_proj(z_cap)  # [B x T, d_model-d_hist]
        z_cap = rearrange(z_cap, '(b t) d -> b t d', b=B)  # [B, T, d_model-d_hist]
        z_ctx = torch.cat([z_cap, hists], dim=-1)  # [B, T, d_model]
        z_ctx += self.ctx_mode_emb  # [B, T, d_model]

        z_q = self.query_encoder.forward({
            'input_ids': q_tokens,
            'attention_mask': q_masks
        })['sentence_embedding']
        z_q = self.query_proj(z_q)  # [B, d_model]

        ####################
        ### debug 0: 원래 코드
        # z_comb = torch.cat([z_ctx, z_q[:, None]], dim=1)  # [B, T+1, d_model]
        # z_comb = self.vl_encoder.forward(z_comb)  # [B, T+1, d_model]
        # z_ctx = z_comb[:, :-1]  # [B, T, d_model]

        # B, T, _ = z_ctx.shape
        # all_true_mask = torch.ones((B, 1, T),
        #     device=z_ctx.device, dtype=bool, requires_grad=False)
        # nlq_results = self.nlq_head.forward(
        #     feat=rearrange(z_ctx, 'b t d -> b d t'),
        #     mask=all_true_mask,
        #     training=self.training,
        #     gt_segments=gt_segments
        # )

        #### debug 1: loss가 0이 안 되는 이유는? => vl encoder를 scratch부터 학습시켜서
        gt_segments = gt_segments.round().to(int).squeeze(1)  # [B, 2]
        z_ctx = F.sigmoid(self.head(z_ctx)).squeeze(-1) # [B, T]
        loss = sum([z_ctx[i, s:e+1].mean() for i, (s, e) in enumerate(gt_segments)]) / B
        nlq_results = {
            'final_loss': loss,
        }

        #### debug 2: nlq_head는 문제 없나? => ㅇㅇ 문제 없는 듯? 쉬운 sample로 골라서 해야 함
        # B, T, _ = z_ctx.shape
        # all_true_mask = torch.ones((B, 1, T),
        #     device=z_ctx.device, dtype=bool, requires_grad=False)
        # nlq_results = self.nlq_head.forward(
        #     feat=rearrange(z_ctx, 'b t d -> b d t'),
        #     mask=all_true_mask,
        #     training=self.training,
        #     gt_segments=gt_segments
        # )

        ### debug 3: VL_encoder는 이런 생판 처음 보는 feature말고 token embedding + projected feature를 넣어줘야 함
        # 그럼 그냥 1-layer transformer로 먼저 대강 해보자
        # z_comb = torch.cat([z_ctx, z_q[:, None]], dim=1)  # [B, T+1, d_model]
        # z_comb = self.vl_encoder.forward(z_comb)  # [B, T+1, d_model]
        # z_ctx = z_comb[:, :-1]  # [B, T, d_model]
        # z_ctx = self.norm_vl(z_ctx)

        # B, T, _ = z_ctx.shape
        # all_true_mask = torch.ones((B, 1, T),
        #     device=z_ctx.device, dtype=bool, requires_grad=False)
        # nlq_results = self.nlq_head.forward(
        #     feat=rearrange(z_ctx, 'b t d -> b d t'),
        #     mask=all_true_mask,
        #     training=self.training,
        #     gt_segments=gt_segments
        # )

        ####################

        if self.training:
            # cls: positive or not, all positive here
            # reg: start, end
            for k, v in nlq_results.items():
                print(f'\t{k}: {v.item():.7f}', end=' ')
            print(flush=True)
            loss = nlq_results['final_loss']
            return loss
        else:
            return nlq_results


def compute_ious(
    pred_s: np.ndarray, pred_e: np.ndarray,
    gt_s: np.ndarray, gt_e: np.ndarray,
):
    # inputs and IoU Matrix: [Q, k=5]
    intersections = np.maximum(np.minimum(pred_e, gt_e) - np.maximum(pred_s, gt_s), 0)
    unions = np.maximum(pred_e, gt_e) - np.minimum(pred_s, gt_s) + 1e-12
    ious = intersections / unions
    return ious


def compute_score_records(
    pred_s: np.ndarray, pred_e: np.ndarray,
    gt_s: np.ndarray, gt_e: np.ndarray,
    ks = [1, 5], iou_thresholds = [0.3, 0.5],
) -> dict[str, np.ndarray[np.bool_]]:
    """
    # Arguments
    `pred_s, pred_e`: `[Q, k]`, should be sorted in advance in descending order.
    `gt_s, gt_e`: `[Q,]` or `[Q, 1]`
    """
    # IoU Matrix: [Q, # preds]
    ious = compute_ious(pred_s, pred_e, gt_s, gt_e)
    # R@1, R@5: [Q,]
    result = {}
    for k, iou_th in itertools.product(ks, iou_thresholds):
        correct = (ious[:, :k] >= iou_th).sum(axis=-1) > 0
        result[f'R@{k} IoU={iou_th}'] = correct
    result['mIoU'] = ious.max(axis=-1)  # mean over this will be an actual mIoU
    return result


class CaptionHistogramNLQModel2(nn.Module):
    def __init__(self, ctx_len=256, d_model=256, d_hist=16):
        super().__init__()
        self.ctx_len = ctx_len
        self.d_model = d_model
        # self.ctx_mode_emb = nn.Parameter(torch.randn((1, 1, d_model)))  # mode embedding
        self.lm = SentenceTransformer('all-mpnet-base-v2')
        hidden_size = self.lm.get_sentence_embedding_dimension()
        self.proj = nn.Linear(hidden_size, d_model)

        self.nlq_head = NLQHead(d_model, max_v_len=ctx_len)

    @property
    def device(self):
        return self.lm.device

    @property
    def tokenizer(self):
        return self.lm.tokenizer

    def forward(self,
        # cap_tokens, cap_masks, hists,
        # q_tokens, q_masks,
        captions, queries, durations,
        gt_segments, **kwargs
    ):
        # captions: B x T, queries: B
        B = len(queries)
        T = len(captions) // B
        txts = []  # B x (T+1)
        for i in range(B):
            txts.extend(captions[i*T:(i+1)*T])
            txts.append(queries[i])
        tokens = self.tokenizer(
            txts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32)  # [B x (T+1), L]
        tokens = {k: v.to(device=self.device) for k, v in tokens.items()}
        z = self.lm.forward(tokens)['sentence_embedding']  # [B x (T+1), d]
        z = rearrange(z, '(b t1) d -> b t1 d', b=B, t1=T+1)
        z = z[:, :-1]  # [B, T, d]
        z = self.proj(z)  # [B, T, d_model]

        gt_segments = gt_segments.type_as(durations)  # [B, 1, 2]
        all_true_mask = torch.ones((B, 1, T), dtype=bool, requires_grad=False, pin_memory=True).to(device=self.device)
        nlq_results = self.nlq_head.forward(
            feat=rearrange(z, 'b t d -> b d t'),
            mask=all_true_mask,
            training=self.training,
            gt_segments=gt_segments,
        )

        if self.training:
            # cls: positive or not, all positive here
            # reg: start, end
            # for k, v in nlq_results.items():
            #     print(f'\t{k}: {v.item():.7f}', end=' ')
            # print(flush=True)
            loss = nlq_results
            return loss
        else:
            preds = np.array([res['segments'].numpy() for res in nlq_results])  # [B, 5, 2]
            scores = np.array([res['scores'].numpy() for res in nlq_results])  # [B, 5]
            gt_segments = gt_segments.squeeze(1).cpu().numpy()  # [B, 2]
            gt_centers = gt_segments.mean(axis=-1)  # [B,]
            w = 30. / durations.cpu().numpy() * T  # 30 seconds
            gt_widened = np.stack([gt_centers - w / 2, gt_centers + w / 2], axis=-1)  # [B, 2]
            gt_widened = np.clip(gt_widened, 1e-12, T-1)

            results = []
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_segments[..., :1], gt_e=gt_segments[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_widened[..., :1], gt_e=gt_widened[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            return results


class CaptionHistogramNLQModel3(nn.Module):
    def __init__(self, ctx_len=256, d_model=256, d_hist=16):
        super().__init__()
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.lm = SentenceTransformer('all-mpnet-base-v2')
        hidden_size = self.lm.get_sentence_embedding_dimension()
        self.proj = nn.Linear(hidden_size, d_model)

        num_views = 16
        self.hist_keys = nn.Parameter(torch.randn(num_views, d_hist))
        self.hist_values = nn.Parameter(torch.randn(num_views, hidden_size))
        self.hist_mode_emb = nn.Parameter(torch.randn((1, 1, hidden_size)))  # mode embedding

        self.nlq_head = NLQHead(d_model, max_v_len=ctx_len)

    @property
    def device(self):
        return self.lm.device

    @property
    def tokenizer(self):
        return self.lm.tokenizer

    def forward(self,
        captions, queries, hists, durations,
        gt_segments, **kwargs
    ):
        # captions: B x T, queries: B
        B = len(queries)
        T = len(captions) // B
        txts = []  # B x (T+1)
        for i in range(B):
            txts.extend(captions[i*T:(i+1)*T])
            txts.append(queries[i])
        tokens = self.tokenizer(
            txts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32)  # [B x (T+1), L]
        tokens = {k: v.to(device=self.device) for k, v in tokens.items()}

        hist_attn = torch.einsum(
            'b t d, v d -> b t v', hists, self.hist_keys)
        hist_attn = F.softmax(10*hist_attn, dim=-1)  # [B, T, V], hard max
        hist_tokens = torch.einsum(
            'b t v, v d -> b t d', hist_attn, self.hist_values)  # [B, T, d]
        hist_tokens += self.hist_mode_emb  # [B, T, d]

        backbone = self.lm[0].auto_model
        input_embeds = backbone.embeddings.word_embeddings(tokens['input_ids'])  # [B x (T+1), L, d]
        input_embeds = rearrange(input_embeds, '(b t) l d -> b t l d', b=B, t=T+1)  # [B, T+1, L, d]
        pad_embed = backbone.embeddings.word_embeddings(torch.zeros((B, 1), dtype=torch.long, device=self.device))  # [B, 1, d]
        hist_padded = torch.cat([hist_tokens, pad_embed], dim=1)  # [B, T+1, d]
        input_embeds = torch.cat([hist_padded[:, :, None], input_embeds], dim=2)  # [B, T+1, 1+L, d]
        input_embeds = rearrange(input_embeds, 'b t1 l1 d -> (b t1) l1 d', b=B)  # [B x (T+1), 1+L, d]

        pos_embeds = backbone.embeddings.position_embeddings(
            backbone.embeddings.create_position_ids_from_inputs_embeds(input_embeds))
        embeds = input_embeds + pos_embeds
        embeds = backbone.embeddings.dropout(backbone.embeddings.LayerNorm(embeds))
        attention_pad = torch.ones(T+1, device=self.device, dtype=bool)
        attention_pad[-1] = False  # query
        attention_pad = repeat(attention_pad, 't -> (b t) 1', b=B)  # [B x (T+1), 1]
        attention_mask = torch.cat([attention_pad, tokens['attention_mask']], dim=1)  # [B x (T+1), 1+L]

        z, = backbone.encoder(
            embeds,
            attention_mask=backbone.get_extended_attention_mask(attention_mask, attention_mask.shape),
            head_mask=backbone.get_head_mask(None, backbone.config.num_hidden_layers)
        )   # [B x (T+1), 1+L, d]
        z = self.lm[1](dict(
            token_embeddings=z,
            attention_mask=attention_mask
        ))['sentence_embedding']  # [B x (T+1), d]
        z = z / z.norm(dim=-1, keepdim=True)
        z = rearrange(z, '(b t1) d -> b t1 d', b=B, t1=T+1)  # [B, T+1, d]
        z = z[:, :-1]  # [B, T, d]
        z = self.proj(z)  # [B, T, d_model]

        gt_segments = gt_segments.type_as(durations)  # [B, 1, 2]
        all_true_mask = torch.ones((B, 1, T), dtype=bool, requires_grad=False, pin_memory=True).to(device=self.device)
        nlq_results = self.nlq_head.forward(
            feat=rearrange(z, 'b t d -> b d t'),
            mask=all_true_mask,
            training=self.training,
            gt_segments=gt_segments,
        )

        if self.training:
            loss = nlq_results
            return loss
        else:
            preds = np.array([res['segments'].numpy() for res in nlq_results])  # [B, 5, 2]
            scores = np.array([res['scores'].numpy() for res in nlq_results])  # [B, 5]
            gt_segments = gt_segments.squeeze(1).cpu().numpy()  # [B, 2]
            gt_centers = gt_segments.mean(axis=-1)  # [B,]
            w = 30. / durations.cpu().numpy() * T  # 30 seconds
            gt_widened = np.stack([gt_centers - w / 2, gt_centers + w / 2], axis=-1)  # [B, 2]
            gt_widened = np.clip(gt_widened, 1e-12, T-1)

            results = []
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_segments[..., :1], gt_e=gt_segments[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_widened[..., :1], gt_e=gt_widened[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            return results


class CaptionHistogramNLQModel4(nn.Module):
    def __init__(self, ctx_len=256, d_model=256, d_hist=16):
        super().__init__()
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.lm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", cache_dir='./checkpoints').encoder
        hidden_size = self.lm.get_sentence_embedding_dimension()
        self.proj = nn.Linear(hidden_size, d_model)

        num_views = 16
        self.hist_keys = nn.Parameter(torch.randn(num_views, d_hist))
        self.hist_values = nn.Parameter(torch.randn(num_views, hidden_size))
        self.hist_mode_emb = nn.Parameter(torch.randn((1, 1, hidden_size)))  # mode embedding

        self.nlq_head = NLQHead(d_model, max_v_len=ctx_len)

    @property
    def device(self):
        return self.lm.device

    @property
    def tokenizer(self):
        return self.lm.tokenizer

    def forward(self,
        captions, queries, hists, durations,
        gt_segments, **kwargs
    ):
        # captions: B x T, queries: B
        B = len(queries)
        T = len(captions) // B
        txts = []  # B x (T+1)
        for i in range(B):
            txts.extend(captions[i*T:(i+1)*T])
            txts.append(queries[i])
        tokens = self.tokenizer(
            txts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32)  # [B x (T+1), L]
        tokens = {k: v.to(device=self.device) for k, v in tokens.items()}

        hist_attn = torch.einsum(
            'b t d, v d -> b t v', hists, self.hist_keys)
        hist_attn = F.softmax(10*hist_attn, dim=-1)  # [B, T, V], hard max
        hist_tokens = torch.einsum(
            'b t v, v d -> b t d', hist_attn, self.hist_values)  # [B, T, d]
        hist_tokens += self.hist_mode_emb  # [B, T, d]

        backbone = self.lm[0].auto_model
        input_embeds = backbone.embeddings.word_embeddings(tokens['input_ids'])  # [B x (T+1), L, d]
        input_embeds = rearrange(input_embeds, '(b t) l d -> b t l d', b=B, t=T+1)  # [B, T+1, L, d]
        pad_embed = backbone.embeddings.word_embeddings(torch.zeros((B, 1), dtype=torch.long, device=self.device))  # [B, 1, d]
        hist_padded = torch.cat([hist_tokens, pad_embed], dim=1)  # [B, T+1, d]
        input_embeds = torch.cat([hist_padded[:, :, None], input_embeds], dim=2)  # [B, T+1, 1+L, d]
        input_embeds = rearrange(input_embeds, 'b t1 l1 d -> (b t1) l1 d', b=B)  # [B x (T+1), 1+L, d]

        pos_embeds = backbone.embeddings.position_embeddings(
            backbone.embeddings.create_position_ids_from_inputs_embeds(input_embeds))
        embeds = input_embeds + pos_embeds
        embeds = backbone.embeddings.dropout(backbone.embeddings.LayerNorm(embeds))
        attention_pad = torch.ones(T+1, device=self.device, dtype=bool)
        attention_pad[-1] = False  # query
        attention_pad = repeat(attention_pad, 't -> (b t) 1', b=B)  # [B x (T+1), 1]
        attention_mask = torch.cat([attention_pad, tokens['attention_mask']], dim=1)  # [B x (T+1), 1+L]

        z, = backbone.encoder(
            embeds,
            attention_mask=backbone.get_extended_attention_mask(attention_mask, attention_mask.shape),
            head_mask=backbone.get_head_mask(None, backbone.config.num_hidden_layers)
        )   # [B x (T+1), 1+L, d]
        z = self.lm[1](dict(
            token_embeddings=z,
            attention_mask=attention_mask
        ))['sentence_embedding']  # [B x (T+1), d]
        z = z / z.norm(dim=-1, keepdim=True)
        z = rearrange(z, '(b t1) d -> b t1 d', b=B, t1=T+1)  # [B, T+1, d]
        z = z[:, :-1]  # [B, T, d]
        z = self.proj(z)  # [B, T, d_model]

        gt_segments = gt_segments.type_as(durations)  # [B, 1, 2]
        all_true_mask = torch.ones((B, 1, T), dtype=bool, requires_grad=False, pin_memory=True).to(device=self.device)
        nlq_results = self.nlq_head.forward(
            feat=rearrange(z, 'b t d -> b d t'),
            mask=all_true_mask,
            training=self.training,
            gt_segments=gt_segments,
        )

        if self.training:
            loss = nlq_results
            return loss
        else:
            preds = np.array([res['segments'].numpy() for res in nlq_results])  # [B, 5, 2]
            scores = np.array([res['scores'].numpy() for res in nlq_results])  # [B, 5]
            gt_segments = gt_segments.squeeze(1).cpu().numpy()  # [B, 2]
            gt_centers = gt_segments.mean(axis=-1)  # [B,]
            w = 30. / durations.cpu().numpy() * T  # 30 seconds
            gt_widened = np.stack([gt_centers - w / 2, gt_centers + w / 2], axis=-1)  # [B, 2]
            gt_widened = np.clip(gt_widened, 1e-12, T-1)

            results = []
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_segments[..., :1], gt_e=gt_segments[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            corrects = compute_score_records(
                pred_s=preds[..., 0], pred_e=preds[..., 1],
                gt_s=gt_widened[..., :1], gt_e=gt_widened[..., 1:],
            )
            recalls = {k: v.mean() for k, v in corrects.items()}
            results.append(recalls)
            return results


def _debug():
    ctx_len = 128
    model = CaptionHistogramNLQModel2(ctx_len=ctx_len).cuda()

    ds = EgoNLQDataset(
        ctx_len=ctx_len,
        tokenizer=model.tokenizer)
    collate_fn = ds.collate_fn
    ds = torch.utils.data.Subset(ds, [123, 119, 125, 126, 46, 5623, 2723, 8283])
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    lr, lrlm = 1e-3, 1e-5
    no_decay_params = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [], 'lr': lr, 'weight_decay': 1e-2},
        {'params': [], 'lr': lr, 'weight_decay': 0.},
        {'params': [], 'lr': lrlm, 'weight_decay': 1e-2},
        {'params': [], 'lr': lrlm, 'weight_decay': 0},
    ]
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay_params):
            if 'lm' in n.split('.'):
                group = 3
            else:
                group = 1
        else:
            if 'lm' in n.split('.'):
                group = 2
            else:
                group = 0
        optimizer_grouped_parameters[group]['params'].append(p)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-12)

    for epoch in range(1000):
        losses_epoch = []
        print(f'[Epoch {epoch+1:03d}]')
        model.train(); model.zero_grad()
        for batch in dl:
            if epoch == 0:
                print(batch)
            loss = model.forward(**batch)['final_loss']
            loss.backward()
            losses_epoch.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        for batch in dl:
            with torch.no_grad():
                results = model.forward(**batch)
                print(results)
                print()
        if (epoch+1) % 10 == 0:
            vram_usage = ' '.join(os.popen('nvidia-smi --query-gpu=memory.used --format=csv').read().strip().split('\n')[1:])
            print(f'loss: {np.mean(losses_epoch):9.7f}, VRAM: {vram_usage}')


def train():
    ctx_len = 128
    model = CaptionHistogramNLQModel2(ctx_len=ctx_len).cuda()

    ds = EgoNLQDataset(
        ctx_len=ctx_len,
        tokenizer=model.tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn)

    lr, lrlm = 1e-3, 1e-5
    no_decay_params = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [], 'lr': lr, 'weight_decay': 1e-2},
        {'params': [], 'lr': lr, 'weight_decay': 0.},
        {'params': [], 'lr': lrlm, 'weight_decay': 1e-2},
        {'params': [], 'lr': lrlm, 'weight_decay': 0},
    ]
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay_params):
            if 'lm' in n.split('.'):
                group = 3
            else:
                group = 1
        else:
            if 'lm' in n.split('.'):
                group = 2
            else:
                group = 0
        optimizer_grouped_parameters[group]['params'].append(p)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-12)

    for epoch in range(100):
        losses_epoch = []
        print(f'[Epoch {epoch+1:03d}]', flush=True)
        model.train(); model.zero_grad()
        pbar = tqdm(dl)
        for idx, batch in enumerate(pbar):
            loss = model.forward(**batch)['final_loss']
            loss.backward()
            losses_epoch.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            optimizer.zero_grad()
            if idx % 10 == 0:
                vram_usage = ' '.join(os.popen('nvidia-smi --query-gpu=memory.used --format=csv').read().strip().split('\n')[1:])
                pbar.set_description(f'loss: {np.mean(losses_epoch):9.7f}, VRAM: {vram_usage}')
            pbar.update(); pbar.refresh()
            if idx == 3: break

        model.eval()
        results = []
        for batch in tqdm(dl, desc='Validation', leave=False):
            with torch.no_grad():
                results.append(model.forward(**batch))
        results = {keys[0]: np.mean(values) for keys, values in zip(*map(dict.items, results))}
        for k, v in results.items():
            print(f'\t{k}: {v:.7f}')


class Lightning(L.LightningModule):
    def __init__(self, ctx_len=128, batch_size=8, model_id=3):
        super().__init__()
        self.model = {
            2: CaptionHistogramNLQModel2,
            3: CaptionHistogramNLQModel3,
            5: NLQModel5,
        }[model_id](ctx_len=ctx_len)
        self.model.lm[0].auto_model.pooler = nn.Identity()
        self.batch_size = batch_size

    def log(self, k, v, **kwargs):
        self.log(k, v,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
            batch_size=self.batch_size, **kwargs)

    def log_dict(self, d, **kwargs):
        self.log_dcit(d,
            logger=True, sync_dist=True, rank_zero_only=True,
            batch_size=self.batch_size, **kwargs)

    def configure_optimizers(self):
        lr, lrlm = 1e-3, 1e-6
        no_decay_params = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [], 'lr': lr, 'weight_decay': 1e-2},
            {'params': [], 'lr': lr, 'weight_decay': 0.},
            {'params': [], 'lr': lrlm, 'weight_decay': 1e-2},
            {'params': [], 'lr': lrlm, 'weight_decay': 0},
        ]
        for n, p in self.model.named_parameters():
            group = 0
            if any(nd in n for nd in no_decay_params):
                group += 1
            if 'lm' in n.split('.'):
                group += 2
            optimizer_grouped_parameters[group]['params'].append(p)
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, eps=1e-12)
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self.model.forward(**train_batch)
        for k in loss:
            self.log(k, loss[k])
        return loss['final_loss']

    def validation_step(self, val_batch, batch_idx):
        nlq_results = self.model.forward(**val_batch)

        preds = np.array([res['segments'].numpy() for res in nlq_results])  # [B, 5, 2]
        scores = np.array([res['scores'].numpy() for res in nlq_results])  # [B, 5]
        gt_segments = gt_segments.squeeze(1).cpu().numpy()  # [B, 2]
        gt_centers = gt_segments.mean(axis=-1)  # [B,]
        w = 30. / durations.cpu().numpy() * T  # 30 seconds
        gt_widened = np.stack([gt_centers - w / 2, gt_centers + w / 2], axis=-1)  # [B, 2]
        gt_widened = np.clip(gt_widened, 1e-12, T-1)

        corrects = compute_score_records(
            pred_s=preds[..., 0], pred_e=preds[..., 1],
            gt_s=gt_segments[..., :1], gt_e=gt_segments[..., 1:],
        )
        self.log_dict({'can_' + k: v.mean() for k, v in corrects.items()})

        corrects = compute_score_records(
            pred_s=preds[..., 0], pred_e=preds[..., 1],
            gt_s=gt_widened[..., :1], gt_e=gt_widened[..., 1:],
        )
        self.log_dict({'wid_' + k: v.mean() for k, v in corrects.items()})


def main():
    torch.set_float32_matmul_precision("high")
    seed_everything(42, workers=True)
    ctx_len, batch_size = 128, 8
    model = Lightning(ctx_len=ctx_len, batch_size=batch_size)
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='gpu', precision="bf16-mixed",
    )
    ds = EgoNLQDataset(ctx_len=ctx_len, split='train', tokenizer=model.model.tokenizer)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn, num_workers=16)
    ds = EgoNLQDataset(ctx_len=ctx_len, split='val', tokenizer=model.model.tokenizer)
    val_loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn, num_workers=16)
    print(len(train_loader), len(val_loader))
    # trainer.ckpt_path = 'lightning_logs/egonlq.ckpt'
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    # train()
    main()
