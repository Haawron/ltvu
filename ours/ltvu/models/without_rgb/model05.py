import logging
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat
from transformers import T5Tokenizer, T5ForConditionalGeneration

from ltvu.models.heads.nlq_head import NLQHead
from ltvu.models.egovlpv1.model import TextOnlyFrozenInTime
from ltvu.data_loader.egonlq import EgoNLQDataset


class NLQModel5(nn.Module):
    def __init__(self, ctx_len=256, d_model=256):
        super().__init__()
        self.ctx_len = ctx_len
        self.d_model = d_model
        model_name = 'google/flan-t5-base'
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lm = T5ForConditionalGeneration.from_pretrained(model_name).encoder
        # hidden_size = self.lm.get_sentence_embedding_dimension()
        # self.proj = nn.Linear(hidden_size, d_model)

        self.nlq_head = NLQHead(d_model, max_v_len=ctx_len)

    @property
    def device(self):
        return self.lm.device

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
            padding='max_length')  # [B x (T+1), L]
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
        pad_embed = backbone.embeddings.word_embeddings(
            torch.zeros((B, 1), dtype=torch.long, device=self.device))  # [B, 1, d]
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
            return nlq_results


p_ner_jsons_dir = Path(f'data/Ego4D-processed/captions/VideoRecap/caption_2s_uniner_outputs')
PROMPT_ENTITY = '#C C picks '


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


def evaluate_entity_proposals(sample, tokenizer, device, thres_percentile=70., aggregate_method='max', tol_sec=5., eps_sec=1e-3):
    # data
    query = sample['query']
    clip_uid = sample["clip_uid"]
    p_ner_json = p_ner_jsons_dir / f'{clip_uid}.json'
    s, e = sample['gt_start_sec'], sample['gt_end_sec']
    ner_json = json.load(p_ner_json.open())
    ents = [ner_cap['entities']['values'] for ner_cap in ner_json]  # list of list of entities
    starts = [ner_cap['start'] for ner_cap in ner_json]
    ends = [ner_cap['end'] for ner_cap in ner_json]
    duration = max(ends)
    valid_ent_covers = [ee - ss for ent, ss, ee in zip(ents, starts, ends) if ent]

    default_result = {
        'clip_uid': clip_uid, 'duration': duration, 'query': query, 'gt_start': s, 'gt_end': e,
        'ent_coverage': sum(valid_ent_covers) / duration,
        'iou': 0., 'iogt': 0., 'proposed_area': 0., 'percentile_actual': 0.,
        'y_': np.zeros(int(duration)), 'thres': None, 'proposals': None,
    }
    if sum(map(len, ents)) == 0:
        return default_result

    df = pd.DataFrame({'ents': ents, 'start': starts, 'end': ends})  # row: [list of ents, start, end]
    df = df.explode('ents')  # row: [ent, start, end]
    df = df.dropna()
    df['iv'] = df[['start', 'end']].apply(tuple, axis=1)
    df_ner2iv = df.groupby(['ents'])['iv'].apply(list).reset_index()
    df_ner2iv['iv'] = df_ner2iv['iv'].apply(lambda ivs: merge_intervals(ivs, tol_sec=2.))

    # query-ner sim mapping with sbert or egovlp
    with torch.no_grad():
        tokens = tokenizer(
            [PROMPT_ENTITY + ent for ent in df_ner2iv['ents']] + [query],
            return_tensors='pt', padding=True, truncation=True).to(device)
        z = lm.forward(tokens)
    z = z['sentence_embedding'] if isinstance(z, dict) else z
    z_ents, z_query = z[:-1], z[-1]  # already normed
    sim = z_ents @ z_query

    sim = sim.detach().cpu()
    df_ner2iv.insert(1, 'sim', sim.numpy())  # row: [ent, sim, ivs]
    df_iv2sim = df_ner2iv.explode('iv')[['iv', 'sim']]  # row: [iv, sim]
    s_ = df_iv2sim['iv'].apply(lambda iv: iv[0]).to_numpy()
    e_ = df_iv2sim['iv'].apply(lambda iv: iv[1]).to_numpy()
    t = np.arange(duration).astype(float)
    y_ = df_iv2sim['sim'].to_numpy()[:, None] * (
        ((s_[:, None]-tol_sec-eps_sec) <= t[None, :])
        & (t[None, :] <= (e_[:, None]+tol_sec+eps_sec)))  # [#Entities, T]
    y_ = {'max': y_.max, 'mean': y_.mean, 'sum': y_.sum}[aggregate_method](axis=0)  # [T,]
    thres = np.percentile(y_, thres_percentile)
    proposals, = np.where(y_ > thres)
    if len(proposals) == 0:
        return default_result | {'y_': y_, 'thres': thres, 'proposals': proposals,}

    proposals = merge_intervals([
        (s, s+1) for s in proposals[:-1]] + [(proposals[-1]-1, proposals[-1])], tol_sec=0)
    proposed_area = sum([e-s for s, e in proposals])
    percentile_actual = proposed_area / duration
    # intersection between proposals and gt
    intersections = [
        (ss, ee) for s_, e_ in proposals
        if (ss:=max(s_, s)) < (ee:=min(e_, e))]
    iou = sum([e-s for s, e in intersections]) / proposed_area
    iogt = sum([e-s for s, e in intersections]) / (e - s)
    # return iou, iogt, proposed_area, duration, percentile_actual
    return default_result | {
        'iou': iou, 'iogt': iogt,
        'proposed_area': proposed_area,
        'percentile_actual': percentile_actual,
        'y_': y_, 'thres': thres, 'proposals': proposals,
    }


if __name__ == '__main__':
    device = 'cuda'
    lm = TextOnlyFrozenInTime(device=device)
    tokenizer = lm.tokenizer
    ds = EgoNLQDataset(tokenizer=tokenizer, split='val')
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=lambda x: x, prefetch_factor=4, persistent_workers=True)

    print('\n\n')
    from itertools import product
    for thres_percentile, aggregate_method, tol_sec in product(
        [60, 70, 80], ['max', 'mean', 'sum'], [5., 10., 15.]
    ):
        # thres_percentile, aggregate_method, tol_sec = 60, 'max', 5.
        records = []
        for batch in tqdm(dl):
            for sample in batch:
                record = evaluate_entity_proposals(
                    sample, tokenizer, device,
                    thres_percentile=thres_percentile, aggregate_method=aggregate_method, tol_sec=tol_sec)
                records.append(record)
        records = pd.DataFrame(records)
        records.to_csv(f'results/records-thresp{thres_percentile}-{aggregate_method}-tol{tol_sec:.1f}s.csv', index=False)
        print(thres_percentile, aggregate_method, tol_sec)
        print(records[['iou', 'iogt', 'ent_coverage', 'percentile_actual']].describe())
        print()
