import logging
from pathlib import Path

from tqdm import tqdm
import cv2
from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from ltvu.models.base import BasicTransformerEncoder
from ltvu.models.heads.nlq_head import NLQHead


logger = logging.getLogger(__name__)


def hist_encoding(frame, bins=16):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [bins], [0, 256])
    hist = hist / np.sum(hist)
    return hist.flatten()


def read_video(video_path, image_transform=None):
    if not video_path.exists():
        raise FileNotFoundError(f'{video_path} does not exist')
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if image_transform is not None:
            frame = image_transform(frame)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def hist_encoding_video(video_path, bins=16):
    transform = lambda frame: hist_encoding(frame, bins=bins)
    hist = read_video(video_path, image_transform=transform)
    return hist


class EgoNLQClipOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
        split = 'train', version = 'v2',
        p_nlq_csvs_dir = Path('data/Ego4D/EgoNLQ/csvs/'),
        p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/'),
        image_transform = None,
    ):
        self.p_nlq_csv = p_nlq_csvs_dir / f'nlq_{split}_{version}.csv'
        self.clip_uids = pd.read_csv(self.p_nlq_csv)['clip_uid'].unique().tolist()
        self.p_clips_dir = p_clips_dir
        self.image_transform = image_transform

    def __len__(self):
        return len(self.clip_uids)

    def __getitem__(self, idx):
        clip_uid = self.clip_uids[idx]
        p_clip = self.p_clips_dir / f'{clip_uid}.mp4'
        clip = read_video(p_clip, image_transform=self.image_transform)
        instance = {
            'clip_uid': clip_uid,
            'clip': clip,
        }
        return instance


def generate_hists(bins=16, split='train'):
    p_hists_dir = Path(f'data/Ego4D-processed/hists/bins-{bins}')
    p_hists_dir.mkdir(exist_ok=True, parents=True)
    transform = lambda frame: hist_encoding(frame, bins=bins)
    dataset = EgoNLQClipOnlyDataset(split=split, image_transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2,
        num_workers=32, shuffle=False, pin_memory=False, drop_last=False,
        collate_fn=lambda x: x,
        prefetch_factor=32, persistent_workers=True)
    for batch in tqdm(dataloader):
        for inst in batch:
            clip_uid = inst['clip_uid']
            hist = inst['clip']
            if hist.shape[0] < 10:
                logger.warning(f'{clip_uid}: {hist.shape[0]} frames is too short for a video clip')
                continue
            np.save(p_hists_dir / f'{clip_uid}.npy', hist)


class EgoNLQHistogramDataset(torch.utils.data.Dataset):
    def __init__(self,
        split = 'train', version = 'v2',
        tokenizer = None,
        max_v_len = 14400,  # in frames
        max_q_len = 32,
        hist_bins = 16,
        hist_gather_factor = 16,
        p_nlq_csvs_dir = Path('data/Ego4D/EgoNLQ/csvs/'),
        p_hists_dir = Path('data/Ego4D-processed/hists/bins-16'),
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                'distilbert-base-uncased', TOKENIZERS_PARALLELISM=False)
            logger.info('Initialized tokenizer from AutoTokenizer')
        self.tokenizer = tokenizer
        self.max_v_len = max_v_len // hist_gather_factor
        self.max_q_len = max_q_len
        self.hist_bins = hist_bins
        self.hist_gather_factor = hist_gather_factor
        self.p_nlq_csv = p_nlq_csvs_dir / f'nlq_{split}_{version}.csv'
        df_nlq = pd.read_csv(self.p_nlq_csv)
        self.df_nlq = df_nlq[df_nlq['clip_uid'].isin([p.stem for p in p_hists_dir.glob('*.npy')])]
        self.p_hists_dir = p_hists_dir

    def __len__(self):
        return len(self.df_nlq)

    def __getitem__(self, idx):
        row = self.df_nlq.iloc[idx]
        clip_uid, query, s, e = row[['clip_uid', 'query', 'q_clip_start_sec', 'q_clip_end_sec']]
        p_hist = self.p_hists_dir / f'{clip_uid}.npy'
        hist = torch.from_numpy(np.load(p_hist))
        if self.hist_gather_factor > 1:
            len_v = hist.shape[0] // self.hist_gather_factor * self.hist_gather_factor
            hist = rearrange(hist[:len_v], '(lv g) d -> lv (g d)', g=self.hist_gather_factor)
        len_v = hist.shape[0]
        assert len_v > 10, f'{clip_uid}: {len_v} frames is too short for a video clip'
        if len_v > self.max_v_len:
            idxs = torch.linspace(0, len_v - 1, self.max_v_len, dtype=torch.int64)
            hist = hist[idxs]
        segment = torch.tensor([[s, e]]) * 30 / self.hist_gather_factor  # [# events = 1, 2], feature idx
        instance = {
            'clip_uid': clip_uid,
            'gt_start_sec': s,
            'gt_end_sec': e,
            'hist': hist,
            'query': query,
            'gt_segment': segment,
        }
        return instance

    def tokenize_text(self, text_batch):
        tokens = self.tokenizer(
            text_batch,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
        )
        return tokens

    def collate_fn(self, batch: list[dict]):
        hist_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.zeros(instance['hist'].shape[0]) for instance in batch], batch_first=True, padding_value=1).bool()
        hist_batch = torch.nn.utils.rnn.pad_sequence(
            [instance['hist'] for instance in batch], batch_first=True, padding_value=0)
        q_batch = [instance['query'] for instance in batch]
        q_tokens = self.tokenize_text(q_batch)
        return {
            'clip_uids': [instance['clip_uid'] for instance in batch],
            'gt_start_secs': torch.tensor([instance['gt_start_sec'] for instance in batch]),
            'gt_end_secs': torch.tensor([instance['gt_end_sec'] for instance in batch]),
            'queries': q_batch,
            'ctx_tokens': hist_batch,
            'ctx_masks': hist_mask,  # 0 for valid, 1 for padding following torch convention
            'qry_tokens': q_tokens['input_ids'],
            'qry_masks': q_tokens['attention_mask'],  # 1 for valid, 0 for padding following the HuggingFace convention
            'gt_segments': torch.stack([instance['gt_segment'] for instance in batch]),
        }


class BasicNLQModel(nn.Module):
    def __init__(self, max_seq_len=1000):
        super().__init__()
        d = 256
        self.ctx_in_proj = nn.Linear(d, d)
        self.ctx_memb = nn.Parameter(torch.randn((1, 1, d)))  # mode embedding
        self.qry_enc = SentenceTransformer("all-mpnet-base-v2")
        self.qry_mid_proj = nn.Linear(768, d)
        self.comb_enc = BasicTransformerEncoder(
            d_input = d,
            d_model = d,
            d_output = d,
            nhead = 4,
            dim_feedforward = 4 * d,
            activation = 'gelu',
            num_layers = 6,
            dropout = 0.,
            droppath = 0.,
            prenorm = True,
            max_len = max_seq_len,
        )
        self.nlq_head = NLQHead(d, max_v_len=max_seq_len)

    @property
    def device(self):
        return self.ctx_memb.device

    def forward(self, ctx_tokens, ctx_masks, qry_tokens, qry_masks, gt_segments, **kwargs):
        ctx_tokens, ctx_masks, qry_tokens, qry_masks, gt_segments = map(
            lambda x: x.to(self.device), [ctx_tokens, ctx_masks, qry_tokens, qry_masks, gt_segments])
        z_comb = self.forward_encoder(ctx_tokens, ctx_masks, qry_tokens, qry_masks, **kwargs)
        if self.training:
            loss = self.forward_head(z_comb[:, 1:], ctx_masks, gt_segments, **kwargs)
            return {'loss': loss}

    def forward_encoder(self, ctx_tokens, ctx_masks, qry_tokens, qry_masks, **kwargs):
        B = ctx_tokens.shape[0]
        valid_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        z_ctx = self.forward_context(ctx_tokens)  # [B, L_ctx, D]
        z_qry = self.forward_query(qry_tokens, qry_masks)  # [B, D]
        z_comb = torch.cat([z_qry[:, None], z_ctx], dim=1)  # [B, 1+L_ctx, D]
        z_mask = torch.cat([valid_mask, ctx_masks], dim=1)  # [B, 1+L_ctx]
        z_comb = self.comb_enc.forward(z_comb, padding_mask=z_mask)  # [B, 1+L_ctx, D]
        return z_comb

    def forward_head(self, z_ctx, ctx_masks, gt_segments, **kwargs):
        nlq_results = self.nlq_head.forward(
            feat=rearrange(z_ctx, '... l_ctx d -> ... d l_ctx'),  # [B, D, L_ctx]
            mask=ctx_masks.unsqueeze(1),   # [B, 1, L_ctx]
            training=self.training,
            gt_segments=gt_segments)
        if self.training:
            for k, v in nlq_results.items():
                print(f'{k}: {v.item():.7f}', end=' ')
            print()
            loss = nlq_results['final_loss']
            return loss

    def forward_context(self, ctx_tokens):
        return F.normalize(self.ctx_in_proj(ctx_tokens)) + self.ctx_memb

    def forward_query(self, qry_tokens, qry_masks):
        z = self.qry_enc.forward({
            'input_ids': qry_tokens, 'attention_mask': qry_masks,
        })['sentence_embedding']
        z = F.normalize(z)
        z = self.qry_mid_proj(z)
        return z

if __name__ == '__main__':
    generate_hists(bins=16, split='val')


# if __name__ == '__main__':
#     import os
#     seed_everything(42, workers=True)
#     logger.setLevel(logging.DEBUG)

#     B, fact = 8, 1
#     lr = 1e-3
#     eps = 1e-3
#     rng = np.random.default_rng()

#     model = BasicNLQModel(max_seq_len=1000).cuda()
#     dataset = EgoNLQHistogramDataset(tokenizer=model.qry_enc.tokenizer)
#     _idxs_subset = rng.choice(len(dataset), B*fact, replace=False)
#     dataloader = torch.utils.data.DataLoader(
#         torch.utils.data.Subset(dataset, _idxs_subset),
#         batch_size=B, num_workers=8, collate_fn=dataset.collate_fn, shuffle=False,
#         pin_memory=True, drop_last=True, persistent_workers=True)

#     no_decay_params = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [], 'lr': lr, 'weight_decay': 1e-2},
#         {'params': [], 'lr': lr, 'weight_decay': 0.}
#     ]
#     for n, p in model.named_parameters():
#         if any(nd in n for nd in no_decay_params):
#             group = 1
#         else:
#             group = 0
#         optimizer_grouped_parameters[group]['params'].append(p)
#     optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

#     for epoch in range(1000):
#         losses_epoch = []
#         print(f'#################### Epoch {epoch+1:02d} ####################')
#         model.train(); model.zero_grad()
#         for batch in tqdm(dataloader):
#             print(set(batch['clip_uids']))
#             batch['ctx_tokens'] = (batch['ctx_tokens'] / (1-eps) + eps).log() / np.log(1/eps)
#             loss = model.forward(**batch)['loss']
#             losses_epoch.append(loss.item())
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#             optimizer.step()
#             optimizer.zero_grad()
#         vram_usage = ' '.join(os.popen('nvidia-smi --query-gpu=memory.used --format=csv').read().strip().split('\n')[1:])
#         print(f'loss: {np.mean(losses_epoch):9.7f}, VRAM: {vram_usage}')
