import os
import json
from pathlib import Path

import torch
import torch.distributed
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate_fn_map, default_collate

import pandas as pd
import numpy as np
from einops import rearrange
from transformers import PreTrainedTokenizer, AutoTokenizer

import lightning as L
from transformers.tokenization_utils_base import BatchEncoding
default_collate_fn_map[pd.DataFrame] = lambda batch, *args, **kwargs: batch


class EgoNLQRawDataset(torch.utils.data.Dataset):
    @classmethod
    def test_me(cls):
        print(f'\n======== Testing {cls.__name__} ========')
        from pprint import pprint
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        ds = EgoNLQRawDataset(tokenizer=tokenizer, max_t=3, max_l=5)
        pprint(len(ds))
        tokens = ds.tokenize_captions_batch([ds[0]['captions']['text'].tolist()] * 2)
        pprint(tokens)
        pprint(tokens.input_ids.shape)  # [2, 3, 5]
        dl = torch.utils.data.DataLoader(ds, batch_size=2,)
        for batch in dl:
            pprint(batch['query_tokens']['input_ids'].shape)
            pprint(batch['captions_tokens']['input_ids'].shape)
            pprint(batch['captions'])
            break
        print('\n\n')

    def __init__(self,
        tokenizer: PreTrainedTokenizer,
        p_nlq_csvs_dir = Path('data/Ego4D/EgoNLQ/csvs'),
        split = 'train',
        max_t = 256,  # T, GVQA는 1200 씀
        max_l = 22,  # L
        caption_stride_sec = 2,
        proposal_mode = False, proposal_width_sec = 30.,
        gather_consecutive_duplicated_captions = False,
    ):
        self.p_nlq_csvs_dir = p_nlq_csvs_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_t = max_t
        self.max_l = max_l
        self.caption_stride_sec = caption_stride_sec
        p_caps_dir = Path('data/Ego4D-processed/captions/VideoRecap')
        self.proposal_mode = proposal_mode
        self.proposal_width_sec = proposal_width_sec
        self.p_cap = p_caps_dir / f'caption_{self.caption_stride_sec:d}s/{split}/VideoRecap_{split}_gathered.json'
        self.p_nlq = self.p_nlq_csvs_dir / f'nlq_{self.split}_v2.csv'
        self.gather_consecutive_duplicated_captions = gather_consecutive_duplicated_captions

        # load data
        _caps = json.load(self.p_cap.open())
        self.caps = {c['clip_uid']: c for c in _caps['clips'] if 'clip_uid' in c}
        _df = pd.read_csv(self.p_nlq)
        self.df_nlq = _df[_df['clip_uid'].isin(self.caps.keys())].reset_index(drop=True)
        rank_prefix = f'[rank: {r}] ' if (r:=os.environ.get('RANK', '')) else ''
        print(rank_prefix + f'Loaded {len(self.df_nlq)} samples from {self.p_nlq}')
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def sanitize_captions(self, captions: pd.Series) -> pd.Series:
        return (
            captions
            .str.replace(r'^[cC] ', '#C C ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)  # remove duplicated spaces
            .str.replace(r'\b(\w\w+)\b\s+\1\b', r'\1', regex=True)  # remove duplicated words
            # female pronouns to male's
            .str.replace(r'\bher\b', 'his', regex=True)
            .str.replace(r'\bshe\b', 'he', regex=True)
            .str.replace(r'\bhers\b', 'his', regex=True)
            .str.strip().str.strip('.')
        )

    def suppress_df_caps(self, df_caps, duration):
        df_caps['_text'] = (
            df_caps['text']
            .str.lower()
            .str.replace(r'^#\w\s+', '', regex=True)
            .str.replace(r'\b(\w+)\b\s+\1\b', r'\1', regex=True)  # remove duplicated words
            .str.replace(r'\bthe\b', '', regex=True)  # remove "the"'s
            .str.replace(r'\ba\b', '', regex=True)  # remove "a"'s
            .str.replace(r'\s+', ' ', regex=True)  # remove duplicated spaces
            .str.strip().str.strip('.')  # need to do this again because of the replacements
        )
        df_caps = df_caps[df_caps['_text'] != df_caps['_text'].shift(1)]
        df_caps['end'] = df_caps['start'].shift(-1, fill_value=duration)
        del df_caps['_text']
        return df_caps

    def tokenize(self, text_batch: str|list[str]) -> BatchEncoding:
        tokens = self.tokenizer(
            text_batch,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=self.max_l,
            return_tensors="pt")  # [B, L]
        if isinstance(text_batch, str):
            tokens = {key: val.squeeze(0) for key, val in tokens.items()}
        return tokens

    def tokenize_captions(self, captions: list[str]) -> BatchEncoding:
        if len(captions) > self.max_t:
            captions = captions[:self.max_t]
        elif len(captions) < self.max_t:
            captions += [''] * (self.max_t - len(captions))
        return self.tokenize(captions)

    def tokenize_captions_batch(self, captions_batch: list[list[str]]) -> BatchEncoding:
        captions_tokens_batch = []
        for captions in captions_batch:
            captions_tokens_batch.append(self.tokenize_captions(captions))
        captions_tokens_batch = BatchEncoding({
            key: torch.stack([elem[key] for elem in captions_tokens_batch], dim=0)
            for key in ['input_ids', 'attention_mask']})  # [B, T, L]
        return captions_tokens_batch

    def __len__(self):
        return len(self.df_nlq)

    def __getitem__(self, idx):
        row = self.df_nlq.iloc[idx]
        clip_uid = row['clip_uid']
        video_uid = row['video_uid']
        query_id = f"{row['annotation_uid']}_{row['query_idx']}"
        query = row['query']
        clip_duration = row['duration_sec']
        s, e = row[['q_clip_start_sec', 'q_clip_end_sec']].values

        df_caps = pd.DataFrame(self.caps[clip_uid]['captions'])
        df_caps['text'] = self.sanitize_captions(df_caps['text'])
        if self.gather_consecutive_duplicated_captions:
            df_caps = self.suppress_df_caps(df_caps, clip_duration)
        gt_segment_sec = torch.tensor([s, e])
        if self.proposal_mode:
            c = (s + e) / 2
            w = self.proposal_width_sec
            gt_segment = torch.tensor([c - w/2, c + w/2]) / self.caption_stride_sec
        else:
            gt_segment = gt_segment_sec / self.caption_stride_sec
        return {
            'clip_uid': clip_uid,
            'video_uid': video_uid,
            'query_id': query_id,
            'duration': clip_duration,
            'query': query,
            'captions': df_caps,
            'gt_start_sec': s,  # in seconds
            'gt_end_sec': e,  # in seconds

            'query_tokens': self.tokenize(query),
            'captions_tokens': self.tokenize_captions(df_caps['text'].tolist()),
            # for train
            'gt_segment': gt_segment,  # effective indices
            # for valid
            'gt_segment_sec': gt_segment_sec,
        }


class EgoNLQRawDataModule(L.LightningDataModule):
    @classmethod
    def test_me(cls):
        print(f'\n======== Testing {cls.__name__} ========')
        from pprint import pprint
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        dm = cls(tokenizer=tokenizer, max_t=256, max_l=22)
        dm.setup()
        pprint(len(dm.train_ds))
        pprint(len(dm.val_ds))
        dl = dm.train_dataloader()
        for batch in dl:
            pprint(f"B, T, L = {batch['captions_tokens']['input_ids'].shape}")
            pprint(f"B, L = {batch['query_tokens']['input_ids'].shape}")
            pprint(batch['captions'])
            break
        print('\n\n')

    def __init__(
        self,
        # DS params
        tokenizer = None,
        caption_stride_sec = 2,
        proposal_mode = False, proposal_width_sec = 30.,
        max_t = 256,
        max_l = 22,
        # DL params
        batch_size = 4,
        **dl_params,
    ):
        super().__init__()
        assert tokenizer is not None, 'tokenizer is required'
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.ds_params = dict(
            tokenizer=tokenizer, max_t=max_t, max_l=max_l,
            caption_stride_sec=caption_stride_sec,
            proposal_mode=proposal_mode, proposal_width_sec=proposal_width_sec,
            gather_consecutive_duplicated_captions=False,
        )
        self.dl_params = dict(
            batch_size=batch_size, drop_last=True,
            num_workers=8, prefetch_factor=2,
            persistent_workers=False, pin_memory=False,
            collate_fn=self.collate_fn,
        ) | dl_params
        self.save_hyperparameters()

    @staticmethod
    def collate_fn(batch):
        batch = default_collate(batch)
        for token_key in (key for key in batch.keys() if 'token' in key):
            batch[token_key] = BatchEncoding(batch[token_key])
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_ds = EgoNLQRawDataset(split='train', **self.ds_params)
        self.val_ds = EgoNLQRawDataset(split='val', **self.ds_params)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_params)

    test_dataloader = val_dataloader
    predict_dataloader = val_dataloader


if __name__ == '__main__':
    EgoNLQRawDataset.test_me()
    EgoNLQRawDataModule.test_me()
