import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed
import torch.utils.data
from tqdm import tqdm

import lightning as L

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


VALID_CLIP_UIDS = set(p.stem for p in Path('/data/datasets/ego4d_data/v2/clips_320p-non_official').glob('*.mp4'))
PATH_LOCAL_DATASETS = Path('/data2/local_datasets') if Path('/data2').exists() else Path('/local_datasets')
PATH_TOKENIZED_ROOTDIR = PATH_LOCAL_DATASETS / 'ego4d_data/v2/tokenized_captions'
PATH_TOKENIZED_ROOTDIR.mkdir(parents=True, exist_ok=True)
FPS = 30


def tokenize(tokenizer, texts: list[str], max_length: int) -> BatchEncoding:
    return tokenizer(
        texts,
        padding='max_length',  # though batch_size is 1, padding is neccessary to achieve consistent input shape
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def get_captions(captioner_name, p_cap_data, max_num_caps: int = 0, resample_method = 'linear'):
    if p_cap_data != Path('not_exists'):
        assert p_cap_data.suffix in ['.json', '.pt'], f'p_cap_data: {p_cap_data} should be .json or .pt'
    num_caps_ = max_num_caps or 10
    default_output = (np.arange(num_caps_), ['hi'] * num_caps_)

    if not p_cap_data.exists():
        tqdm.write(f'{p_cap_data} not found')
        frame_idxs, caps = default_output

    elif p_cap_data.suffix == '.pt':  # tokenized as token indices
        frame_idxs, caps = torch.load(p_cap_data)

    else:
        if captioner_name == 'VideoRecap':
            cap_data = json.load(p_cap_data.open())
            if 'captions' not in cap_data:
                frame_idxs, caps = default_output
            else:
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

    # T-slice the captions
    num_caps = len(caps) if isinstance(caps, list) else caps.input_ids.shape[0]
    if max_num_caps > 0 and num_caps > max_num_caps:
        # compute resampling indices
        if resample_method == 'linear':
            frame_idxs_target = np.linspace(frame_idxs.min(), frame_idxs.max(), 2*max_num_caps+1)[1::2].astype(int)
            args = np.argmin(np.abs(frame_idxs_target[None] - frame_idxs[:, None]), axis=0)

        elif resample_method == 'uniform':
            # e.g. 4 out of 6 -> 0., 1.5, 3., 4.5, 6. -> 0, 2, 3, 4, 6
            chunk_anchors = np.linspace(0, num_caps, max_num_caps+1, endpoint=True).round().astype(int)
            chunk_sizes = np.diff(chunk_anchors)  # e.g. 2, 1, 1, 2
            assert (chunk_sizes > 0).all(), \
                f'chunk_sizes: {chunk_sizes} should be all positive '\
                'because max_num_caps is always less than num_caps'
            chunk_offsets = np.random.randint(0, chunk_sizes, size=max_num_caps)  # e.g. 1, 0, 0, 1
            args = np.clip(chunk_anchors[:-1] + chunk_offsets, 0, num_caps-1)  # e.g. 1, 2, 3, 5

        # perform resampling
        frame_idxs = frame_idxs[args]
        if isinstance(caps, list):
            caps = [caps[i] for i in args]
        elif isinstance(caps, BatchEncoding):
            caps['input_ids'] = caps.input_ids[args]
            caps['attention_mask'] = caps.attention_mask[args]
            caps['token_type_ids'] = caps.token_type_ids[args]

    return frame_idxs, caps


def get_tokenizer(tokenizer_name: str):
    if tokenizer_name == 'egovlp':
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


class NLQDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='val'):
        self.config = config
        dataset_config = config.dataset
        self.captioner_name = dataset_config.captioner_name
        self.tokenizer_name = dataset_config.tokenizer_name

        self.caption_pair_relation = dataset_config.get('caption_pair_relation')

        self.N_cap: int = dataset_config.max_num_caps  # 0 for no truncation
        self.L_cap: int = dataset_config.max_cap_len
        self.L_q:   int = dataset_config.max_q_len

        self.split = split

        # prepare processors like a tokenizer
        self.tokenizer = get_tokenizer(self.tokenizer_name)

        # prepare official NLQ annotations
        self.p_anns_dir = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/unified/')
        if split in ['train', 'val']:
            self.p_ann = self.p_anns_dir / f'annotations.NLQ_{split}.json'
            self.ann_data = [
                q for q in json.load(open(self.p_ann))
                if q['video_id'] in VALID_CLIP_UIDS]
        elif split == 'trainval':
            self.ann_data = []
            for split in ['train', 'val']:
                p_ann = self.p_anns_dir / f'annotations.NLQ_{split}.json'
                self.ann_data.extend([
                    q for q in json.load(open(p_ann))
                    if q['video_id'] in VALID_CLIP_UIDS])

        # validity of a clip uid in this context
        #   = in the official NLQ annotations && in the clip directory
        self.clip_uids = sorted(set(q['video_id'] for q in self.ann_data))

        # prepare caption data paths
        self.p_caps_dir = Path('data/captions') / self.captioner_name
        self.p_cap_tokens_dir = PATH_TOKENIZED_ROOTDIR / self.captioner_name / f'{self.tokenizer_name}-{self.L_cap:03d}'
        self.p_cap_data_map: dict[str, Path] = {}  # json or tokenized pt file
        for p_cap_data in sorted(self.p_caps_dir.glob('**/*.json')):
            clip_uid = p_cap_data.stem
            if clip_uid not in self.clip_uids or 'local' in str(p_cap_data):
                continue
            p_cap_tokens = self.p_cap_tokens_dir / p_cap_data.relative_to(self.p_caps_dir).with_suffix('.pt')
            if p_cap_tokens.exists():
                self.p_cap_data_map[clip_uid] = p_cap_tokens
            elif p_cap_data.exists():
                self.p_cap_data_map[clip_uid] = p_cap_data
            else:
                raise FileNotFoundError(f'{p_cap_tokens} nor {p_cap_data} not found')

        print(f'NLQDataset: {len(self.clip_uids)} clips, {len(self.ann_data)} anns')

    def __len__(self):
        # return len(self.clip_uids)
        return len(self.ann_data)  # num all queries

    def __getitem__(self, idx):
        ann = self.ann_data[idx]

        # captions and caption tokens
        clip_uid = ann['video_id']
        p_cap_data = self.p_cap_data_map.get(clip_uid, Path('not_exists'))
        if self.split == 'train':
            frame_idxs, caps = get_captions(self.captioner_name, p_cap_data, self.N_cap, resample_method='uniform')
        elif self.split == 'val':
            frame_idxs, caps = get_captions(self.captioner_name, p_cap_data, self.N_cap)  # N_cap
        else:
            raise ValueError(f'split: {self.split}')

        if isinstance(caps, list):
            cap_tokens = tokenize(self.tokenizer, caps, self.L_cap)
        elif isinstance(caps, BatchEncoding):
            cap_tokens = caps
            if cap_tokens.input_ids.shape[-1] > self.L_cap:
                cap_tokens['input_ids'] = cap_tokens.input_ids[:, :self.L_cap]
                cap_tokens['attention_mask'] = cap_tokens.attention_mask[:, :self.L_cap]
                cap_tokens['token_type_ids'] = cap_tokens.token_type_ids[:, :self.L_cap]
        else:
            raise ValueError(f'caps: {caps}')

        # query and query tokens
        query = ann['question']
        sample_id = ann['sample_id']
        q_tokens = tokenize(self.tokenizer, [query], self.L_q)

        # GT for the query-caption relation
            # compute gt_mat, whose (i, j) element is 1.0 if
            #   the j-th caption range contacts the i-th query's GT interval(segment)
        num_caps = cap_tokens.input_ids.shape[0]  # can be != self.N_cap
        segment = np.array([ann['clip_start_sec'], ann['clip_end_sec']])
        gt_mat = np.zeros((1, num_caps), dtype=float)
        for i, (s, e) in enumerate([segment]):
            s_idx, e_idx = int(s * FPS), int(e * FPS)
            s_ord = np.argmin(np.abs(frame_idxs - s_idx))
            e_ord = np.argmin(np.abs(frame_idxs - e_idx))
            gt_mat[i, s_ord:e_ord+1] = 1.
            assert gt_mat[i].any(), \
                f'gt_mat[{i}] is all zeros\n'\
                f'gt_mat: {gt_mat}\n' \
                f's: {s}, e: {e}, s_idx: {s_idx}, e_idx: {e_idx}, s_ord: {s_ord}, e_ord: {e_ord}\n' \
                f'{p_cap_data}\n' \
                f'{ann}'

        # GT for caption-caption relation
        cap_mat = None
        if self.caption_pair_relation is None:
            pass
        elif self.caption_pair_relation == 'adjacency':
            # tridiagonal matrix with all ones
            cap_mat = np.tri(num_caps, num_caps, k=1) - np.tri(num_caps, num_caps, k=-2)
        elif self.caption_pair_relation == 'color_historgram_similarity':
            p_hist = Path('data/hists/bins-16') / f'{clip_uid}.pt'
            hist = torch.load(p_hist)  # [N_cap, 16 x 3]
            num_hists = hist.shape[0]

            hist_idxs = frame_idxs + 10*np.random.randn(num_caps)  # prob of offset <=10 frames = 0.68
            hist_idxs = np.clip(hist_idxs.round(), 0, num_hists-1).astype(int)
            hist_sampled = hist[hist_idxs]
            dist_mat = torch.cdist(hist_sampled, hist_sampled, p=2)  # [N_cap, N_cap], L2 distance
            cap_mat = 1. - dist_mat / dist_mat[~torch.eye(num_caps, dtype=bool)].max()
            cap_mat[torch.eye(num_caps, dtype=bool)] = 1.

            # hist = torch.load(p_hist)[frame_idxs]  # [N_cap, 16 x 3]
            # dist_mat = torch.cdist(hist, hist, p=2)  # [N_cap, N_cap], L2 distance
            # cap_mat = 1. - dist_mat / dist_mat.max()

        # convert to tensors
        totensor = lambda x: (
            x.to(torch.float)
            if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float))  # automatically selects bit-width
        return {
            # infos
            'clip_uid': clip_uid,
            'captions': caps,  # [N_cap]
            'query': query,
            'sample_id': sample_id,

            # tensors
            'segments': totensor(segment[None]),  # [N_q, 2]
            'caption_frame_idxs': totensor(frame_idxs),  # [N_cap]
            'cap_tokens': cap_tokens,  # [N_cap, L_cap]
            'q_tokens': q_tokens,  # [N_q, L_q]
            'gt_mat': totensor(gt_mat),  # [N_q, N_cap]
            'cap_mat': totensor(cap_mat),  # [N_cap, N_cap]
        }


class NLQPredictionDatset(NLQDataset):
    def __init__(self, config):
        super().__init__(config, split='trainval')
        self.ann_data_map = defaultdict(list)
        for ann in self.ann_data:
            clip_uid = ann['video_id']
            self.ann_data_map[clip_uid].append(ann)
        assert len(self.clip_uids) == len(self.ann_data_map)

    def __len__(self):
        return len(self.clip_uids)

    def __getitem__(self, idx):
        # captions and caption tokens
        clip_uid = self.clip_uids[idx]
        p_cap_data = self.p_cap_data_map[clip_uid]
        frame_idxs, caps = get_captions(self.captioner_name, p_cap_data, max_num_caps=0)
        if isinstance(caps, BatchEncoding):
            cap_tokens = caps
        else:
            cap_tokens = tokenize(self.tokenizer, caps, self.L_cap)

        # query and query tokens
        queries, sample_ids = [], []
        for ann in self.ann_data_map[clip_uid]:
            queries.append(ann['question'])
            sample_ids.append(ann['sample_id'])
        q_tokens = tokenize(self.tokenizer, queries, self.L_q)

        return {
            'clip_uid': clip_uid,
            'sample_ids': sample_ids,
            'queries': queries,

            'caption_frame_idxs': frame_idxs,
            'cap_tokens': cap_tokens,
            'q_tokens': q_tokens,
        }


class LitDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dataset_config = config.dataset
        # nominal batch size should be fixed to 1, as the number of clip captions might solely go over the model's max_length
        self.nominal_batch_size = 1
        assert self.nominal_batch_size == 1
        self.num_workers = dataset_config.num_workers

        self.dl_params = dict(
            batch_size=self.nominal_batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
            prefetch_factor=dataset_config.prefetch_factor,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
        )

    def prepare_data(self):  # pre-compute caption tokens according to the given length
        captioner_name = self.config.dataset.captioner_name
        max_cap_len = self.config.dataset.max_cap_len
        tokenizer_name = self.config.dataset.tokenizer_name
        batch_flag = self.config.batch_flag
        force_retokenize = self.config.dataset.force_retokenize

        p_caps_root_dir = Path('data/captions') / captioner_name
        assert p_caps_root_dir.exists()
        p_caps = sorted(p for p in p_caps_root_dir.glob('**/*.json') if 'local' not in str(p))
        p_out_pts_dir = PATH_TOKENIZED_ROOTDIR / captioner_name / f'{tokenizer_name}-{max_cap_len:03d}'
        p_out_pts_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = get_tokenizer(tokenizer_name)

        pbar = tqdm(p_caps, dynamic_ncols=True, disable=batch_flag=='1')
        all_time_longest = 0
        for i, p_cap_json in enumerate(pbar):
            # read infos
            clip_uid = p_cap_json.stem
            pbar.set_description(clip_uid)
            pr_cap_json = p_cap_json.relative_to(p_caps_root_dir)
            p_out_pt = p_out_pts_dir / pr_cap_json.with_suffix('.pt')
            if p_out_pt.exists() and not force_retokenize:
                continue
            p_out_pt.parent.mkdir(parents=True, exist_ok=True)

            # read, process, and save the data
            frame_idxs, caps = get_captions(captioner_name, p_cap_json, max_num_caps=0)
            tokens = tokenize(tokenizer, caps, max_cap_len)
            torch.save((frame_idxs, tokens), p_out_pt)

            # log
            num_valid_tokens = tokens.attention_mask.sum(dim=1)
            longest_idx = num_valid_tokens.argmax().item()
            longest_cap = caps[longest_idx]
            longest_length = num_valid_tokens[longest_idx].item()
            all_time_longest = max(all_time_longest, longest_length)
            pbar.set_postfix(current_longest=all_time_longest)
            if i % 100 == 0:
                pbar.write(f'[{clip_uid}]\nLongest caption over {len(caps)} caps with {longest_length} tokens:\n{longest_cap}\n\n')
        pbar.write(f'Saved tokenized captions to {p_out_pts_dir}')

    def setup(self, stage=None):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # BUG: Occasionally raises timeout when prepare_data called

        if stage in ['fit', 'validate', 'train']:
            self.datasets = {
                'train': NLQDataset(self.config, split='train'),
                'val': NLQDataset(self.config, split='val'),
            }

        elif stage == 'predict':
            self.datasets = {
                'pred': (ds:=NLQPredictionDatset(self.config)),
            }
            if self.trainer.global_rank == 0:
                print(f'\n\nL_cap: {ds.L_cap}, N_cap: {ds.N_cap}\n\n')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets['train'],
            shuffle=True,
            drop_last=True,
            **self.dl_params,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets['val'],
            shuffle=False,
            **self.dl_params,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets['pred'],
            shuffle=False,
            **self.dl_params,
        )


if __name__ == '__main__':
    def make_fake_config():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--captioner_name', type=str, default='llava-v1.6-34b')
        parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2')
        parser.add_argument('--max_num_caps', type=int, default=96, help='Effectively the batch size for the model')
        parser.add_argument('--max_cap_len', type=int, default=256)
        parser.add_argument('--max_q_len', type=int, default=22)
        config = parser.parse_args()
        return config

    import os
    import shutil
    from pprint import pformat
    from tqdm import tqdm
    config = make_fake_config()
    config.dataset = config
    config.dataset.tokenizer_name = config.model_name
    dataset = torch.utils.data.ConcatDataset([
        NLQDataset(config, split='train'),
        NLQDataset(config, split='val')
    ])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # nominal batch size should be fixed to 1, as merely the number of clip captions might go over the model's max_length
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=lambda x: x[0],
        prefetch_factor=64,
    )

    # print
    cli_width = shutil.get_terminal_size().columns
    torch.set_printoptions(sci_mode=False, linewidth=cli_width-20)
    for idx, batch in enumerate(tqdm(dataloader)):
        msg = pformat(batch, compact=True, width=cli_width, sort_dicts=False)
        if idx % 1000 == 1:
            tqdm.write(msg)
