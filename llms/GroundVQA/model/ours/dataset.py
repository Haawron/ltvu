# Joint dataset of CloseQA, OpenQA, and NLQ

import os
import math
import json
import random
from pathlib import Path
from typing import Iterable, Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer


class BaseDataset(Dataset):
    def __init__(self, data_dir, split, feature_type, max_v_len):
        super().__init__()
        self.split = split
        self.video_features = h5py.File(os.path.join(data_dir, feature_type + '.hdf5'), 'r')
        self.annotations = json.loads(Path(os.path.join(data_dir, f'annotations.{split}.json')).read_text())
        self.max_v_len = max_v_len
        print(f'{split} set: {len(self.annotations)}')

    def __len__(self):
        return len(self.annotations)

    def _get_video_feature(self, video_id):
        video_feature = torch.from_numpy(self.video_features[video_id][:])
        v_len = video_feature.shape[0]
        sample_ratio = 1.0
        if v_len > self.max_v_len:
            sample_idx = torch.linspace(0, v_len-1, self.max_v_len).long()
            video_feature = video_feature[sample_idx]
            sample_ratio = self.max_v_len / v_len
            v_len = self.max_v_len
        return video_feature, v_len, sample_ratio


class NLQDataset(BaseDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len):
        super().__init__(data_dir, split, feature_type, max_v_len)

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']

        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        if 'clip_start_sec' in self.annotations[index]:
            start_time = self.annotations[index].get('clip_start_sec')
            end_time = self.annotations[index].get('clip_end_sec')
        else:
            start_time = self.annotations[index].get('moment_start_frame') / 30
            end_time = self.annotations[index].get('moment_end_frame') / 30

        query_type = self.annotations[index].get('query_type')
        if query_type == 'narration':  # only for NaQ, otherwise GT s, e are provided
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)

        return {
            'video_id': video_id,  # same as clip_uid here
            'question': f"question: {question} video: ",
            'answer': 'None',
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': 'NLQ'
        }


SAMPLE_CLIP_UID = 'f06d1935-550f-4caa-909c-b2db4c28f599'
class NLQDatasetOnLLaVA(NLQDataset):
    def __init__(
        self, data_dir, split, feature_type, max_v_len,
        llava_dir: str,
        scope: Literal['global', 'local', 'both'],
        # target_stride_sec_global: float,
        # target_stride_sec_local: float,
        load_feature: bool,
        feature_aggregation: Literal['init', 'answer', 'all']
    ):
        super().__init__(data_dir, split, feature_type, max_v_len)
        self.p_llava_dir = Path(llava_dir)
        valid_clip_uids = set(p.stem for p in self.p_llava_dir.glob('**/*.pt'))
        required_clip_uids = set(a['video_id'] for a in self.annotations)
        diff = required_clip_uids - valid_clip_uids
        print(f'Clips not existing in LLaVA: {diff} ({len(diff)})')
        self.annotations = [a for a in self.annotations if a['video_id'] in valid_clip_uids]
        self.scope = scope
        # self.target_stride_sec_global = target_stride_sec_global
        # self.target_stride_sec_local = target_stride_sec_local
        self.load_feature = load_feature
        self.feature_aggregation = feature_aggregation

    def __getitem__(self, index):
        # TODO: 인풋 옵션
            # [x]: z_init vs. z_answer vs. z_all (vs. z_init[:non_pad]; 이렇게 안 뽑아서 아예 불가)
            # [ ]: vs. answer -> token (-> Sentence Embedding)
            # [ ]: vs. (answer, query) -> token (-> Cross Embedding)
            # [ ]: global vs. local vs. both --> local 다루려면 파일 포맷을 제대로 정해야 함
        output: dict = super().__getitem__(index)
        video_id = output['video_id']
        p_llava_feature = self.p_llava_dir / self.scope / f'{video_id}.pt'
        llava_feature_data = torch.load(p_llava_feature, map_location='cpu')
        llava_features = {}
        llava_features[self.scope] = self._get_llava_feature(
            llava_feature_data)[self.feature_aggregation]
        output['llava_feat'] = llava_features
        return output

    def _get_llava_feature(
        self,
        llava_feature_data,
        T_target = 16,
        # T_vfeat,
        # target_stride_sec=30.,
    ):
        FPS = 30
        llava_features = {'init': [], 'answer': [], 'all': []}
        for frame_idx, source_scope, (num_in_tokens, num_out_tokens), z_init, z_answer in llava_feature_data:
            z_all = (num_in_tokens * z_init + num_out_tokens * z_answer) / (num_in_tokens + num_out_tokens)
            llava_features['init'].append(z_init.squeeze())  # [D_llava=4096(8B) or 7168(34B)]
            llava_features['answer'].append(z_answer.squeeze())
            llava_features['all'].append(z_all.squeeze())
        # source_stride_frame = llava_feature_data[1][0] - llava_feature_data[0][0]  # 300 = 10s
        # target_stride_frame = target_stride_sec * FPS
        # target_stride_index = source_stride_frame / target_stride_frame  # 3
        T_source = len(llava_feature_data)
        # T_target = math.ceil(T_source / target_stride_index)
        llava_features = {k: torch.stack(v) for k, v in llava_features.items()}
        for k, v in llava_features.items():
            D_llava, dtype = v.shape[-1], v.dtype
            v = v[None, None].float()
            v = F.interpolate(v, size=(T_target, D_llava), mode='nearest')
            # v = F.interpolate(v, size=(T_vfeat, D_llava), mode='nearest')
            llava_features[k] = v.squeeze([0, 1]).to(dtype=dtype)  # [T_vfeat, D_llava]
        return llava_features


class QADataset(BaseDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len, qa_type, CloseQA_weight=50):
        super().__init__(data_dir, split, feature_type, max_v_len)
        self.qa_type = qa_type  # CloseQA, OpenQA, Mixed
        self.choice_indices = ['A', 'B', 'C', 'D']
        self.CloseQA_weight = CloseQA_weight
        self.openqa_weight = 100 - CloseQA_weight

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer'].strip()

        qa_type = self.qa_type
        if qa_type == 'Mixed':  # randomly choose a qa type
            qa_type = random.choices(['CloseQA', 'OpenQA'], weights=[self.CloseQA_weight, self.openqa_weight], k=1)[0]
        if qa_type == 'OpenQA':
            question_str = f"question: {question} video: "
            answer_str = answer
        elif qa_type == 'CloseQA':
            wrong_answers = self.annotations[index]['wrong_answers']
            # shuffle choices
            choices = [answer] + wrong_answers
            random.shuffle(choices)
            answer_index = choices.index(answer)
            choices = [f'({self.choice_indices[idx]}) {choices[idx]}' for idx in range(len(choices))]  # ["(A) xx", "(B) xx", "(C) xx", "(D) xx"]
            choices_str = ' '.join(choices)  # (A) xx (B) xx (C) xx (D) xx
            question_str = f"question: {question} choices: {choices_str}. video: "
            answer_str = choices[answer_index]  # (A/B/C/D) xx
        else:
            raise NotImplementedError

        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        start_frame = self.annotations[index].get('moment_start_frame')
        end_frame = self.annotations[index].get('moment_end_frame')
        start_time = start_frame / 30
        end_time = end_frame / 30

        if 'video_start_sec' not in self.annotations[index]:  # LLM generated QA
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)

        return {
            'video_id': video_id,
            'question': question_str,
            'answer': answer_str,
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': qa_type
        }


class JointDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset], tokenizer_path) -> None:
        super().__init__(datasets)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir='./cache_dir')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # BUG: Set this per convenience for GPT-2

    def collate_fn(self, batch):
        question = [b['question'] for b in batch]
        question_tok = self.tokenizer(question, padding=True, return_tensors='pt', add_special_tokens=False)

        answer = [b['answer'] for b in batch]
        labels = self.tokenizer(answer, padding=True, return_tensors='pt').input_ids
        # NOTE: NLQ data does not have an answer
        for idx, a in enumerate(answer):
            if a == 'None':
                labels[idx] = torch.ones_like(labels[idx]) * -100

        video_feature = [b['v_feat'] for b in batch]
        video_feature_padded = pad_sequence(video_feature, batch_first=True)
        video_mask = pad_sequence([torch.ones(len(v)) for v in video_feature], batch_first=True).bool()

        result = {
            'video_id': [b['video_id'] for b in batch],
            'q_text': question,
            'q_token': question_tok.input_ids,
            'q_mask': question_tok.attention_mask.bool(),
            'v_feat': video_feature_padded,
            'v_mask': video_mask,
            'v_len': np.asarray([b['v_len'] for b in batch], dtype=np.long),
            'gt_segments': torch.stack([b['segments'] for b in batch]),
            'gt_labels': torch.stack([b['one_hot_labels'] for b in batch]),
            'query_id': [b['query_id'] for b in batch],
            'sample_ratio': [b['sample_ratio'] for b in batch],
            'a_text': answer,
            'labels': labels,
            'task': [b['task'] for b in batch],
        }
        if 'llava_feat' in batch[0]:
            result['llava_feat'] = default_collate([b['llava_feat'] for b in batch])
            # print(result['llava_feat']['global'].shape)

        return result


class JointDataModule(pl.LightningDataModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        ds_kws = {}
        if self.config.get('additional_feature_type', '') == 'llava':
            DatasetClassForNLQ = NLQDatasetOnLLaVA
            ds_kws |= self.config.get('llava', {})
        else:
            DatasetClassForNLQ = NLQDataset

        CloseQA_weight = self.config.get('closeqa_weight', 50)
        print(f'CloseQA percentage: {CloseQA_weight}%')

        self.train_dataset = JointDataset(
            [
                QADataset('data/unified', train_split, self.config.feature_type, self.config.max_v_len, 'Mixed', CloseQA_weight)
                for train_split in self.config.qa_train_splits]
            + [
                DatasetClassForNLQ('data/unified', train_split, self.config.feature_type, self.config.max_v_len, **ds_kws)
                for train_split in self.config.nlq_train_splits],
            self.config.tokenizer_path
        )

        test_datasets = []
        for split in self.config.test_splits:
            if split == 'QaEgo4D_test':
                test_datasets.append(QADataset('data/unified', split, self.config.feature_type, self.config.max_v_len, 'OpenQA'))
            elif split == 'QaEgo4D_test_close':
                test_datasets.append(QADataset('data/unified', split, self.config.feature_type, self.config.max_v_len, 'CloseQA'))
            elif split in ['NLQ_val', 'NLQ_test_unannotated']:
                test_datasets.append(DatasetClassForNLQ('data/unified', split, self.config.feature_type, self.config.max_v_len, **ds_kws))
            elif split in ['NLQ_train']:  # for debug
                test_datasets.append(DatasetClassForNLQ('data/unified', split, self.config.feature_type, self.config.max_v_len, **ds_kws))
            else:
                print(split)
                raise NotImplementedError
        self.val_dataset = self.test_dataset = JointDataset(test_datasets, self.config.tokenizer_path)

        print(f'#total train: {len(self.train_dataset)}')
        print(f'#total val: {len(self.val_dataset)}')
        print(f'#total test: {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
