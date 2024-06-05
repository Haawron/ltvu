import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss, TripletDistanceMetric
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        self.p_ann_jsons_dir = Path('/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/unified/')
        self.p_ann_json = self.p_ann_jsons_dir / f'annotations.NLQ_{split}.json'

        self.p_caps_dir = Path('/data/gunsbrother/prjs/ltvu/llms/LLaVA/results/egonlq/llava-v1.6-34b/global')
        self.all_clip_uids = list(set([p.stem for p in self.p_caps_dir.glob('*.json')]))
        self.ann_data = []
        for sample in json.load(self.p_ann_json.open()):
            if sample['video_id'] in self.all_clip_uids:
                clip_uid = sample['video_id']
                p_cap = self.p_caps_dir / f'{clip_uid}.json'
                num_caps = len(json.load(p_cap.open())['answers'])
                if abs(num_caps - sample['clip_duration'] * 30 / 300) < 5:
                    self.ann_data.append(sample)
        self.all_sample_ids = [q['sample_id'] for q in self.ann_data]
        self.caption_sample_factor = 4  # 4 -> query : caption = 1 : 3
        self.training = split == 'train'

    def __len__(self):
        if self.training:
            return self.caption_sample_factor * len(self.all_sample_ids)
        else:
            return len(self.all_sample_ids)

    def __getitem__(self, idx):
        FPS = 30
        mode, sample_idx = divmod(idx, len(self.all_sample_ids))
        sample = self.ann_data[sample_idx]
        clip_uid = sample['video_id']

        p_cap = self.p_caps_dir / f'{clip_uid}.json'
        cap_data = json.load(p_cap.open())['answers']
        caps = [entry[-1] for entry in cap_data]
        frame_idxs = [entry[0] for entry in cap_data]
        num_caps = len(caps)
        output = {
            'anchor': '',
            'positive': '',
            'negative': '',
        }

        if mode == 0:  # query / caption+ / caption-
            q = sample['question']
            s, e = sample['clip_start_sec'], sample['clip_end_sec']
            e_idx = min(1 + np.searchsorted(frame_idxs, e * FPS), num_caps-1)
            s_idx = min(np.searchsorted(frame_idxs, s * FPS), e_idx-1)
            # s_idx, e_idx = int(s * FPS / 300), min(math.ceil(e * FPS / 300), num_caps-1)
            caps_pos, caps_neg = [], []
            for cap_idx in range(num_caps):
                if s_idx <= cap_idx <= e_idx:
                    caps_pos.append(caps[cap_idx])
                else:
                    caps_neg.append(caps[cap_idx])
            output['anchor'] = q
            if self.training:
                output['positive'] = np.random.choice(caps_pos)
                if caps_neg:
                    output['negative'] = np.random.choice(caps_neg)
            else:
                output['positive'] = caps_pos[0]
                if caps_neg:
                    output['negative'] = caps_neg[0]

        elif mode > 0:  # caption / caption+ / caption-
            assert self.training
            anc_idx = np.random.randint(num_caps)
            pos_idx = np.clip(np.random.choice([anc_idx-1, anc_idx+1]), 0, num_caps-1)
            neg_idx = np.random.choice(list(set(range(num_caps)) - set([anc_idx, anc_idx+1, anc_idx-1])))
            output['anchor'] = caps[anc_idx]
            output['positive'] = caps[pos_idx]
            output['negative'] = caps[neg_idx]

        return output


def tofile(split='train', p_csvs_dir=Path('data/triplet-4')):
    from tqdm import trange
    dataset = TripletDataset(split)
    p_csvs_dir.mkdir(parents=True, exist_ok=True)
    p_csv = p_csvs_dir / f'{split}.csv'
    sep = ','
    with p_csv.open('w') as f:
        f.write(f'anchor{sep}positive{sep}negative\n')
        for i in trange(len(dataset)):
            sample = dataset[i]
            anchor = sample['anchor'].replace('"', '""')
            positive = sample['positive'].replace('"', '""')
            negative = sample['negative'].replace('"', '""')
            f.write(f'"{anchor}"{sep}"{positive}"{sep}"{negative}"\n')

    print(f"Saved {split} dataset to {p_csv}")


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # 3. Load a dataset to finetune on
    p_csvs_dir = Path('data/triplet-4')  # 4-moded
    # for split in ['train', 'val']:
    #     # p_csv = p_csvs_dir / f'{split}.csv'
    #     # if not p_csv.exists():
    #     tofile(split, p_csvs_dir)

    dataset = load_dataset("csv", 'triplet', data_files={
        'train': str(p_csvs_dir / 'train.csv'),
        'dev': str(p_csvs_dir / 'val.csv'),
    })
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]

    # 1. Load a model to finetune with 2. (Optional) model card data
    # model_name = "multi-qa-mpnet-base-dot-v1"
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name, local_files_only=True)
    print('Loaded model:', model_name)

    # 4. Define a loss function
    # loss = MultipleNegativesRankingLoss(model)
    loss = TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=.1)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"results/{model_name}",
        # overwrite_output_dir=True,
        # Optional training parameters:
        num_train_epochs=2,
        dataloader_num_workers=8,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1/2,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch,
        # Optional tracking/debugging parameters:
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        save_only_model=True,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_all-nli-dev_cosine_accuracy',
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    # dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 8. Save the trained model
    model.save_pretrained(f"models/{model_name}/final")

if __name__ == '__main__':
    main()
