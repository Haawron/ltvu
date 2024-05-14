import logging
import itertools
import json
from typing import Literal
import re
import os
import pickle
from collections import defaultdict
from pathlib import Path
from io import StringIO
import fire

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from einops import rearrange, repeat
from sentence_transformers import SentenceTransformer, util as sbert_util

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
    LearningRateMonitor,
    Callback,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from ltvu.models.heads.nlq_head import NLQHead, ctr_diou_loss_1d, sigmoid_focal_loss
from ltvu.models.egovlpv1.model import TextOnlyFrozenInTime
from ltvu.data_loader.egonlq import EgoNLQRawDataModule

from ltvu.utils import *


class UnifiedTextEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        train_backbone: bool = False,
        head: Literal['', 'linear', 'nlq_head'] = '',
        device = 'cuda'
    ):
        super().__init__()
        self.train_backbone = train_backbone
        self.lm, self.tokenizer = self.get_lm(model_name, device)
        self.is_sbert = isinstance(self.lm, SentenceTransformer)
        self.dim = self.lm.get_sentence_embedding_dimension()
        if head == 'linear':
            self.head_caption = nn.Linear(self.dim, self.dim)
            self.head_query = nn.Linear(self.dim, self.dim)
        elif head == 'nlq_head':
            self.head = NLQHead(self.dim, max_v_len=256)
        else:
            self.head_caption = nn.Identity()
            self.head_query = nn.Identity()

    def get_lm(self, model_name, device):
        if model_name.lower() == 'egovlp':
            class TextOnlyFrozenInTimeWrapper(TextOnlyFrozenInTime):
                def forward(self, *args, **kwargs):
                    z = super().forward(*args, **kwargs)
                    return z / z.norm(dim=-1, keepdim=True)
                def get_sentence_embedding_dimension(self):
                    return self.model.projection_dim
            lm = TextOnlyFrozenInTimeWrapper(device=device)
            tokenizer = lm.tokenizer
        else:
            class SentenceTransformerWrapper(SentenceTransformer):
                def forward(self, *args, **kwargs):
                    return super().forward(*args, **kwargs)['sentence_embedding']
            lm = SentenceTransformerWrapper(model_name, device=device)
            # tokenizer = lm.tokenizer

        if not self.train_backbone:
            lm.eval()
            for p in lm.parameters():
                p.requires_grad = False
        return lm, tokenizer

    def compute_total_loss(
        self,
        logits,
        starts,
        ends,
        gt_starts,
        gt_ends,
        width_sec = 30.
    ):
        """
        Args:
            logits: B x [T_i]
            starts, ends: B x [T_i]
            gt_starts, gt_ends: B

        Returns:
            loss_dict: dict[str, torch.Tensor]
        """
        losses_focal = []
        # losses_diou = []

        # TODO: parallelize by applying padding
        for logit, start, end, gt_s, gt_e in zip(logits, starts, ends, gt_starts, gt_ends):
            start, end = start - width_sec / 2, end + width_sec / 2
            in_gt_mask = ((gt_s <= start) & (start <= gt_e)) | ((gt_s <= end) & (end <= gt_e))
            losses_focal.append(self.compute_focal_loss(logit, in_gt_mask))
        loss_focal = torch.stack(losses_focal).mean()
        return {
            'loss_final': loss_focal,  # + 2 * loss_diou,
            'loss_focal': loss_focal,
        }

    def compute_focal_loss(self, logits, gts):
        # logits: [B, T_i]
        # gts: [B, T_i]
        return sigmoid_focal_loss(logits, gts).mean()

    def compute_diou_loss(self, preds, gts):
        # preds: [B, 2]
        # gts: [B, 2]
        return ctr_diou_loss_1d(preds, gts).mean()

    def forward_encoder(self, batch) -> tuple[list[torch.Tensor], torch.Tensor]:
        # caps_bds = batch['captions_bound']
        caps_ls = batch['captions_length']
        z_query = self.lm.forward(batch['query_tokens'])  # [B, d]
        z_query = self.head_query(z_query)  # [B, d]
        z_caps_ = self.lm.forward(batch['captions_tokens'])  # [Σ T_i, d]
        z_caps_ = self.head_caption(z_caps_)
        z_caps: tuple[torch.Tensor] = z_caps_.split(caps_ls)  # B x [T_i, d]
        # z_caps = [z_caps_[bi:bj] for bi, bj in zip([0] + caps_bds[:-1], caps_bds)]  # B x [T_i, d]
        return z_caps, z_query

    def forward(self, batch):
        # encoder LM
        z_caps, z_query = self.forward_encoder(batch)  # B x [T_i, d], [B, d]

        # head + loss
        caps_starts, caps_ends = batch['caption_start_secs'], batch['caption_end_secs']
        gt_starts, gt_ends = batch['gt_start_sec'], batch['gt_end_sec']
        sims, args = [], []
        for z_cap, z_q in zip(z_caps, z_query.unsqueeze(1)):
            sim = sbert_util.dot_score(z_cap, z_q).squeeze(1)  # [T_i]
            arg = sim.detach().argsort(descending=True)
            sims.append(sim)
            args.append(arg)

        loss_dict = self.compute_total_loss(sims, caps_starts, caps_ends, gt_starts, gt_ends)

        return {
            'sims': sims,
            'args': args,
            'loss_dict': loss_dict,
        }

    # TODO: preds 얻을 때까지, eval(preds, gts) 쪼개기
    @staticmethod
    @torch.no_grad()
    def evaluate_topk_span_recalls(
        sims: list[torch.Tensor],
        caps_starts: list[torch.Tensor],
        caps_ends: list[torch.Tensor],
        gt_start_secs: torch.Tensor,
        gt_end_secs: torch.Tensor,
        dfs_caps: list[pd.DataFrame],
        width_sec: float = 30.,
        ks: list[int] = [1, 5],
        recall_threses: list[float] = [.3, .5],
    ):
        preds = []
        max_k = max(ks)
        for sim, caps_start, caps_end in zip(sims, caps_starts, caps_ends):
            caps_c = (caps_start + caps_end) / 2  # [T_i]
            pred_c, _ = nms_1d_centers(
                caps_c.cpu().numpy(),
                sim.float().cpu().numpy(),
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
        results = {}
        for k in ks:
            for thres in recall_threses:
                results[f'iogt>={thres} R@{k:02d}'] = (iogts[:, :k] > thres).any(axis=1).mean()
                results[f'iou>={thres} R@{k:02d}'] = (ious[:, :k] > thres).any(axis=1).mean()
        return results, {
            'iogts': iogts,
            'ious': ious,
            'preds': preds,
            'gts': gts[:, 0],
        }


class UnifiedTextEncoderLitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        head: Literal['', 'linear', 'nlq_head'] = '',
        train_backbone: bool = False,
        recall_ks: list[int] = np.arange(1, 21),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = UnifiedTextEncoder(
            model_name, train_backbone=train_backbone, head=head, device=None)
        self.recall_ks = recall_ks
        self.validation_step_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)

        p_logdir = Path(self.trainer.log_dir if self._trainer is not None else 'results/without_rgb/mips/tmp')
        if not p_logdir.exists():
            p_logdir.mkdir(exist_ok=True, parents=True)
        self.p_logdir = p_logdir
        self.save_hyperparameters()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def training_step(self, batch, batch_idx):
        output = self.model.forward(batch)
        loss_dict = output['loss_dict']
        batch_size = batch['query_tokens']['input_ids'].shape[0]
        self.log(
            'loss', loss_dict['loss_final'].detach(), batch_size=batch_size, sync_dist=True, logger=False,
            on_step=True, on_epoch=True, prog_bar=True, rank_zero_only=False)
        return loss_dict['loss_final']

    def validation_step(self, batch, batch_idx):
        output = self.model.forward(batch)
        sims = output['sims']
        dfs_caps = batch['captions']
        eval_results, minor_results = self.model.evaluate_topk_span_recalls(
            sims,
            batch['caption_start_secs'], batch['caption_end_secs'],
            batch['gt_start_sec'], batch['gt_end_sec'],
            dfs_caps = dfs_caps,
            ks = self.recall_ks,
        )
        for k, v in minor_results.items():
            self.validation_step_outputs[k].append(v)  # maybe [B, max k]
        batch_size = batch['query_tokens']['input_ids'].shape[0]
        self.log_dict(
            eval_results, batch_size=batch_size, sync_dist=True,
            on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=False)

    test_step = validation_step

    def configure_optimizers(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        optim = torch.optim.AdamW(params, lr=1e-3)
        return optim

    def on_validation_epoch_end(self):
        """
        Gather the validation_step_outputs across GPUs and save to a pickle file.
        """
        # epoch = self.trainer.current_epoch
        outputs = {}
        for k, v in list(self.validation_step_outputs.items()):
            tensor = torch.from_numpy(np.stack(v)).to(self.device)  # [N_B_g, B, ...]
            num_batches = torch.tensor([tensor.shape[0]], device=self.device)  # N_B_g
            all_num_batches = self.trainer.strategy.all_gather(num_batches)  # [G]
            max_num_batches = max(all_num_batches)  # N_B = max_g N_B_g
            if num_batches < max_num_batches:
                padding = torch.zeros(max_num_batches-num_batches, *tensor.shape[1:], device=self.device)
                tensor = torch.cat([tensor, padding], dim=0)
            tensor_gathered = self.trainer.strategy.all_gather(tensor)  # [G, N_B, B, ...]
            tensor_gathered = torch.cat([t[:nb] for t, nb in zip(tensor_gathered, all_num_batches)])  # [ΣN_B_g, B, ...]
            tensor_gathered = rearrange(tensor_gathered, 'bl b ... -> (bl b) ...').cpu().numpy()  # [N, ...]
            outputs[k] = tensor_gathered
        if not self.trainer.is_global_zero:
            return
        p_pkl = self.p_logdir / f'{self.model_name}.pkl'
        pickle.dump(outputs, p_pkl.open('wb'))
        self.validation_step_outputs = defaultdict(list)


# def exp1():
#     L.seed_everything(42, workers=True)

#     device = 'cuda'
#     lm = UnifiedTextEncoder('multi-qa-mpnet-base-dot-v1', device)
#     ds = EgoNLQRawDataset(split='val')
#     dl = torch.utils.data.DataLoader(
#         ds, batch_size=1, shuffle=False, pin_memory=True,
#         drop_last=False, collate_fn=lambda x: x,
#         num_workers=8, prefetch_factor=8, persistent_workers=True,
#     )

#     print('\n\n')
#     p_results_dir = Path('results/without_rgb/mips')
#     p_results_dir.mkdir(exist_ok=True, parents=True)
#     from itertools import product
#     for w, k in product(
#         [10., 30., 60., 90.], [5] #[5, 10, 15]
#     ):
#         records = []
#         for batch in tqdm(dl):
#             for sample in batch:
#                 record = evaluate_entity_proposals(sample, lm, width_sec=w)
#                 records.append(record)
#         records = pd.DataFrame(records)
#         records['iogt_recall0.3'] = (records['iogt'] >= .3).mean()
#         records['iogt_recall0.5'] = (records['iogt'] >= .5).mean()
#         print(records.describe())
#         p_result_csv = p_results_dir / f'w{w}_k{k}.csv'
#         records.to_csv(p_result_csv, index=False)
#         print()


def test1():
    from pprint import pprint
    model = UnifiedTextEncoder('multi-qa-mpnet-base-dot-v1', device='cuda')
    dm = EgoNLQRawDataModule(
        batch_size=4,
        tokenizer=model.tokenizer)
    dm.setup()
    for batch in dm.train_dataloader():
        pprint(batch)
        pprint(batch['captions_tokens']['input_ids'].shape)
        pprint(batch['captions_tokens']['attention_mask'].shape)
        pprint(batch['captions_tokens']['attention_mask'])
        pprint(batch['captions_bound'])
        output = model.forward(batch)
        pprint(output)
        break


def test_nvidia_smi():
    print(get_gpu_stats())


def test_losses():
    bsz = 4
    inputs: torch.Tensor = (torch.randn(bsz, 2) ** 2).sort(dim=-1).values
    targets: torch.Tensor = (torch.randn(bsz, 2) ** 2).sort(dim=-1).values
    loss = ctr_diou_loss_1d(inputs, targets).mean()
    loss2 = sigmoid_focal_loss(inputs, targets).mean()
    print(inputs)
    print(targets)
    print()
    print(loss)
    print(loss2)


def exp_sim_topk_span_recalls(model_name='multi-qa-mpnet-base-dot-v1'):
    L.seed_everything(42, workers=True)
    trainer = L.Trainer(precision='bf16-mixed', strategy="ddp")
    torch.set_float32_matmul_precision('high')
    plm = UnifiedTextEncoderLitModule(model_name)
    dm = EgoNLQRawDataModule(batch_size=64, tokenizer=plm.tokenizer)
    trainer.validate(plm, datamodule=dm)


def exp_train(
    model_name='multi-qa-mpnet-base-dot-v1', exp_name=None,
    head='nlq_head', train_backbone=False,
    gather_consecutive_duplicated_captions=True,
):
    L.seed_everything(42, workers=True)
    if head == 'nlq_head':
        if gather_consecutive_duplicated_captions:
            # log warning
            logging.warning(
                'Gathering consecutive duplicated captions is not supported with NLQHead. '
                'Setting gather_consecutive_duplicated_captions=False.'
            )
        gather_consecutive_duplicated_captions = False
    exp_name = exp_name or model_name
    common_logger_options = dict(save_dir='results/without_rgb/mips/', name=exp_name)
    trainer = L.Trainer(
        # fast_dev_run=10, detect_anomaly=True,
        limit_train_batches=10,
        gradient_clip_val=1.,
        accumulate_grad_batches=2,
        precision='bf16-mixed',
        max_epochs=50,
        # reload_dataloaders_every_n_epochs=1,

        logger=[
            TensorBoardLogger(
                sub_dir='tb/',  # TensorBoard logs will be saved in /save_dir/name/version/sub_dir
                **common_logger_options),
            CSVLogger(**common_logger_options),
        ],

        # strategy="ddp",
        deterministic=True,
        callbacks=[
            # EarlyStopping(monitor='val_loss'),
            ckpt_callback:=ModelCheckpoint(
                # dirpath='checkpoints/',
                monitor='iou>=0.3 R@05', mode='max',
            ),
            # LearningRateMonitor("epoch"),
        ],
    )

    if trainer.is_global_zero:
        print('\n\n')
        print('Logdir: ', trainer.logger.log_dir)
        print('\n\n')
    torch.set_float32_matmul_precision('high')
    plm = UnifiedTextEncoderLitModule(
        model_name, head=head, recall_ks=[1, 5, 10, 15, 20], train_backbone=train_backbone)
    dm = EgoNLQRawDataModule(
        batch_size=128, tokenizer=plm.tokenizer,
    )

    trainer.fit(plm, datamodule=dm)
    trainer.test(plm, datamodule=dm)

    print(ckpt_callback.best_model_path)

# Use cases
# ./run.sh exp_train multi-qa-mpnet-base-dot-v1
if __name__ == '__main__':
    fire.Fire()
