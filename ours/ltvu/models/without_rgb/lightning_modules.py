import logging
import json
import yaml
import os
import re
import math
import inspect
from typing import Literal
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import fire

import torch
import torch.distributed
import numpy as np
import pickle
from einops import rearrange

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import torch.distributed

from .language_models import TEXTONLYNLQMODELS, TextOnlyNLQModelBase
from ltvu.data_loader.egonlq import EgoNLQRawDataModule
from ltvu.utils import evaluate_topk_recalls, evaluate_topk_span_recalls
from ltvu.utils.callbacks import JSONPredictionWriter


DEFAULT_RESULTS_DIR = 'results/without_rgb/mips-trained'


class TextOnlyNLQLitModule(L.LightningModule):
    def __init__(
        self,
        language_model_name,
        model_version: int = 1,
        max_ctx_len: int = 256,
        head_name: str = 'af',
        lr = 1e-5,
        proposal_mode = None,
        proposal_width_sec = None,

        recall_ks = [1, 5, 10, 15, 20],
        model_kws = dict(),
        lm_kws = dict(),
        head_kws = dict(),
    ) -> None:
        super().__init__()
        self.language_model_name = language_model_name
        self.model: TextOnlyNLQModelBase = TEXTONLYNLQMODELS[model_version](
            model_name=language_model_name,
            max_ctx_len=max_ctx_len,
            head_name=head_name,
            lm_kws=lm_kws,
            head_kws=head_kws,
            **model_kws
        )
        self.lr = lr
        self.recall_ks = recall_ks
        if proposal_mode:
            assert proposal_width_sec is not None
        self.proposal_mode = proposal_mode
        self.proposal_width_sec = proposal_width_sec

        p_logdir = Path(
            self.trainer.log_dir
            if self._trainer is not None
            else f'{DEFAULT_RESULTS_DIR}/tmp')
        if not p_logdir.exists():
            p_logdir.mkdir(exist_ok=True, parents=True)
        self.p_logdir = p_logdir
        self.save_hyperparameters()

        self.validation_step_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def forward_model(self, batch):
        output_dict = self.model.forward(batch)
        if not self.training and self.proposal_mode and 'preds' in output_dict:
            preds = output_dict['preds']
            for i in range(len(preds)):
                segs_sec = preds[i]['segments']
                cs = segs_sec.mean(dim=1)  # [max k]
                w = self.proposal_width_sec
                output_dict['preds'][i]['segments'] = torch.stack([cs-w/2, cs+w/2], dim=1)
        return output_dict

    def training_step(self, batch, batch_idx):
        output_dict = self.forward_model(batch)
        loss = output_dict['loss']
        batch_size = len(list(batch.values())[0])
        self.log(
            'loss', loss.detach(), batch_size=batch_size,
            sync_dist=True, logger=True, on_step=True, on_epoch=True, prog_bar=True, rank_zero_only=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(list(batch.values())[0])
        output_dict = self.forward_model(batch)
        scores_dict, minor_results = evaluate_topk_recalls(
            preds=np.stack([sample['segments'] for sample in output_dict['preds']]),  # [B, max k, 2]
            gt_segment=batch['gt_segment_sec'].cpu().numpy(),  # [B, 2]
            ks=self.recall_ks,
        )
        for k, v in minor_results.items():
            self.validation_step_outputs[k].append(v)  # maybe [B, max k]
        self.log_dict(
            scores_dict, batch_size=batch_size, sync_dist=True,
            on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=False)

    def test_step(self, batch, batch_idx):
        batch_size = len(list(batch.values())[0])
        output_dict = self.forward_model(batch)
        scores_dict, minor_results = evaluate_topk_recalls(
            preds=np.stack([sample['segments'] for sample in output_dict['preds']]),  # [B, max k, 2]
            gt_segment=batch['gt_segment_sec'].cpu().numpy(),  # [B, 2]
            ks=self.recall_ks,
        )
        minor_results['preds'] =  minor_results['preds'].tolist()
        output_dict['logits'] = output_dict['logits'].tolist()
        json_records = []
        for bid in range(batch_size):
            record = {
                'clip_uid': batch['clip_uid'][bid],
                'query_id': batch['query_id'][bid],
                'pred_window': minor_results['preds'][bid],  # 5 x 2
                'logits': output_dict['logits'][bid],
                'gt_window': {
                    'clip_start_sec': batch['gt_segment_sec'][bid][0].item(),
                    'clip_end_sec': batch['gt_segment_sec'][bid][1].item(),
                    'query': batch['query'][bid],
                }
            }
            try:
                json.dumps(record)  # assert serializable
            except TypeError as e:
                logging.error(f'Error in serializing record: {record}')
                raise e
            json_records.append(record)
        if batch_idx == 0 and self.trainer.is_global_zero:
            print('\n\n================ Example JSON Records ================\n')
            pprint(json_records[:1], compact=True)
            print(  '\n======================================================\n')
        scores_dict = {
            k: v if isinstance(v, float|np.floating) else v
            for k, v in scores_dict.items()}
        self.log_dict(
            scores_dict, batch_size=batch_size, sync_dist=True,
            on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=False)
        return json_records

    def configure_optimizers(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        optim = torch.optim.AdamW(params, lr=self.lr)
        return optim

    # TODO: move custom callbacks
    def on_before_optimizer_step(self, optimizer) -> None:
        unintended_no_grad_captured = False
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f'{n} [{p.shape}] requires grad but got None')
                unintended_no_grad_captured = True
        if unintended_no_grad_captured:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise ValueError("No gradients")

    def on_load_checkpoint(self, checkpoint):
        def key_mapper(mappings: list[tuple[re.Pattern, str]], key: str):
            """Mapper function for key renaming.

            Args:
                mappings (list[tuple[re.Pattern, str]]): list of (pattern, replacement) pairs
                key (str): The key to be renamed
            """
            mapped = key
            for p, replacement in mappings:
                if p.match(key):
                    mapped = p.sub(replacement, mapped)
            if (torch.distributed.is_initialized()
                and self.trainer.global_rank == 0
                and mapped != key
            ):
                print(f'{key:70s} -> {mapped}')
            return mapped

        remove_keys = list(map(re.compile, [
            '^model.head.cross_encoder.embeddings.word_embeddings.weight'.replace('.', r'\.'),
            r'^model\.lm\.m\..*',
            r'^model\.lm\.encoder\..*',
            r'^model\.lm\.p\..*',
        ]))
        for k in checkpoint['state_dict'].keys():
            if any(p.match(k) for p in remove_keys):
                if torch.distributed.is_initialized():
                    if self.trainer.global_rank != 0:
                        continue
                print(f'Removing {k}')
        change_keys = [
            # Changing head arch from a normal bert to a wrapper
            (re.compile(r'^model\.head\.cross_encoder\.(?!bert|linear)'), 'model.head.cross_encoder.bert.'),
        ]
        checkpoint['state_dict'] = {
            key_mapper(change_keys, k): v
            for k, v in checkpoint['state_dict'].items()
            if all(not p.match(k) for p in remove_keys)
        }
        super().on_load_checkpoint(checkpoint)


def get_trainer(*, exp_name, batch_size, p_log_dir=None, split='val'):
    ngpus = torch.cuda.device_count()

    is_in_a_batch_job: bool = ('batch' in os.environ.get('SLURM_JOB_PARTITION').lower())
    filename = f'predictions-{split}.json'

    callbacks = [
        ModelSummary(max_depth=3),
    ]
    if p_log_dir is not None:  # test or predict
        loggers = None
        callbacks += [JSONPredictionWriter(p_json=p_log_dir / filename)]
    else:
        common_logger_options = dict(
            save_dir=DEFAULT_RESULTS_DIR,
            name=exp_name)
        loggers = [
            l:=CSVLogger(**common_logger_options),
            TensorBoardLogger(
                sub_dir='tb/',  # TensorBoard logs will be saved in /save_dir/name/version/sub_dir
                version=l.version,
                **common_logger_options),
        ]
        callbacks += [
            ModelCheckpoint(
                monitor=(metric:='iogt>=0.3 R@01'),
                mode='max',
                filename=f'{{epoch}}-{{{metric}:.4f}}'),
            JSONPredictionWriter(
                p_json=Path(l.log_dir) / filename)
        ]

    callbacks.extend([

    ])

    batch_size_baseline = 16 * 8  # 16 per gpu * 8 gpus
    trainer = L.Trainer(
        ###### DEBUG OPTIONS ######
        # fast_dev_run=10,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=10,
        # detect_anomaly=True,
        # strategy="ddp_find_unused_parameters_true",
        ###########################

        gradient_clip_val=1.,
        accumulate_grad_batches=math.ceil(batch_size_baseline/batch_size/ngpus),
        precision='bf16-mixed',
        max_epochs=10,
        # reload_dataloaders_every_n_epochs=1,

        logger=loggers,

        strategy=DDPStrategy(gradient_as_bucket_view=True),
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=not is_in_a_batch_job,
        num_sanity_val_steps=0 if is_in_a_batch_job else 2,
        enable_model_summary=False,  # added in callback
    )

    if trainer.is_global_zero:
        print('\n\n')
        print('Logdir: ', p_log_dir if p_log_dir is not None else trainer.logger.log_dir)
        print('\n\n')

    return trainer


def exp_train(
    language_model_name: str = 'multi-qa-mpnet-base-dot-v1',  # 'google/flan-t5-base'
    model_version: int = 1,
    head_name = 'af',
    batch_size: int = 2,
    max_ctx_len: int = 256,
    lr: float = 2e-4,
    caption_stride_sec: int = 2,
    gather_consecutive_captions_factor: int = 1,
    proposal_mode: bool = True,
    proposal_width_sec: float = 30.,
    proposal_width_sec_train: float = 30.,
    exp_name: None|str = None,
    train_backbone: bool = False,
    model_kws: dict = dict(),
    lm_kws: dict = dict(),
    head_kws: dict = dict(),
    ds_kws: dict = dict(),
    dl_kws: dict = dict(),

    ckpt_path: str = '',
    prediction_split: Literal['train', 'val'] = 'val',
):
    if ckpt_path:
        # FIXME: 우선도가 이상함, [CLI > YAML > 디폴트] 여야 하는데 바뀜
        ckpt_path = Path(ckpt_path)
        p_log_dir = ckpt_path.parent.parent
        trainer = get_trainer(
            batch_size=batch_size,
            exp_name=None,
            p_log_dir=p_log_dir,
            split=prediction_split)
        p_yaml = p_log_dir / 'hparams.yaml'
        hparams = yaml.safe_load(p_yaml.open())
        language_model_name = hparams['language_model_name']
        lm_kws = hparams.get('lm_kws', {})
        head_kws = hparams.get('head_kws', {})
    else:
        exp_name = exp_name or language_model_name.replace('/', '--')
        trainer = get_trainer(
            batch_size=batch_size,
            exp_name=exp_name)

    if 'RANK' not in os.environ or os.environ['RANK'] == '0':
        pprint(locals())  # args of this function
        print()

    plm = TextOnlyNLQLitModule(
        language_model_name=language_model_name,
        model_version=model_version,
        head_name=head_name,
        max_ctx_len=max_ctx_len,
        recall_ks=[1, 5],
        lr=lr,
        proposal_mode=proposal_mode,
        proposal_width_sec=proposal_width_sec,
        model_kws=model_kws | dict(
            # FIXME: spaghetti config
            # model needs to pass this arg to head to later eval on test
            caption_stride_sec=gather_consecutive_captions_factor*caption_stride_sec),
        lm_kws=lm_kws,
        head_kws=head_kws,
    )
    pdm = EgoNLQRawDataModule(
        ds_kws=ds_kws | dict(
            caption_stride_sec=caption_stride_sec,
            gather_consecutive_captions_factor=gather_consecutive_captions_factor,
            tokenizer=plm.tokenizer,
            proposal_mode=proposal_mode,
            proposal_width_sec=proposal_width_sec,
            proposal_width_sec_train=proposal_width_sec_train,
        ),
        dl_kws=dl_kws | dict(
            batch_size=batch_size,
            num_workers=16, prefetch_factor=32//batch_size,
        ),
    )

    if ckpt_path:
        if prediction_split != 'val':
            pdm.test_dataloader = pdm.train_dataloader
        trainer.test(plm, datamodule=pdm, ckpt_path=ckpt_path)
    else:
        trainer.fit(plm, datamodule=pdm)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(plm, datamodule=pdm, ckpt_path=ckpt_path)
        if trainer.global_rank == 0:
            print(ckpt_path)
            print()


if __name__ == '__main__':
    # setup
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    fire.Fire(exp_train)
