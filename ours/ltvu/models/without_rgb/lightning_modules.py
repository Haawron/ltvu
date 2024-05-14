import logging
import json
import yaml
import os
import math
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import fire

import torch
import numpy as np
import pickle
from einops import rearrange

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import torch.distributed

from .language_models import TEXTONLYNLQMODELS, TextOnlyNLQModel, TextOnlyNLQModel2
from ltvu.data_loader.egonlq import EgoNLQRawDataModule
from ltvu.utils import evaluate_topk_recalls, evaluate_topk_span_recalls
from ltvu.utils.callbacks import JSONPredictionWriter


DEFAULT_RESULTS_DIR = 'results/without_rgb/mips-trained'


class TextOnlyNLQLitModule(L.LightningModule):
    def __init__(
        self,
        language_model_name,
        model_version: int = -1,
        max_ctx_len: int = 256,
        head_name: str = 'af',
        lr = 1e-5,
        recall_ks = [1, 5, 10, 15, 20],
        **model_kws
    ):
        super().__init__()
        self.language_model_name = language_model_name
        self.model: TextOnlyNLQModel2 = TEXTONLYNLQMODELS[model_version](
            model_name=language_model_name,
            max_ctx_len=max_ctx_len,
            head_name=head_name,
            **model_kws
        )
        self.lr = lr
        self.recall_ks = recall_ks

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

    def training_step(self, batch, batch_idx):
        output = self.model.forward(batch)
        loss = output['loss']
        batch_size = len(list(batch.values())[0])
        self.log(
            'loss', loss.detach(), batch_size=batch_size, sync_dist=True, logger=True, on_step=True, on_epoch=True, prog_bar=True, rank_zero_only=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(list(batch.values())[0])
        output_dict = self.model.forward(batch)
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
        output_dict = self.model.forward(batch)
        scores_dict, minor_results = evaluate_topk_recalls(
            preds=np.stack([sample['segments'] for sample in output_dict['preds']]),  # [B, max k, 2]
            gt_segment=batch['gt_segment_sec'].cpu().numpy(),  # [B, 2]
            ks=self.recall_ks,
        )
        json_records = []
        for pred_segments, (clip_uid, query_id, query, clip_start_sec, clip_end_sec) in zip(
            minor_results['preds'].tolist(), zip(
                batch['clip_uid'], batch['query_id'],
                batch['query'],
                batch['gt_start_sec'].tolist(), batch['gt_end_sec'].tolist(),
            )
        ):
            record = {
                'clip_uid': clip_uid,
                'query_id': query_id,
                'pred_window': pred_segments,  # 5 x 2
                'gt_window': {
                    'clip_start_sec': clip_start_sec,
                    'clip_end_sec': clip_end_sec,
                    'query': query,
                }
            }
            # assert serializable
            json_records.append(record)
        if batch_idx == 0 and self.trainer.is_global_zero:
            print('\n\n==== Example JSON Records ====\n')
            pprint(json_records[:1])
            print('\n==============================\n')
        self.log_dict(
            scores_dict, batch_size=batch_size, sync_dist=True,
            on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=False)
        return json_records

    def configure_optimizers(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        optim = torch.optim.AdamW(params, lr=self.lr)
        return optim

    def on_validation_epoch_end(self):
        """
        Gather the validation_step_outputs across GPUs and save to a pickle file.
        """
        outputs = {}
        for k, v in list(self.validation_step_outputs.items()):
            tensor = torch.from_numpy(np.stack(v)).to(self.device)  # [N_B_g, B, ...]
            num_batches = torch.tensor([tensor.shape[0]], device=self.device)  # N_B_g
            all_num_batches = self.trainer.strategy.all_gather(num_batches)  # [G]
            max_num_batches = max(all_num_batches)  # N_B = max_g N_B_g
            if num_batches < max_num_batches:
                padding = torch.zeros(
                    max_num_batches-num_batches, *tensor.shape[1:],
                    device=self.device)
                tensor = torch.cat([tensor, padding], dim=0)
            tensor_gathered = self.trainer.strategy.all_gather(tensor)  # [G, N_B, B, ...]
            tensor_gathered = torch.cat(
                [t[:nb] for t, nb in zip(tensor_gathered, all_num_batches)])  # [ΣN_B_g, B, ...]
            tensor_gathered = rearrange(tensor_gathered,
                'bl b ... -> (bl b) ...').cpu().numpy()  # [N, ...]
            outputs[k] = tensor_gathered
        if not self.trainer.is_global_zero:
            return
        p_pkl = self.p_logdir / f'{self.language_model_name.replace("/", "--")}.pkl'
        pickle.dump(outputs, p_pkl.open('wb'))
        self.validation_step_outputs = defaultdict(list)

    def on_before_optimizer_step(self, optimizer) -> None:
        unintended_no_grad_captured = False
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f'{n} [{p.shape}] requires grad but got None')
                unintended_no_grad_captured = True
        if unintended_no_grad_captured:
            raise ValueError("No gradients")


def get_trainer(*, exp_name, p_log_dir=None):
    ngpus = torch.cuda.device_count()

    is_in_a_batch_job: bool = ('batch' in os.environ.get('SLURM_JOB_PARTITION').lower())

    if p_log_dir is not None:  # test or predict
        loggers = None
        callbacks = [JSONPredictionWriter(p_json=p_log_dir / f'predictions.json')]
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
        callbacks = [
            ModelCheckpoint(
                monitor=(metric:='iogt>=0.3 R@05'),
                mode='max',
                filename=f'{{epoch}}-{{{metric}:.4f}}'),
            JSONPredictionWriter(
                p_json=Path(l.log_dir) / f'predictions.json')
        ]

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
        accumulate_grad_batches=math.ceil(32/ngpus),
        precision='bf16-mixed',
        max_epochs=10,
        # reload_dataloaders_every_n_epochs=1,

        logger=loggers,

        strategy="auto",
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=not is_in_a_batch_job,
        num_sanity_val_steps=0 if is_in_a_batch_job else 2,
        enable_model_summary=True,
    )

    if trainer.is_global_zero:
        print('\n\n')
        print('Logdir: ', p_log_dir if p_log_dir is not None else trainer.logger.log_dir)
        print('\n\n')

    return trainer


def exp_train(
    model_name: str = 'google/flan-t5-base',
    head_name = 'af',
    batch_size: int = 2,
    lr: float = 2e-4,
    proposal_mode: bool = True,
    proposal_width_sec: float = 30.,
    exp_name: None|str = None,
    train_backbone: bool = False,
    ckpt_path: str = '',
):
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        p_log_dir = ckpt_path.parent.parent
        trainer = get_trainer(exp_name=None, p_log_dir=p_log_dir)
        p_yaml = p_log_dir / 'hparams.yaml'
        hparams = yaml.safe_load(p_yaml.open())
        for k, v in hparams.items():
            globals()[k] = v
    else:
        if 'RANK' not in os.environ or os.environ['RANK'] == '0':
            pprint(locals())  # args of this function
            print()
        exp_name = exp_name or model_name.replace('/', '--')
        trainer = get_trainer(exp_name=exp_name)

    caption_stride_sec = 2
    plm = TextOnlyNLQLitModule(
        language_model_name=model_name,
        head_name=head_name,
        max_ctx_len=256,
        recall_ks=[1, 5],
        lr=lr,
        train_backbone=train_backbone,
        caption_stride_sec=caption_stride_sec,
    )
    dm = EgoNLQRawDataModule(
        batch_size=batch_size,
        num_workers=16, prefetch_factor=64//batch_size,
        tokenizer=plm.tokenizer,
        proposal_mode=proposal_mode,
        proposal_width_sec=proposal_width_sec,
        caption_stride_sec=caption_stride_sec,
    )

    if ckpt_path:
        trainer.test(plm, datamodule=dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(plm, datamodule=dm)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(plm, datamodule=dm, ckpt_path=ckpt_path)
        if trainer.global_rank == 0:
            print(ckpt_path)
            print()


if __name__ == '__main__':
    # setup
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    fire.Fire(exp_train)
