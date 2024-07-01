from dotenv import load_dotenv
load_dotenv()

import datetime
import os
import time
from pathlib import Path
from typing import Any

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import lightning as L
import lightning.pytorch.utilities as L_utils
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint,
    BasePredictionWriter)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import (
    DDPStrategy,
)

from model.lightning_module import LitModule
from model.dataset import LitDataModule


@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


class FeatureWriter(BasePredictionWriter):
    def __init__(self, p_ckpt: Path, enable_progress_bar: bool = True):
        super().__init__('batch')
        self.p_ckpt = p_ckpt = Path(p_ckpt)
        self.p_output_dir = Path(*p_ckpt.parts[:p_ckpt.parts.index('lit')]) / 'features'
        self.p_output_dir.mkdir(parents=True, exist_ok=True)
        self.is_dist = torch.distributed.is_initialized()
        log_to_console(f'\n\nOutput dir: {self.p_output_dir}\n\n')
        self.enable_progress_bar = enable_progress_bar
        self.p_out_caps_dir = self.p_output_dir / 'captions'
        self.p_out_qs_dir = self.p_output_dir / 'queries'
        self.p_out_caps_dir.mkdir(parents=True, exist_ok=True)
        self.p_out_qs_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: dict[str, Any],  # for a single clip
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx
    ):
        # captions
        frame_idxs = batch['caption_frame_idxs']
        clip_uid = batch['clip_uid']
        z_cap = prediction['z_cap']
        assert len(frame_idxs) == z_cap.shape[0]
        p_out_caps = self.p_out_caps_dir / f'{clip_uid}.pt'
        caps_list = []  # N_cap x Tuple
        for frame_idx, z_cap_t in zip(frame_idxs, z_cap):
            caps_list.append((frame_idx, z_cap_t.cpu().squeeze()))
        torch.save(caps_list, p_out_caps)

        # queries
        sample_ids = batch['sample_ids']
        z_q = prediction['z_q']
        for z_q_i, sample_id in zip(z_q, sample_ids):
            z_q_i = z_q_i.cpu().squeeze()  # [D]
            p_out_query = self.p_out_qs_dir / f'{sample_id}.pt'
            torch.save(z_q_i, p_out_query)


@hydra.main(config_path='config', config_name='base', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.get('seed', 42), workers=True)
    torch.set_float32_matmul_precision('medium')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    # if __debug__:
    #     print('\n\nDebug mode is on\n\n')
    #     config.dataset.num_workers = 0
    #     config.dataset.prefetch_factor = None
    #     config.dataset.pin_memory = False
    #     config.dataset.persistent_workers = False

    enable_progress_bar = config.batch_flag != '1'

    if config.others.ckpt_path is not None:
        # setup config dynamically
        p_ckpt = Path(config.ckpt_path)
        p_hparams = p_ckpt.parent.parent / 'hparams.yaml'
        assert p_hparams.exists(), f'{p_hparams} not found'
        config.dataset.max_num_caps = 0
        config.dataset.max_cap_len = 512

        # if not trainer is initialized
        log_to_console(f'\n\n{__file__}:')
        log_to_console(f'\tsetting L_cap={config.dataset.max_cap_len} and N_cap=0 (not limited) for prediction\n\n')

        # setup lightning
        plm = LitModule.load_from_checkpoint(p_ckpt, hparams_file=p_hparams)
        pdm = LitDataModule(config)
        summ = ModelSummary(max_depth=3)
        summ.on_predict_start = summ.on_fit_start
        trainer = L.Trainer(
            **OmegaConf.to_container(config.trainer, resolve=True),
            enable_model_summary=False,
            default_root_dir=default_root_dir,
            logger=False,
            callbacks=[summ, FeatureWriter(p_ckpt=p_ckpt, enable_progress_bar=enable_progress_bar)],
            enable_progress_bar=enable_progress_bar)
        trainer.predict(plm, datamodule=pdm)

    else:
        plm = LitModule(config)
        pdm = LitDataModule(config)
        trainer = L.Trainer(
            **OmegaConf.to_container(config.trainer, resolve=True),
            strategy=DDPStrategy(timeout=datetime.timedelta(seconds=600)),
            enable_model_summary=False,
            default_root_dir=default_root_dir,
            logger=[
                TensorBoardLogger(
                    save_dir=default_root_dir,
                    version=os.environ.get("SLURM_JOB_ID"),
                    name="lit",
                    default_hp_metric=False),
                CSVLogger(
                    save_dir=default_root_dir,
                    version=os.environ.get("SLURM_JOB_ID"),
                    name="lit")
            ],
            callbacks=[
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(
                    save_last=False,
                    monitor='nlq/R1@0.3',
                    auto_insert_metric_name=False,
                    mode='max',
                    save_top_k=1,
                    filename='step={step}-nlq_R1@0.3={nlq/R1@0.3:.4f}'),
                ModelCheckpoint(
                    save_last=False,
                    monitor='nlq/R5@0.3',
                    auto_insert_metric_name=False,
                    mode='max',
                    save_top_k=config.others.save_top_k,
                    filename='step={step}-nlq_R5@0.3={nlq/R5@0.3:.4f}'),
            ],
            enable_progress_bar=enable_progress_bar,
        )

        if config.batch_flag == '1' and trainer.global_rank == 0:
            jid = os.environ.get("SLURM_JOB_ID")
            cmd = f"scontrol write batch_script {jid} {default_root_dir}/slurm-{jid}.sh"
            os.system(cmd)

        if config.others.check_valid_first:
            trainer.validate(plm, datamodule=pdm)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                time.sleep(5)

        trainer.fit(plm, datamodule=pdm)


if __name__ == '__main__':
    main()
