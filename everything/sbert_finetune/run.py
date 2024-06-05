from dotenv import load_dotenv
load_dotenv()

import os

import torch
import hydra
from omegaconf import OmegaConf, DictConfig

import lightning as L
import lightning.pytorch.utilities as L_utils
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from model.lightning_module import LitModule
from model.dataset import LitDataModule


@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


@hydra.main(config_path='config', config_name='base', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.get('seed', 42), workers=True)
    torch.set_float32_matmul_precision('medium')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    os.environ["SLURM_JOB_NAME"] = "bash"
    if config.batch_flag != '1':
        log_to_console(OmegaConf.to_yaml(config, resolve=True))

    # if __debug__:
    #     print('\n\nDebug mode is on\n\n')
    #     config.dataset.num_workers = 0
    #     config.dataset.prefetch_factor = None
    #     config.dataset.pin_memory = False
    #     config.dataset.persistent_workers = False

    plm = LitModule(config)
    pdm = LitDataModule(config)

    trainer = L.Trainer(
        **OmegaConf.to_container(config.trainer, resolve=True),
        enable_model_summary=False,
        default_root_dir=default_root_dir,
        logger=TensorBoardLogger(
            save_dir=default_root_dir,
            version=os.environ.get("SLURM_JOB_ID"),
            name="lit",
            default_hp_metric=False
        ),
        callbacks=[
            ModelSummary(max_depth=3),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                save_last=False,
                monitor='nlq/R5@0.3',
                mode='max',
                save_top_k=1,
                filename='{step}-{' + 'nlq_R5@0.3' + ':.3f}'),
        ],
        enable_progress_bar=config.batch_flag != '1',
    )

    if config.batch_flag == '1' and trainer.global_rank == 0:
        jid = os.environ.get("SLURM_JOB_ID")
        cmd = f"scontrol write batch_script {jid} {default_root_dir}/slurm-{jid}.sh"
        os.system(cmd)

    trainer.fit(plm, datamodule=pdm)


if __name__ == '__main__':
    main()
