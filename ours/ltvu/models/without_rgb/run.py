import os
import hydra
from pathlib import Path

import torch.distributed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from .lightning_modules import TextOnlyNLQLitModule
from ltvu.data_loader.egonlq import EgoNLQRawDataModule
from ltvu.utils.callbacks import JSONPredictionWriter


OmegaConf.register_new_resolver("replace_slash", lambda s: s.replace("/", "--"))


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def run(cfg: DictConfig):
    # accumulate_grad_batches
    # logger
    # callbacks
    # enable_progress_bar
    # num_sanity_val_steps
    ngpus: int = torch.cuda.device_count()
    is_in_a_batch_job: bool = ('batch' in os.environ.get('SLURM_JOB_PARTITION').lower())
    filename = f'predictions-{cfg.experiment.prediction_split}.json'
    cfg.trainer.callbacks = [ModelSummary(max_depth=3)]
    print(cfg.hydra.run.dir)
    # if cfg.experiment.ckpt_path_test is None:
    #     cfg.trainer.callbacks += [
    #         ModelCheckpoint(
    #             monitor=(metric:='iogt>=0.3 R@01'),
    #             mode='max',
    #             filename=f'{{epoch}}-{{{metric}:.4f}}'),
    #         JSONPredictionWriter(
    #             p_json=Path(l.log_dir) / filename)
    #     ]
    #     cfg.trainer.logger = [
    #         CSVLogger(**common_logger_options),
    #         TensorBoardLogger(
    #             sub_dir='tb/',  # TensorBoard logs will be saved in /save_dir/name/version/sub_dir
    #             version=l.version,
    #             **common_logger_options),
    #     ]
    # else:
    #     cfg.trainer.callbacks += [
    #         JSONPredictionWriter(
    #             p_json=Path(l.log_dir) / filename)
    #     ]
    print(OmegaConf.to_yaml(cfg))
    print(cfg.experiment.ckpt_path_test)
    trainer = instantiate(cfg.trainer)
    print(trainer)


if __name__ == "__main__":
    run()
