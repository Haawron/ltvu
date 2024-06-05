from dotenv import load_dotenv

load_dotenv()

import os
import re
import math
from pprint import pprint
from argparse import ArgumentParser, Namespace

import hydra
import torch
import torch.distributed
import pytorch_lightning as pl

from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from model.ours.dataset import JointDataModule
from model.ours.lightning_module import LightningModule


def dict_parser(s: str):
    return eval('{' + re.sub(r'(\w+)=(["\']?\w+["\']?)', r'"\1":\2', s) + '}')

def add_common_trainer_util_args(parser, default_monitor_variable='val_loss', default_monitor_mode='min'):
    if default_monitor_mode not in ['min', 'max']:
        raise ValueError(default_monitor_mode)
    parser.add_argument('--lr_find_kwargs', default=dict(min_lr=5e-6, max_lr=1e-2), type=dict_parser,
                        help='Arguments for LR find (--auto_lr_find). Default "min_lr=5e-6,max_lr=1e-2"')
    parser.add_argument('--random_seed', default=42, type=lambda s: None if s == 'None' else int(s),
                        help='Seed everything. Set to "None" to disable global seeding')
    parser.add_argument('--auto_resume', default=False, action='store_true',
                        help='Automatically resume last saved checkpoint, if available.')
    parser.add_argument('--test_only', default=False, action='store_true',
                        help='Skip fit and call only test. This implies automatically detecting newest checkpoint, '
                             'if --checkpoint_path is not given.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Load this checkpoint to resume training or run testing. '
                             'Pass in the special value "best" to use the best checkpoint according to '
                             'args.monitor_variable and args.monitor_mode. '
                             'Using "best" only works with test_only mode.')
    parser.add_argument('--ignore_existing_checkpoints', default=False, action='store_true',
                        help='Proceed even with training a new model, even if previous checkpoints exists.')
    parser.add_argument('--monitor_variable', default=default_monitor_variable, type=str,
                        help='Variable to monitor for early stopping and for checkpoint selection. '
                             f'Default: {default_monitor_variable}')
    parser.add_argument('--monitor_mode', default=default_monitor_mode, type=str, choices=['min', 'max'],
                        help='Mode for monitoring the monitor_variable (for early stopping and checkpoint selection). '
                             f'Default: {default_monitor_mode}')
    parser.add_argument('--reset_early_stopping_criterion', default=False, action='store_true',
                        help='Reset the early stopping criterion when loading from checkpoint. '
                             'Prevents immediate exit after switching to more complex dataset in curriculum strategy')

def apply_argparse_defaults_to_hydra_config(config: DictConfig, parser: ArgumentParser, verbose=False):
    args = parser.parse_args([])  # Parser is not allowed to have required args, otherwise this will fail!
    defaults = vars(args)

    def _apply_defaults(dest: DictConfig, source: dict, indentation=''):
        for k, v in source.items():
            if k in dest and isinstance(v, dict):
                current_value = dest[k]
                if current_value is not None:
                    assert isinstance(current_value, DictConfig)
                    _apply_defaults(current_value, v, indentation + ' ')
            elif k not in dest:
                dest[k] = v
                if verbose:
                    print(indentation, 'set default value for', k)

    with open_dict(config):
        _apply_defaults(config, defaults)


def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if trainer_cfg['gpus'] > 1 and strategy is None:
        strategy = 'ddp'  # Select ddp by default
    if strategy == 'ddp':
        trainer_cfg['strategy'] = DDPPlugin(
            find_unused_parameters=False,#trainer_cfg['find_unused_parameters'],
            gradient_as_bucket_view=True)
    return trainer_cfg


@hydra.main(config_path='config', config_name='base', version_base='1.3')
def train(config: DictConfig):
    fake_parser = ArgumentParser()
    add_common_trainer_util_args(fake_parser, default_monitor_variable='val_loss')
    apply_argparse_defaults_to_hydra_config(config.trainer, fake_parser)
    pl.seed_everything(config.trainer.random_seed, workers=True)
    trainer_cfg = Namespace(**_adjust_ddp_config(config.trainer))
    trainer_cfg.weights_summary = False
    trainer_cfg.default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # if abs(float(config.optim.optimizer.lr) - (base_lr:=1e-5)) < 1e-12:
    #     raise ValueError(f'Got the base LR itself: {base_lr:.1e}. Learning rate is not scaled properly. Please check the config file.')

    data = JointDataModule(config.dataset)
    data.setup()

    total_steps = trainer_cfg.max_epochs * math.floor(len(data.train_dataset) / trainer_cfg.gpus / config.dataset.batch_size / trainer_cfg.accumulate_grad_batches)
    model = LightningModule(config, total_steps, trainer_cfg.max_epochs)

    print('=' * 80)
    if trainer_cfg.checkpoint_path:
        print(f'\n[{__name__}] Load checkpoint: {trainer_cfg.checkpoint_path}')
        state_dict = torch.load(trainer_cfg.checkpoint_path, map_location='cpu')['state_dict']
        if not trainer_cfg.load_nlq_head:
            print('Train NLQ head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not "nlq_head" in k}
        if not trainer_cfg.load_decoder:
            print('Train LM decoder head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not ("decoder" in k or "lm_head" in k)}
        model_variant = config.model.get('model_variant', 't5')
        if model_variant == 't5_ca':
            from model.ours.ltvu.t5_with_ca_encoder import shift_cross_attention_added_block_names
            state_dict = shift_cross_attention_added_block_names(state_dict, cross_attention_layer_idxs=config.model.t5_ca_layer_idxs)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print('Missing Keys:')
            pprint(missing_keys)
            print('Unexpected Keys:')
            pprint(unexpected_keys)
        print()
    else:
        print(f'\n[{__name__}] Train from scratch\n')
    print('=' * 80)

    if trainer_cfg.test_only:  # evaluation
        trainer = pl.Trainer.from_argparse_args(
            trainer_cfg,
            enable_checkpointing=False,
            logger=False
        )
        if trainer_cfg.val:
            trainer.validate(model, data.val_dataloader())
        else:
            trainer.test(model, data.test_dataloader())
    else:  # training
        model_checkpoint = []
        if 'QaEgo4D_test' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False,
                    monitor='val_ROUGE',
                    mode='max',
                    save_top_k=1,
                    filename='{step}-{' + 'val_ROUGE' + ':.3f}')
            )
        if 'QaEgo4D_test_close' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False,
                    monitor='val_close_acc',
                    mode='max',
                    save_top_k=1,
                    filename='{step}-{' + 'val_close_acc' + ':.3f}')
            )
        if 'NLQ_val' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False,
                    monitor='val/R5_03',
                    mode='max',
                    save_top_k=1,
                    filename='{step}-{' + 'val_R5_03' + ':.3f}')
            )
        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            trainer_cfg,
            callbacks=[
                ModelSummary(max_depth=2),
                LearningRateMonitor(logging_interval='step'),
                # StochasticWeightAveraging(swa_lrs=1e-2),
                *model_checkpoint
            ],
            logger=TensorBoardLogger(
                save_dir=trainer_cfg.default_root_dir,
                version=os.environ.get("SLURM_JOB_ID"),
                name="lit",
                # sub_dir='tb',
                default_hp_metric=False
            )
        )
        # write a copy of the batch script to the PL output directory
        if config.batch_flag and trainer.global_rank == 0:
            jid = trainer.slurm_job_id
            cmd = f"scontrol write batch_script {jid} {trainer_cfg.default_root_dir}/slurm-{jid}.sh"
            os.system(cmd)
        trainer.fit(
            model, data.train_dataloader(), data.val_dataloader(),
        )


if __name__ == '__main__':
    train()
