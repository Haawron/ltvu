import hydra.utils

from pprint import pprint

import torch

import lightning as L

from .model import SentenceGroundingModel


class LitModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = SentenceGroundingModel(config)

    def log_dict_split_prefixed(self, d, split='val', **kwargs):
        for k, v in d.items():
            if isinstance(v, dict):
                # e.g., nlq_score/AUC -> nlq_score/val_AUC
                for kk, vv in v.items():
                    self.log(f'{k}/{split}_{kk}', vv, **kwargs)
            else:
                # e.g., loss/q_cap -> loss/val_q_cap
                comps = k.split('/')
                comps[-1] = f'{split}_{comps[-1]}'
                k = '/'.join(comps)
                self.log(k, v, **kwargs)

    def training_step(self, batch, batch_idx):
        bsz = 1 #len(batch['queries'])
        output_dict = self.model(**batch)
        # self.log_dict_split_prefixed(output_dict['loss_dict'], split='train', batch_size=bsz, sync_dist=True)
        self.log_dict(output_dict['loss_dict'], batch_size=bsz, prog_bar=True, rank_zero_only=True)
        return output_dict['loss']

    def validation_step(self, batch, batch_idx):
        bsz = 1 #len(batch['queries'])
        output_dict = self.model(**batch)
        self.log_dict_split_prefixed(output_dict['loss_dict'], split='val', batch_size=bsz, sync_dist=True)
        self.log_dict(output_dict['score_dict'], batch_size=bsz, sync_dist=True)
        return output_dict

    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            pprint(
                {k: v.item() for k, v in self.trainer.callback_metrics.items()},
                sort_dicts=False
            )

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.optimizer = hydra.utils.instantiate(
            self.config.optim.optimizer, params=params)
        return self.optimizer

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step == 0:
            unintended_no_grad_captured = False
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is None:
                    print(f'{n} [{p.shape}] requires grad but got None')
                    unintended_no_grad_captured = True
            if unintended_no_grad_captured:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                raise ValueError("No gradients")

    # def on_validation_epoch_end(self):
    #     pass
