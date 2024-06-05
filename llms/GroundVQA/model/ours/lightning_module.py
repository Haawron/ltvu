import json
import copy
import random
from pathlib import Path
from pprint import pformat
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

from eval import calc_metrics
from eval_nlq import ReferringRecall

from .model import GroundVQA


class LightningModule(pl.LightningModule):
    def __init__(self, config, total_steps, max_epochs=20):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path, cache_dir='./cache_dir')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: GroundVQA = instantiate(config.model, max_v_len=config.dataset.max_v_len, d_env=config.dataset.get('d_env'))
        self.nlq_evaluator = ReferringRecall(
            dataset="ego4d",
            gt_file=config.dataset.nlq_val_anno
        )
        self._log_indices = {}
        self.total_steps = total_steps
        self.max_epochs = max_epochs

    def get_progress_bar_dict(self) -> torch.Dict[str, int | str]:
        bar_dict = super().get_progress_bar_dict()
        bar_dict['v_num'] = str(bar_dict['v_num'])  # prevent scientific notation
        return bar_dict

    def training_step(self, batch, batch_idx):
        total_loss, ce_loss, time_loss, cls_loss, reg_loss = self.model(**batch)
        self.log('loss/total', total_loss, rank_zero_only=True)
        self.log('loss/cls', cls_loss, rank_zero_only=True)
        self.log('loss/reg', reg_loss, rank_zero_only=True)
        if not self.config.model.ignore_decoder:
            self.log('loss/ce', ce_loss, rank_zero_only=True)
            self.log('loss/time', time_loss, rank_zero_only=True)
        return {
            'loss': total_loss,
        }

    def validation_step(self, batch, batch_idx):
        nlq_results, answer_tokens = self.model.generate(**batch)
        if not self.config.model.ignore_decoder:
            pred_answer = self.tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
        else:
            pred_answer = [''] * len(batch['video_id'])
        return {
            'question': batch['q_text'],
            'video_id': batch['video_id'],
            'answer': batch['a_text'] if 'a_text' in batch else '',
            'pred_answer': pred_answer,
            'nlq_results': nlq_results,
            'query_id': batch['query_id'],
            'sample_ratio': batch['sample_ratio'],
            'task': batch['task']
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _log_some_outputs(self, outputs, name):
        num_val_steps_to_log, num_samples_per_batch_to_log = 5, 3  # Could be configurable via cfg
        steps_to_log_indices = random.sample(range(len(outputs)), k=min(len(outputs), num_val_steps_to_log))
        self._log_indices[name] = {
            'steps': steps_to_log_indices,
            'samples': [
                random.sample(
                    range(len(outputs[step]['answer'])),
                    k=min(len(outputs[step]['answer']),
                    num_samples_per_batch_to_log))
                for step in steps_to_log_indices
            ]
        }
        for i, step in enumerate(steps_to_log_indices):
            indices = self._log_indices[name]['samples'][i]
            for b in indices:
                sample = (
                    f'Video: "{outputs[step]["video_id"][b]}". \n'
                    f'Question: "{outputs[step]["question"][b]}". \n'
                    f'Target: "{outputs[step]["answer"][b]}". \n'
                    f'Output: "{outputs[step]["pred_answer"][b]}"'
                )
                self.logger.experiment.add_text(f'{name} {str(i * len(indices) + b)}', sample,
                                                global_step=self.global_step)

    def aggregate_metrics(self, outputs, prefix):
        # evaluate CloseQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'CloseQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            num_correct = 0
            for hypo, target in zip(all_hypos, all_targets):
                if hypo == target:
                    num_correct += 1
            acc = num_correct / len(all_targets) * 100
            metrics = {f'{prefix}_close_acc': acc}
        else:
            metrics = {}

        # evaluate OpenQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'OpenQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            open_qa_metrics = calc_metrics(all_hypos, [[x] for x in all_targets], test=prefix=='test')
            for k, v in open_qa_metrics.items():
                metrics[f'{prefix}_{k}'] = v

        # evalute NLQ
        nlq_preds = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] != 'NLQ':
                    continue
                qid = output['query_id'][i]
                temp_list = qid.split("_")
                sample_ratio = output['sample_ratio'][i]
                new_prediction = [
                    [   segment[0] / sample_ratio,
                        segment[1] / sample_ratio,
                        score  ]
                    for segment, score in zip(
                        output['nlq_results'][i]['segments'].cpu().detach().tolist(),
                        output['nlq_results'][i]['scores'].cpu().detach().tolist(),
                )]
                nlq_preds.append({
                    'query_idx': int(temp_list[1]),
                    'annotation_uid': temp_list[0],
                    'predicted_times': new_prediction,
                    'clip_uid': output['video_id'][i]
                })
        if len(nlq_preds) > 0:
            performance, score_str, all_ious = self.nlq_evaluator.evaluate(nlq_preds, verbose=False, return_ious=True)
            metrics[f'{prefix}/R1_03'] = performance[0, 0] * 100
            metrics[f'{prefix}/R5_03'] = performance[0, 1] * 100
            metrics[f'{prefix}/R1_05'] = performance[1, 0] * 100
            metrics[f'{prefix}/R5_05'] = performance[1, 1] * 100
            metrics[f'{prefix}/Mean_R1'] = (performance[0, 0] + performance[1, 0]) * 100 / 2

        # save predictions
        results = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                result = {
                    'clip_uid': output['video_id'][i],
                    'query_id': output['query_id'][i],
                    'pred_answer': output['pred_answer'][i],
                    'gt_answer': output['answer'][i],
                    'pred_window': (output['nlq_results'][i]['segments'].cpu().detach() / output['sample_ratio'][i]).tolist(),
                    'gt_window': self.nlq_evaluator.gt_dict[(output['video_id'][i], output['query_id'][i].split('_')[0])]["language_queries"][int(output['query_id'][i].split('_')[1])],
                }
                for pred, iou in zip(result['pred_window'], all_ious[result['query_id']]):  # list of lists of floats(s, e)
                    pred.append(iou.item())
                results.append(result)

        if self.config.dataset.get('additional_feature_type') is None:
            (p_out_json:=Path('analysis/VLG_OpenQA.json')).parent.mkdir(parents=True, exist_ok=True)
            with p_out_json.open('w') as f:
                json.dump(results, f)

        return metrics

    # def training_epoch_end(self, outputs):
        # self._log_some_outputs(outputs, 'train')
        # metrics = self.aggregate_metrics(outputs, prefix='train')
        # self.log_dict(metrics, sync_dist=True)

    def validation_epoch_end(self, outputs):
        def _mean(key):
            return torch.stack([data[key] for data in outputs]).mean()

        # self._log_some_outputs(outputs, 'val')
        metrics = self.aggregate_metrics(outputs, prefix='val')
        if tqdm._instances:
            msg = pformat(metrics)
            next(iter(tqdm._instances)).write(msg)
        metrics.update({key.replace('val/', 'hp/'): value for key, value in metrics.items() if 'val/' in key})  # for hparam logging
        metrics.update({f'val/{name}': _mean(name) for name in outputs[0].keys() if 'loss' in name})
        self.log_dict(metrics, sync_dist=True)

    def test_epoch_end(self, outputs):
        # self._log_some_outputs(outputs, 'test')
        metrics = self.aggregate_metrics(outputs, prefix='test')
        self.log_dict(metrics, sync_dist=True)
        if self.config.trainer.save_nlq_results is not None:
            src = 'data/joint/annotations.QaEgo4D_test_close.json'
            dst = self.config.trainer.save_nlq_results
            self.save_nlq_results(src, dst, outputs)

    def save_nlq_results(self, src, dst, preds):
        # aggregate preds
        pred_dict = {}
        for batch_pred in preds:
            for i in range(len(batch_pred['video_id'])):
                qid = batch_pred['query_id'][i]
                sample_ratio = batch_pred['sample_ratio'][i]
                pred_start = batch_pred['nlq_results'][i]['segments'][0].cpu().detach().tolist()[0] / sample_ratio
                pred_end = batch_pred['nlq_results'][i]['segments'][0].cpu().detach().tolist()[1] / sample_ratio
                assert qid not in pred_dict
                pred_dict[qid] = {
                    'pred_start_sec': pred_start,
                    'pred_end_sec': pred_end
                }

        save_results = []
        for src_data in json.load(open(src)):
            pred_data = pred_dict[src_data['sample_id']]
            save_data = copy.deepcopy(src_data)
            save_data['moment_start_frame'] = pred_data['pred_start_sec'] * 30
            save_data['moment_end_frame'] = pred_data['pred_end_sec'] * 30
            save_results.append(save_data)
        with open(dst, 'w') as f:
            json.dump(save_results, f)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.config.optim.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.optim.optimizer.lr
        )
        if self.config.optim.lr_scheduler:
            # lr_scheduler = OneCycleLR(
            #     optimizer=optimizer,
            #     max_lr=self.config.optim.optimizer.lr,
            #     total_steps=self.total_steps,
            #     anneal_strategy='linear'
            # )
            lr_scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.total_steps//(self.max_epochs // 5),
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
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

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            "hp/Mean_R1": 0,
            "hp/R1_03": 0, "hp/R5_03": 0,
            "hp/R1_05": 0, "hp/R5_05": 0,
        })

    # def on_train_end(self):
    #     self.logger.log_hyperparams(self.hparams, {
    #         "hp/Mean_R1": 0,
    #         "hp/R1_03": 0, "hp/R5_03": 0,
    #         "hp/R1_05": 0, "hp/R5_05": 0,
    #     })
