import json

import torch
import lightning as L
from transformers import AutoTokenizer

from ltvu.data_loader.egonlq import EgoNLQwithNarrationsDataset
from ltvu.models.egovlpv1.model import TextOnlyFrozenInTime



class LightningModule(L.LightningModule):
    # def __init__(self, config, total_steps):
    def __init__(self, total_steps=500000):
        super().__init__()
        # self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', TOKENIZERS_PARALLELISM=False)
        self.model = TextOnlyFrozenInTime()
        # self.model = instantiate(config.model, max_v_len=config.dataset.max_v_len)
        # self.nlq_evaluator = ReferringRecall(
        #     dataset="ego4d",
        #     gt_file=config.dataset.nlq_val_anno
        # )
        self._log_indices = {}
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        time_loss = self.model(batch)
        self.log('time_loss', time_loss, rank_zero_only=True)
        return {
            'loss': time_loss,
        }

    def validation_step(self, batch, batch_idx):
        self.nlq_results = self.model(batch, training=False)
        print(self.nlq_results)
        return {
            'question': batch[0]['captions'],
            'nlq_results': self.nlq_results,
            # 'query_id': batch['query_id'],
            # 'sample_ratio': batch['sample_ratio'],
            # 'task': batch['task']
        }

    def on_validation_epoch_end(self):
        def _mean(key):
            return torch.stack([data[key] for data in self.nlq_results]).mean()

        # self._log_some_outputs(outputs, 'val')
        metrics = self.aggregate_metrics(self.nlq_results, prefix='val')
        metrics.update({
            f'val_{name}': _mean(name) for name in self.nlq_results[0].keys() if 'loss' in name
        })
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-5,
            weight_decay=.01,
        )
        return optimizer

    def aggregate_metrics(self, outputs, prefix):
        # evalute NLQ
        metrics, nlq_preds = {}, []
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
            metrics[f'{prefix}_R1_03'] = performance[0, 0] * 100
            metrics[f'{prefix}_R5_03'] = performance[0, 1] * 100
            metrics[f'{prefix}_R1_05'] = performance[1, 0] * 100
            metrics[f'{prefix}_R5_05'] = performance[1, 1] * 100
            metrics[f'{prefix}_Mean_R1'] = (performance[0, 0] + performance[1, 0]) * 100 / 2

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

        from pathlib import Path
        (p_out_json:=Path('analysis/VLG_OpenQA.json')).parent.mkdir(parents=True, exist_ok=True)
        with p_out_json.open('w') as f:
            json.dump(results, f)

        return metrics


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')  # bf16 mixed
    model = LightningModule()
    dataset = EgoNLQwithNarrationsDataset()
    # print(model.tokenizer)
    # sample_captions = [
    #     '#C washes dishes.',
    #     '#C is picking up dishes.',
    #     '#C is looking for a hammer.',
    #     'I am in the kitchen.',
    # ]
    # tokens = model.tokenizer(
    #     sample_captions,
    #     return_tensors='pt', padding=True, truncation=True,
    # )
    # print(tokens['input_ids'])
    # print(tokens['attention_mask'])
    # print()
    # feats = model.model.compute_text(tokens)
    # print(feats.shape)
    # print(feats @ feats.T)  # norm 해야 되나?

    # print('\n\n#########################\n\n')

    # from itertools import repeat
    # loader = torch.utils.data.DataLoader(list(repeat([dataset[0]], 8)), batch_size=1, num_workers=8)
    # trainer = L.Trainer(limit_train_batches=16, max_epochs=1)
    # trainer.fit(model, loader, loader)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)
    trainer = L.Trainer(limit_train_batches=16, max_epochs=1)
    trainer.fit(model, loader, loader)
