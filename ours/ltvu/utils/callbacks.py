import json
import shutil
from pathlib import Path
from typing_extensions import override

import torch.distributed

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter


class JSONPredictionWriter(BasePredictionWriter):
    def __init__(self, p_json: str|Path):
        super().__init__('batch_and_epoch')
        self.p_json = Path(p_json)

        self.p_json_tmp_dir = self.p_json.parent / 'tmp'
        self.p_json_tmp_dir.mkdir(parents=True, exist_ok=True)

    # def write_on_batch_end(  # on prediction batch end
    #     self,
    #     trainer,
    #     pl_module,
    #     prediction,
    #     batch_indices,
    #     batch,
    #     batch_idx,
    #     dataloader_idx
    # ):
    #     """Actually donsn't write anything, just to show the signature."""
    #     pass

    @override
    def write_on_epoch_end(self,
        trainer,
        pl_module,
        predictions,
        batch_indices
    ):
        if not torch.distributed.is_initialized():
            json.dump(predictions, self.p_json.open('w'))
            print(f'JSON saved to {self.p_json}')
        else:
            tmpfilename = self.p_json.stem + f'_{trainer.global_rank}'
            p_json_tmp = self.p_json_tmp_dir / f'{tmpfilename}.json'
            json.dump(predictions, p_json_tmp.open('w'))
            print(f'JSON saved to {p_json_tmp}')
            torch.distributed.barrier()
            if trainer.is_global_zero:
                p_jsons = sorted(self.p_json_tmp_dir.glob(f'*.json'))
                jsons = [json.load(p.open('r')) for p in p_jsons]
                num_entries = list(map(len, jsons))
                jsons = sum(jsons, [])
                json.dump(jsons, self.p_json.open('w'))
                print(f'JSONs merged and saved to {self.p_json}')
                print(f'Number of entries: {num_entries}, Total: {len(jsons)}')
                shutil.rmtree(self.p_json_tmp_dir)
            torch.distributed.barrier()

    @override
    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.interval.on_epoch:
            return
        trainer.test_loop.predictions = []

    @override
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: list[dict],
        # batch: Any,
        # batch_idx: int,
        # dataloader_idx: int = 0,
        *args, **kwargs
    ) -> None:
        if not self.interval.on_batch:
            return
        trainer.test_loop.predictions.extend(outputs)

    @override
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.interval.on_epoch:
            return
        self.write_on_epoch_end(
            trainer,
            pl_module,
            trainer.test_loop.predictions,
            None
        )
