import fire
import json
from vllm import LLM
import math
from src.serve.inference import inference
import re
import string
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm.auto import tqdm


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parser(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []

class NEREvaluator:
    def evaluate(self, preds: list, golds: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for pred, gold in zip(preds, golds):
            gold_tuples = parser(gold)
            pred_tuples = parser(pred)
            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }


def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    data_path: Union[str, Path] = Path('./src/eval/test_data/CrossNER_AI.json'),
    tensor_parallel_size: int = 1,
):
    llm = LLM(model=model_path, download_dir='./cache_dir', tensor_parallel_size=tensor_parallel_size)

    def do(data_path: Union[str, Path], p_out_dir=None):
        data_path = Path(data_path)
        with data_path.open() as fh:
            examples = json.load(fh)
        golds = [example['conversations'][-1]['value'] for example in examples]
        outputs = inference(llm, examples)

        eval_result = NEREvaluator().evaluate(outputs, golds)
        print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')

        records = []
        for output, example in zip(outputs, examples):
            words = eval(output)
            words = words[1:] if len(words) >= 1 and 'camera' in words[0] else words
            records.append({
                **example['info'],
                'entities': {'type': example['entity_type'], 'values': words},
            })
        if p_out_dir:
            p_out_dir = Path(p_out_dir)
            p_out_dir.mkdir(parents=True, exist_ok=True)
            p_out_json = p_out_dir / data_path.name
            with p_out_json.open('w') as f:
                json.dump(records, f)
            print(f'Saved to {p_out_json}')

    if (p:=Path(data_path)).is_dir():
        assert 'input' in p.name
        p_out_dir = p.with_name(p.name.replace('input', 'output'))
        for i, p_json in tqdm(list(enumerate(p.glob('**/*.json')))):
            do(p_json, p_out_dir=p_out_dir)
    else:
        do(p)


if __name__ == "__main__":
    fire.Fire(main)
