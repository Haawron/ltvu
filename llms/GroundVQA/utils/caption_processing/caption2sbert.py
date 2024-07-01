import sys
import json
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


@torch.no_grad()
def main():
    # Captioner
    captioner_name = 'llava-v1.6-34b'
    # captioner_name = 'LLaVA-NeXT-Video-7B-DPO'

    # Language Encoder
    model_path = 'all-mpnet-base-v2'
    # model_path = 'multi-qa-mpnet-base-dot-v1'
    # model_path = Path('/data/gunsbrother/prjs/ltvu/everything/results/all-mpnet-base-v2/checkpoint-200')
    # model_path = Path('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs/batch/2024-06-05/15-28-58/lit/102720/checkpoints/step=4158-nlq_R5@0.3=0.000.ckpt')
    if isinstance(model_path, str):  # HF or SBert model
        model_name = model_path
        postfix = ''
    else:  # Fine-tuned model
        model_name = model_path.parent.name
        postfix = '-tuned-2024-06-05-15-28-58'

    print('Loading model ...')
    # model = SentenceTransformer(str(model_path)).cuda().eval()
    model = SentenceTransformer('all-mpnet-base-v2').cuda().eval()
    # model.load_state_dict(
    #     {k.replace('model.model.', '0.auto_model.'): v for k, v in torch.load(model_path)['state_dict'].items()},
    #     strict=False
    # )
    print('Model loaded.')

    # p_caps_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/LLaVA/results/egonlq/{captioner_name}/global')
    # p_caps_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/LLaVA-NeXT/work_dirs/{captioner_name}/global')
    p_caps_dir = Path('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/captions/llava-v1.6-34b/global')
    p_out_dir = Path(f'/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/{captioner_name}/{model_name}{postfix}')
    p_out_dir.mkdir(parents=True, exist_ok=True)

    p_caps = list(p_caps_dir.glob('*.json'))
    print(f'Found {len(p_caps)} files.')

    print('Save dir: ', p_out_dir)
    print('\n')
    num_caps_cum = 0
    pbar = tqdm(p_caps, total=len(p_caps), dynamic_ncols=True)
    for p_cap in pbar:
        cap_data = json.load(p_cap.open())['answers']
        p_out = p_out_dir / p_cap.with_suffix('.pt').name
        if p_out.exists():
            data = torch.load(p_out)
            num_caps = len(data)
            if num_caps == len(cap_data):
                sys.stdout.write('\033[A\033[2K  \033[A')
                sys.stdout.write(f'{p_out.name} already exists. Skipping.')
                sys.stdout.write('\033[B')
                num_caps_cum += num_caps
                pbar.set_postfix(num_caps=num_caps, num_caps_cum=num_caps_cum)
                continue

        frame_idxs = [entry[0] for entry in cap_data]
        caps = [entry[2] for entry in cap_data]
        embeddings = model.encode(caps, convert_to_tensor=True, convert_to_numpy=False).cpu()  # [T, D]
        output_list = [(frame_idx, emb) for frame_idx, emb in zip(frame_idxs, embeddings)]
        torch.save(output_list, p_out)
        num_caps = len(caps)
        num_caps_cum += num_caps
        pbar.set_postfix(num_caps=num_caps, num_caps_cum=num_caps_cum)
        # tqdm.write(f'Saved to {p_out}.')
    print(f'Finished. Total {num_caps_cum} captions. Saved to {p_out_dir}.')


if __name__ == '__main__':
    main()
