import pickle
import itertools

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ltvu.models.egovlpv1.model import FrozenInTime


def compute_ious(
    pred_s: np.ndarray, pred_e: np.ndarray,
    gt_s: np.ndarray, gt_e: np.ndarray,
):
    # inputs and IoU Matrix: [Q, k=5]
    intersections = np.maximum(np.minimum(pred_e, gt_e) - np.maximum(pred_s, gt_s), 0)
    unions = np.maximum(pred_e, gt_e) - np.minimum(pred_s, gt_s) + 1e-12
    ious = intersections / unions
    return ious


def compute_score_records(
    pred_s: np.ndarray, pred_e: np.ndarray,
    gt_s: np.ndarray, gt_e: np.ndarray,
    ks = [1, 5], iou_thresholds = [0.3, 0.5],
) -> dict[str, npt.NDArray[np.bool_]]:
    """
    # Arguments
    `pred_s, pred_e`: `[Q, k]`, should be sorted in advance in descending order.
    `gt_s, gt_e`: `[Q,]` or `[Q, 1]`
    """
    # IoU Matrix: [Q, # preds]
    ious = compute_ious(pred_s, pred_e, gt_s, gt_e)
    # R@1, R@5: [Q,]
    result = {}
    for k, iou_th in itertools.product(ks, iou_thresholds):
        correct = (ious[:, :k] >= iou_th).sum(axis=-1) > 0
        result[f'R@{k} IoU={iou_th}'] = correct
    result['mIoU'] = ious.max(axis=-1)  # for mIoU(mean) computing; why max? ==> the best IoU for each query
    return result


def _base_usage(
    use_egoclip_span_as_pred = False,
    narr_span = (5., 5.),
    debug = False,
):
    device = 'cuda'
    kwargs_encode = {
        'device': device,
        'convert_to_tensor': False, 'convert_to_numpy': True,
        'normalize_embeddings': True
    }
    ######################## BEGIN model setup ########################
    # model = SentenceTransformer("all-mpnet-base-v2", device=device)
    model = FrozenInTime(
        video_params={
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'pretrained': True,
            'num_frames': 16,
            'time_init': 'zeros',
        },
        text_params={
            'model': 'distilbert-base-uncased',
            'pretrained': True,
            'input': 'text',
        },
        projection_dim=256,
        load_checkpoint='pretrained/egovlp-config-removed.pth',
        projection='minimal')
    model.video_model = nn.Identity()
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased', TOKENIZERS_PARALLELISM=False)
    @torch.no_grad()
    def __encode(texts: list[str], device='cuda', **kwargs):
        tokens = tokenizer(
            texts,
            return_tensors='pt', padding=True, truncation=True)
        emb = model.compute_text(tokens.to(device)).detach()
        emb /= emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy()
        return emb
    model.encode = __encode
    ######################## END model setup ########################

    with open('data/egoclip/egoclip_nlq.pkl', 'rb') as f:  # TODO: only valid
        data = pickle.load(f)

    score_records = {
        'R@1 IoU=0.3': [],
        'R@5 IoU=0.3': [],
        'R@1 IoU=0.5': [],
        'R@5 IoU=0.5': [],
        'mIoU': [],
    }

    count_computed = 0
    for i, clip_instances in enumerate(tqdm(data['clips'])):
        if len(clip_instances['narrations']['text']) < 10:
            continue
        Z_q: np.ndarray = model.encode(clip_instances['queries']['query'], **kwargs_encode)
        Z_narr: np.ndarray = model.encode(clip_instances['narrations']['text'], **kwargs_encode)
        simmat = util.dot_score(Z_q, Z_narr).numpy()  # [Q, Narrs]
        top5s = simmat.argsort(axis=-1)[:, -1:-6:-1]  # [Q, k=5]
        # [Q, k=5]
        if use_egoclip_span_as_pred:
            pred_s = np.take_along_axis(np.array(clip_instances['narrations']['start'])[:, None], top5s, axis=0)
            pred_e = np.take_along_axis(np.array(clip_instances['narrations']['end'])[:, None], top5s, axis=0)
        else:
            left, right = narr_span
            pred_t = np.take_along_axis(np.array(clip_instances['narrations']['time'])[:, None], top5s, axis=0)
            pred_s = np.clip(pred_t - left, 0, clip_instances['clip_end_sec'] - clip_instances['clip_start_sec'])
            pred_e = np.clip(pred_t + right, 0, clip_instances['clip_end_sec'] - clip_instances['clip_start_sec'])
        gt_s = np.array(clip_instances['queries']['start'])[:, None]
        gt_e = np.array(clip_instances['queries']['end'])[:, None]
        # IoU Matrix: [Q, k=5]
        intersections = np.maximum(np.minimum(pred_e, gt_e) - np.maximum(pred_s, gt_s), 0)
        unions = np.maximum(pred_e, gt_e) - np.minimum(pred_s, gt_s) + 1e-12
        ious = intersections / unions
        # R@1, R@5: [Q,]
        r1_03 = (ious[:, :1] >= 0.3).sum(axis=-1) > 0
        r5_03 = (ious >= 0.3).sum(axis=-1) > 0
        r1_05 = (ious[:, :1] >= 0.5).sum(axis=-1) > 0
        r5_05 = (ious >= 0.5).sum(axis=-1) > 0
        max_ious = ious.max(axis=-1)  # for mIoU(mean) computing; why max? ==> the best IoU for each query
        score_records['R@1 IoU=0.3'] += r1_03.tolist()
        score_records['R@5 IoU=0.3'] += r5_03.tolist()
        score_records['R@1 IoU=0.5'] += r1_05.tolist()
        score_records['R@5 IoU=0.5'] += r5_05.tolist()
        score_records['mIoU'] += max_ious.tolist()
        count_computed += simmat.shape[0]  # Q
        if debug and i == 0:
            print(Z_q.shape)
            print(Z_narr.shape)
            print(simmat)
            print(simmat.shape)
            print()

            print(ious)
            print(ious.shape)
            print()

            print(r1_03)
            print(r5_03)
            print(r1_05)
            print(r5_05)
            print(max_ious)

    print()
    if not use_egoclip_span_as_pred:
        print(f'Narration spans (left, right): ({narr_span[0]}s, {narr_span[1]}s)')
    print()
    for name, record in score_records.items():
        print(f'{name}: {np.mean(record) * 100:.2f}')

    print(f'\ncount_computed: {count_computed}')


@torch.no_grad()
def get_token_feature():
    model = SentenceTransformer("all-mpnet-base-v2").cuda()
    model.eval()
    print(model.tokenizer)
    print()
    print(model[0])  # model[0] = {auto_model: {embeddings, encoder, pooler}}
    print()
    print(model[1])  # model[1] = {pooling}
    print()
    print(model[0].auto_model.embeddings)
    y = model.tokenizer(['hi', 'hello', 'how are you?'],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=32)  # input_ids, attnmask
    print(y)

    mpt = model[0].auto_model
    input_embeds = mpt.embeddings.word_embeddings(y['input_ids'].cuda())
    print(input_embeds.shape)
    pos_embeds = mpt.embeddings.position_embeddings(mpt.embeddings.create_position_ids_from_inputs_embeds(input_embeds))
    embeddings = input_embeds + pos_embeds
    embeddings = mpt.embeddings.dropout(mpt.embeddings.LayerNorm(embeddings))
    output, = mpt.encoder(
        embeddings,
        attention_mask=mpt.get_extended_attention_mask(y['attention_mask'].cuda(), y['input_ids'].shape),
        head_mask=mpt.get_head_mask(None, mpt.config.num_hidden_layers)
    )
    output = model[1](dict(
        token_embeddings=output,
        attention_mask=y['attention_mask'].cuda()
    ))['sentence_embedding']
    output /= output.norm(dim=-1, keepdim=True)
    print(output)
    print(output.shape)

    output2 = model.forward(dict(
        input_ids=y['input_ids'].cuda(),
        attention_mask=y['attention_mask'].cuda()
    ))['sentence_embedding']
    print(output2)
    print(output2.shape)

    print((output - output2).norm())
    print()


if __name__ == '__main__':
    # _base_usage()

    # for l in [1., 5., 10., 15., 20.]:
    #     for r in [1., 5., 10., 15., 20.]:
    #         _base_usage(narr_span=(l, r))
    #         print('\n=====================================\n')

    get_token_feature()
