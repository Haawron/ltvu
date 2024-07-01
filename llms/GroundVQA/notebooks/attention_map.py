import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import hydra

import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer

from model.ours.lightning_module import LightningModule


def mpl_heatmap(square_matrix, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    square_matrix -= square_matrix.min()
    square_matrix /= square_matrix.max()
    square_matrix **= 0.5  # gamma correction -> more sensitive to lower values
    ax.imshow(square_matrix, cmap='magma')
    return ax


hydra_inst = hydra.initialize(config_path='../config', job_name='train', version_base='1.3')
config = hydra.compose(config_name='base', overrides=[f'model=groundvqa_b'])

model = LightningModule(config, 10, 100).cuda()
state_dict = torch.load('GroundVQA/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt')
model.load_state_dict(state_dict['state_dict'])

clip_uid = '5e59031d-0deb-4557-a3e1-ba0ba2bb5465'
v_feat = torch.load(f'data/features/egovlp_internvideo/{clip_uid}.pt')
print('Original visual feature size:', v_feat.shape)
v_feat = v_feat[torch.linspace(0, v_feat.shape[0]-1, config.dataset.max_v_len).long()].cuda()
v_mask = torch.ones((1, v_feat.shape[0]), dtype=torch.bool).cuda()
print('Reduced visual feature size and its mask:', v_feat.shape, v_mask.shape)
num_v_tokens = v_feat.shape[0]

anns = json.load(open('data/unified/annotations.NLQ_val.json'))
sample_ids = []
questions, gt_segments = [], []
for ann in anns:
    if ann['video_id'] == clip_uid:
        sample_ids.append(ann['sample_id'])
        questions.append(ann['question'])
        gt_segments.append((ann['clip_start_sec'], ann['clip_end_sec']))
        duration_sec = ann['clip_duration']
gt_segments = np.array(gt_segments)
gt_idxs = (gt_segments / duration_sec * num_v_tokens).round().astype(int)
tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
tokens = tokenizer(questions, padding=True, return_tensors='pt', add_special_tokens=False).to('cuda')

for batch_idx in range(len(questions)):
    q_token = tokens.input_ids[batch_idx:batch_idx+1]
    q_mask = tokens.attention_mask[batch_idx:batch_idx+1]
    num_q_tokens = q_token.shape[1]
    with torch.no_grad():
        model.eval()
        v_input = model.model.lm_proj(v_feat) + model.model.v_emb
        q_input = model.model.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_input, v_input], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        encoder_outputs = model.model.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True
        )
        nlq_results = model.model.nlq_head(
            feat=encoder_outputs.last_hidden_state[:, num_q_tokens:].permute(0, 2, 1),
            mask=v_mask.unsqueeze(1),
            training=False,
            v_lens=torch.tensor([num_v_tokens])
        )
        pred_segments = nlq_results[0]['segments'].cpu().numpy()  # [k, 2]
        pred_idxs = (pred_segments / duration_sec * num_v_tokens).round().astype(int)

    num_layers = len(encoder_outputs.attentions)
    num_heads = encoder_outputs.attentions[0].shape[1]
    downsample = 10
    figsize_factor = 2
    ticks = np.arange(0, num_v_tokens // downsample, 25)
    ticklabels = downsample * ticks
    ticks += num_q_tokens
    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(figsize_factor*20, figsize_factor*22),
        sharey=True, sharex=True, gridspec_kw={'wspace': 1e-2, 'hspace': 1e-2})
    plt.subplots_adjust(left=0.05, right=0.95)
    for idx_layer in range(num_layers):
        for idx_head in range(num_heads):
            ax = axes[idx_layer, idx_head]
            down_idxs = np.concatenate([np.arange(num_q_tokens), np.arange(num_q_tokens, num_q_tokens+num_v_tokens, downsample)])
            attention_map = encoder_outputs.attentions[idx_layer][0][idx_head].cpu().numpy()
            attention_map = attention_map[down_idxs][:, down_idxs]
            # sns.heatmap(attention_map, cmap='magma', square=True, cbar=False, ax=ax)  # slow
            mpl_heatmap(attention_map, ax=ax)
            if idx_layer == num_layers-1:
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticklabels, fontsize=8, rotation=0)
            else:
                ax.set_xticks([])
            if idx_layer == 0:
                ax.set_title(f'Head {idx_head}', fontsize=12)
            if idx_head == 0:
                ax.set_yticks(ticks)
                ax.set_yticklabels(ticklabels, fontsize=8, rotation=0)
                ax.set_ylabel(f'Layer {idx_layer}', fontsize=12)
            else:
                ax.set_yticks([])
    br = '\n'
    msg = br.join([
        'Attention Maps',
        f'Clip UID: {clip_uid} - Sample ID: {sample_ids[batch_idx]} - Question: {questions[batch_idx]}',
        f'GT Segments: {gt_segments[batch_idx].round(2)} - GT Indexes: {gt_idxs[batch_idx]}',
        f'Predicted Segments: {str(pred_segments.round(2)).replace(br, " ")}',
        f'Predicted Indexes: {str(pred_idxs).replace(br, " ")}',
    ])
    fig.suptitle(msg, fontsize=22)
    p_out = Path(f'analysis/attention_maps/{sample_ids[batch_idx]}.png')
    plt.savefig(p_out)
    plt.close(fig)
    print(f'Saved {p_out}')
