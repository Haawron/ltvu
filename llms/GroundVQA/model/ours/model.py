import math
from typing import Literal
from pprint import pprint

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead
from model.ours.ltvu.t5_with_ca_encoder import T5WithCAEncoderAndDecoder, find_all_cross_attention_parameters
from model.ours.ltvu.detr import DETRDecoder, TransposedDETRDecoder


T_scope = Literal['global', 'local']
# T_aggregation = Literal['init', 'answer', 'all']
T_llava_feature = dict[T_scope, torch.Tensor]


def interleave_tensors(z, z_mask, e):
    # z: [B, T1, D], z_mask: [B, T1], e: [B, T2, D]
    B, T1, D = z.shape
    _, T2, _ = e.shape
    tensors = []
    for i in range(B):
        num_valid = z_mask[i].sum().item()
        r = math.ceil(num_valid / T2)
        T_padded = T2 * r
        if T_padded < T1:
            z_i = z[i, :T_padded]
        else:
            z_i = torch.cat([z[i], torch.zeros((T_padded - T1, D), device=z.device)], dim=0)
        z_i = z_i.reshape(T2, r, D)
        z_i = torch.cat([z_i, e[i][:, None]], dim=1)  # [T2, r+1, D]
        z_i = z_i.reshape(-1, D)  # [T2*(r+1), D]
        if T_padded < T1:
            z_i = torch.cat([z_i, z[i, T_padded:]], dim=0)  # [T1+T2, D]
        else:
            z_i = z_i[:T1+T2]
        tensors.append(z_i)
    z = torch.stack(tensors, dim=0)  # [B, T1+T2, D]
    z_mask = torch.cat([torch.ones((B, T2), dtype=z_mask.dtype, device=z.device), z_mask], dim=1)  # [B, T1+T2]
    return z, z_mask


MODEL_VARIANTS = Literal[
    't5',
    't5_ca',
    'input_embed',
    'input_concat',
    'input_interleave'
]
ENV_EXT_VARIANTS = Literal[
    'id',
    'detr',
    't-detr'
]


class GroundVQA(nn.Module):
    def __init__(
        self,
        lm_path,
        input_dim,
        model_variant: MODEL_VARIANTS = 't5',
        t5_ca_layer_idxs = [4],
        freeze_word = False,
        freeze_all_but_ca = False,
        max_v_len = 256,  # lightning module에서 1200 넣어줌
        ignore_decoder = False,
        d_env: int = 384,
        env_ext_variant: ENV_EXT_VARIANTS = 'id',
        num_envs: int = 5,
    ):
        super().__init__()
        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.model_variant = model_variant
        self.env_ext_variant = env_ext_variant

        if model_variant in ['t5']:
            print('Using T5')
            lm: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
                lm_path, local_files_only=True)

        elif model_variant == 't5_ca':
            print('Using T5 with Cross-Attention')
            lm = T5WithCAEncoderAndDecoder.from_pretrained(
                lm_path, cross_attention_layer_idxs=t5_ca_layer_idxs, local_files_only=True)
            self.ca_proj = nn.Linear(d_env, lm.config.d_model)

        elif model_variant in ['input_embed', 'input_interleave']:
            print('Using T5 with Input Embedding')
            lm: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
                lm_path, local_files_only=True)
            self.env_in_proj = nn.Linear(d_env, lm.config.d_model)

        elif model_variant in ['input_concat']:
            print('Using T5 with Input Concat')
            lm: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
                lm_path, local_files_only=True)
            self.env_cat_in_proj = nn.Linear(d_env + lm.config.d_model, lm.config.d_model)
        else:
            raise ValueError(f'Unknown model_variant: {model_variant}')

        self.lm = lm
        self.ignore_decoder = ignore_decoder
        if ignore_decoder:
            self.lm.decoder = None
            self.lm.lm_head = None

        self.lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, self.lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, self.lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=self.lm_dim, max_v_len=max_v_len)

        if env_ext_variant == 'detr':
            self.environment_extractor = DETRDecoder(
                d_model=d_env,
                nhead=8,
                num_layers=3,
                dim_feedforward=4*d_env,
                num_queries=max_v_len
            )
        elif env_ext_variant == 't-detr':
            self.environment_extractor = TransposedDETRDecoder(
                d_model=d_env,
                nhead=8,
                num_layers=3,
                dim_feedforward=4*d_env,
                num_memories=num_envs
            )

        if freeze_all_but_ca:
            assert model_variant == 't5_ca'
            ca_params = find_all_cross_attention_parameters(lm.state_dict(), t5_ca_layer_idxs)
            names = []
            for name, param in lm.named_parameters():
                if name not in ca_params and 'ca_proj' not in name:
                    param.requires_grad = False
                else:
                    names.append(name)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('Paramters to be trained')
                pprint(names)
                print()

    def forward(
        self, v_feat, v_mask, q_token, q_mask, gt_segments, gt_labels,
        # arg names for features are temporary
        env_feat=None,
        labels=None,
        **remains
    ):
        # encoder
        encoder_out, mask = self.forward_encoder(
            v_feat, v_mask, q_token, q_mask,
            env_feat=env_feat
        )

        # localizer
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.

        # decoder
        if self.ignore_decoder:
            return time_loss, 0, time_loss, nlq_results['cls_loss'], nlq_results['reg_loss']
        else:
            outputs = self.lm(
                encoder_outputs=(encoder_out,),
                attention_mask=mask,
                labels=labels,
            )
            lm_loss = outputs.loss
            total_loss = 0.5 * time_loss + 0.5 * lm_loss
            return total_loss, lm_loss, time_loss

    def generate(
        self,
        v_feat, v_mask, q_token, q_mask, v_len,
        env_feat: None|T_llava_feature = None,
        **remains
    ):
        encoder_out, mask = self.forward_encoder(
            v_feat, v_mask, q_token, q_mask,
            env_feat=env_feat
        )
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]

        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            training=False,
            v_lens=v_len
        )
        if self.ignore_decoder:
            answer_tokens = None
        else:
            answer_tokens = self.lm.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                attention_mask=mask,
                max_new_tokens=32
            )

        return nlq_results, answer_tokens

    def forward_encoder(
        self,
        v_feat, v_mask, q_token, q_mask,
        env_feat: None|T_llava_feature = None,
    ):
        B, T_vid, D = v_feat.shape
        v_feat0 = self.lm_proj(v_feat)
        v_feat = v_feat0 + self.v_emb.expand((B, T_vid, -1))  # [B, T_vid, D], video feats as embeddings
        q_feat = self.lm.encoder.embed_tokens(q_token)  # [B, L_q, D], token_embeddings
        lm_input = torch.cat([q_feat, v_feat], dim=1)  # [B, L, D]
        lm_mask = torch.cat([q_mask, v_mask], dim=1)  # [B, L]

        if env_feat is not None and isinstance(env_feat, dict):
            env_feat = env_feat['global']

        if self.model_variant == 't5_ca':
            assert env_feat is not None
            env_feat = self.forward_environment_extractor(env_feat, T_vid=T_vid)
            env_feat = self.ca_proj(env_feat)
            env_mask = torch.ones(B, env_feat.shape[1], device=env_feat.device)
            env_imask = self.lm.invert_attention_mask(env_mask)
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
                encoder_hidden_states=env_feat,
                encoder_attention_mask=env_imask,
            )
        elif self.model_variant == 'input_embed':
            assert env_feat is not None  # [B, T_env=48 or 897, D_env=384 or 128]
            env_feat = self.forward_environment_extractor(env_feat, T_vid=T_vid)
            env_feat = self.env_in_proj(env_feat)  # [B, T_env, D]
            env_feat = self.interp_t(env_feat, T_target=T_vid)  # [B, T_vid, D]
            lm_input = torch.cat([q_feat, v_feat + env_feat], dim=1)  # [B, L, D]
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
            )
        elif self.model_variant == 'input_interleave':
            assert env_feat is not None
            env_feat = self.forward_environment_extractor(env_feat, T_vid=T_vid)
            env_feat = self.env_in_proj(env_feat)  # [B, T_env, D]
            v_feat, v_mask = interleave_tensors(v_feat0, v_mask, env_feat)  # [B, L + T_env, D], [B, L + T_env]
            v_feat = v_feat + self.v_emb.expand((B, v_feat.shape[1], -1))  # [B, L + T_env, D]
            lm_input = torch.cat([q_feat, v_feat], dim=1)  # [B, L + T_env, D]
            lm_mask = torch.cat([q_mask, v_mask], dim=1)  # [B, L + T_env]
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
            )
        elif self.model_variant == 'input_concat':
            assert env_feat is not None
            env_feat = self.forward_environment_extractor(env_feat, T_vid=T_vid)
            # env_feat = self.env_in_proj(env_feat)
            env_feat = self.interp_t(env_feat, T_target=T_vid)
            res = v_feat
            v_feat = torch.cat([v_feat, env_feat], dim=2)  # [B, T_vid, D + D_env]
            v_feat = res + self.env_cat_in_proj(v_feat)  # [B, T_vid, D]
            lm_input = torch.cat([q_feat, v_feat], dim=1)
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
            )
        else:
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
            )
        return out.last_hidden_state, lm_mask

    def forward_environment_extractor(self, z_llava, T_vid=900):
        if self.env_ext_variant == 'id':
            z_env = z_llava
        elif self.env_ext_variant == 'detr':
            z_env = self.environment_extractor(z_llava)  # [B, T_max, D_env]
            z_env = z_env[:, :T_vid]
        elif self.env_ext_variant == 't-detr':
            z_env = self.environment_extractor(z_llava)  # [B, T_max, D_env]
        return z_env

    @staticmethod
    def interp_t(tensor, T_target, mode='nearest'):
        # tensor: [B, T, D] -> [B, T_target, D]
        D, dtype = tensor.shape[-1], tensor.dtype
        return F.interpolate(
            tensor[None].float(),
            size=(T_target, D),
            mode=mode).squeeze(0).to(dtype=dtype)
