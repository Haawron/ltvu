from typing import Literal

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead
from model.ours.ltvu.t5_with_ca_encoder import T5WithCAEncoderAndDecoder


T_scope = Literal['global', 'local']
# T_aggregation = Literal['init', 'answer', 'all']
T_llava_feature = dict[T_scope, torch.Tensor]


class GroundVQA(nn.Module):
    def __init__(
        self,
        lm_path,
        input_dim,
        model_variant: Literal['t5', 't5_ca'] = 't5',
        t5_ca_layer_idxs = [4],
        freeze_word = False,
        max_v_len = 256,
        ignore_decoder = False
    ):
        super().__init__()
        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        # TODO: Input concat/interleave/CA
        if model_variant == 't5_ca':
            print('Using T5 with Cross-Attention')
            model_class = T5WithCAEncoderAndDecoder
            model_args = [t5_ca_layer_idxs]
        else:
            print('Using T5')
            model_class = T5ForConditionalGeneration
            model_args = []
        # config = T5Config.from_pretrained(lm_path, local_files_only=True)
        self.lm: T5ForConditionalGeneration = model_class.from_pretrained(lm_path, *model_args, local_files_only=True)
        # self.lm: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)
        config = self.lm.config

        if model_variant == 't5_ca':
            self.ca_proj = nn.Linear(7168, config.d_model)   # FIXME: CA-dim hard-coded

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

    def forward(
        self, v_feat, v_mask, q_token, q_mask, gt_segments, gt_labels,
        # arg names for features are temporary
        llava_feat=None,
        labels=None,
    **remains
    ):
        # encoder
        encoder_out, mask = self.forward_encoder(
            v_feat, v_mask, q_token, q_mask,
            llava_feat=llava_feat
        )

        # localizer
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.0

        # decoder
        if self.ignore_decoder:
            return time_loss, 0, time_loss
        else:
            outputs = self.lm(
                encoder_outputs=(encoder_out,),
                attention_mask=mask,
                labels=labels,
            )
            lm_loss = outputs.loss
            total_loss = 0.5 * time_loss + 0.5 * lm_loss
            return total_loss, lm_loss, time_loss

    def generate(self, v_feat, v_mask, q_token, q_mask, v_len, **remains):
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
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
        llava_feat: None|T_llava_feature = None,
    ):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))  # [B, L_vid, D], video feats as embeddings
        q_feat = self.lm.encoder.embed_tokens(q_token)  # [B, L_q, D], token_embeddings
        lm_input = torch.cat([q_feat, v_feat], dim=1)  # [B, L, D]
        lm_mask = torch.cat([q_mask, v_mask], dim=1)  # [B, L]
        if llava_feat is not None:
            llava_feat_global = self.ca_proj(llava_feat['global'])
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
                encoder_hidden_states=llava_feat_global,
                # encoder_attention_mask=None,  # handled by the model
            )
        else:
            out = self.lm.encoder(
                inputs_embeds=lm_input,
                attention_mask=lm_mask,
            )
        return out.last_hidden_state, lm_mask
