from typing import Sequence
from collections import namedtuple

import torch
import torch.distributed
import torch.nn as nn
from einops import rearrange, repeat
from sentence_transformers import SentenceTransformer, CrossEncoder

from ltvu.models.without_rgb.heads import TXActionFormerHead, ActionFormerHead, SimilarityHead
from ltvu.models.egovlpv1.model import TextOnlyFrozenInTime

# for type hinting
from sentence_transformers.models import Pooling
from transformers import (
    PreTrainedModel
)
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Stack
from transformers.models.mpnet.modeling_mpnet import MPNetEmbeddings, MPNetEncoder
from transformers.tokenization_utils_base import BatchEncoding


InputBatchDict = namedtuple('BatchInput', [
    'captions_length',
    'query_tokens',
    'captions_tokens',
    'caption_start_secs', 'caption_end_secs',
    'gt_start_sec', 'gt_end_sec',
])

LanguageModelOutputDict = namedtuple('LanguageModelOutputDict', ['z_ctx', 'z_ctx_', 'z_q'])
HeadOutputDict = namedtuple('HeadOutputDict', ['logits', 'preds'])  # preds: records, {'segments', 'scores'}, FIXME: 학습 중엔 loss 포함
# LossDict = namedtuple('LossDict', ['loss'])
OutputDict = namedtuple('OutputDict',
    LanguageModelOutputDict._fields + HeadOutputDict._fields# + LossDict._fields
)


class SentenceTransformerWrapper(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self[0].auto_model, 'pooler'):
            self[0].auto_model.pooler.requires_grad_(False)
        self.am_i_t5 = isinstance(self[0].auto_model, T5EncoderModel)
        # self.m: PreTrainedModel = self[0].auto_model
        # self.encoder: T5Stack|MPNetEncoder = self.m.encoder
        # self.p: Pooling = self[1]

    def forward(self, tokens: BatchEncoding):
        """
        Args:
            tokens (BatchEncoding):
                input_ids: [B, L]
                attention_mask: [B, L]
        """
        e = self.forward_embeddings(tokens)
        z = self.forward_encoder_and_pooler(e, tokens.attention_mask)
        return z

    def forward_embeddings(self, tokens: BatchEncoding):
        if self.am_i_t5:
            e = self[0].auto_model.encoder.embed_tokens(tokens.input_ids)
        else:
            embeddings: MPNetEmbeddings = self[0].auto_model.embeddings
            e = embeddings.forward(**tokens)  # [B, L, D]
        return e

    def forward_encoder_and_pooler(self, e, mask):
        if self.am_i_t5:
            z = self[0].auto_model.encoder.forward(
                inputs_embeds=e,
                attention_mask=mask,
            ).last_hidden_state  # [B, L, D]
            z = self[1].forward({
                'token_embeddings': z,
                'attention_mask': mask,
            })
        else:
            emask = self[0].auto_model.get_extended_attention_mask(mask, e.shape[:2])
            z, = self[0].auto_model.encoder.forward(
                hidden_states=e,
                attention_mask=emask,
                head_mask=[None]*self[0].auto_model.config.num_hidden_layers)
            z = self[1].forward({
                'token_embeddings': z,
                'attention_mask': emask,
            })
        return z['sentence_embedding']  # [B, D]


TEXTONLYNLQMODELS = []


class TextOnlyNLQModelBase(nn.Module):
    @classmethod
    def test_me(cls):
        from pprint import pprint
        print(f'\n======== Testing {cls.__name__} ========')
        from ltvu.data_loader.egonlq import EgoNLQRawDataModule
        model = cls('multi-qa-mpnet-base-dot-v1').cuda()
        dm = EgoNLQRawDataModule(tokenizer=model.tokenizer, batch_size=2)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        batch['captions_tokens'].to(model.device)
        batch['query_tokens'].to(model.device)
        batch['gt_segment'] = batch['gt_segment'].to(model.device)
        output = model.forward(batch)
        pprint(output)
        output['loss'].backward()
        print('\n======== Done ========\n\n')

    def __init__(
        self,
        model_name: str,
        head_name: str = 'af',
        max_ctx_len = 256,
        caption_stride_sec = 2,
        pred_width_sec = 30.,
        lm_kws = dict(),
        head_kws = dict(),
        **kwargs,
    ):
        super().__init__()
        self.max_ctx_len = max_ctx_len
        if 'egovlp' in model_name.lower():
            pass
        else:
            self.lm = SentenceTransformerWrapper(model_name)
            self.tokenizer = self.lm.tokenizer
            self.lm_dim = self.lm.get_sentence_embedding_dimension()

        if head_name == 'af':
            self.head = TXActionFormerHead(
                self.lm_dim, max_ctx_len=max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                **head_kws)
        elif head_name == 'af-notx':
            self.head = ActionFormerHead(
                self.lm_dim, max_ctx_len=max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                **head_kws)
        elif head_name == 'sim':
            self.head = SimilarityHead(
                self.lm_dim, max_ctx_len=max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                pred_width_sec=pred_width_sec,
                **head_kws)

        print(f'Using {self.__class__.__name__} with {model_name} and {head_name}')

    @property
    def device(self) -> torch.device:
        return next(self.lm.parameters()).device

    def encode(self, sentences: Sequence[str]) -> LanguageModelOutputDict:
        pass

    def forward_lm(self, **kwargs) -> LanguageModelOutputDict:
        pass

    def forward_head(self, **kwargs) -> HeadOutputDict:
        pass

    def forward(self, batch: InputBatchDict, **kwargs) -> OutputDict:
        output_lm = self.forward_lm(**batch)
        output_head = self.forward_head(**output_lm, **batch)
        return output_lm | output_head


class TimeEmbedding(nn.Module):
    def __init__(self, max_ctx_len, lm_dim, dropout=.1, layer_norm_eps=1e-5):
        super().__init__()
        self.time_embed = nn.Parameter(
            torch.randn(max_ctx_len, lm_dim))
        self.norm = nn.LayerNorm(lm_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_ctx, m_ctx):
        # m_ctx: [B, T]
        B, T, L, D = e_ctx.shape
        time_embed = repeat(self.time_embed, 't d -> b t l d', b=B, l=L)
        time_embed = time_embed * m_ctx[..., None, None].float()
        e_ctx = e_ctx + time_embed
        e_ctx = self.norm(e_ctx)
        e_ctx = self.dropout(e_ctx)
        return e_ctx  # [B, T, L, D]


class TextOnlySentenceNLQModel(TextOnlyNLQModelBase):
    def __init__(self, *args, enable_time_embed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            config = self.lm[0].auto_model.config
            self.time_embed = TimeEmbedding(
                self.max_ctx_len, self.lm_dim,
                dropout=config.hidden_dropout_prob,
                layer_norm_eps=config.layer_norm_eps)
        else:
            self.time_embed = nn.Identity()

    def forward_lm(
        self,
        captions_tokens: BatchEncoding,  # [B, T, L]
        query_tokens: BatchEncoding,  # [B, L]
        **kwargs
    ) -> LanguageModelOutputDict:
        B, T, L = captions_tokens.input_ids.shape
        m_ctx = captions_tokens.attention_mask.any(dim=-1)  # [B, T]
        _caps = captions_tokens.copy()
        # T-padded entries have no non-pad tokens
        # and the attention_mask donesn't have any 1's
        # Language models output NaNs for those entries
        # So, we replace the first token with the UNK token
        # for those T-padded entries
        _caps['input_ids'][..., 0] += ~m_ctx * self.tokenizer.unk_token_id
        _caps['attention_mask'][..., 0] = 1
        _caps['input_ids'] = rearrange(_caps.input_ids, 'b t l -> (b t) l')
        _caps['attention_mask'] = rearrange(_caps.attention_mask, 'b t l -> (b t) l')
        if self.enable_time_embed:
            e_ctx = self.lm.forward_embeddings(_caps)  # [B x T, L, D]
            e_ctx = rearrange(e_ctx, '(b t) l d -> b t l d', b=B)
            e_ctx = self.time_embed(e_ctx, m_ctx)
            e_ctx = rearrange(e_ctx, 'b t l d -> (b t) l d')
            z_caps = self.lm.forward_encoder_and_pooler(e_ctx, _caps['attention_mask'])  # [B x T, D]
        else:
            z_caps = self.lm.forward(_caps)  # [B x T, D]
        z_query = self.lm.forward(query_tokens)  # [B, D]
        z_caps = rearrange(z_caps, '(b t) d -> b t d', b=B)
        return {
            'z_ctx': z_caps,  # [B, T, D]
            'm_ctx': m_ctx,  # [B, T]
            'z_q': z_query,  # [B, D]
        }

    def forward_head(
        self,
        z_ctx,  # [B, T, D]
        m_ctx,  # [B, T]
        z_q,  # [B, D]
        gt_segment=None,  # [B, 2]
        **kwargs
    ) -> HeadOutputDict:
        return self.head.forward(
            z_ctx=z_ctx,
            m_ctx=m_ctx,
            z_q=z_q,
            gt_segment=gt_segment,
            **kwargs)


TEXTONLYNLQMODELS.append(TextOnlySentenceNLQModel)


class TextOnlyCrossNLQModel(TextOnlyNLQModelBase):
    def forward_lm(
        self,
        captions_tokens: BatchEncoding,  # [B, T, L]
        query_tokens: BatchEncoding,  # [B, L]
        **kwargs
    ) -> LanguageModelOutputDict:
        B, T, L = captions_tokens.input_ids.shape
        m_ctx = captions_tokens.attention_mask.any(dim=-1)  # [B, T]
        _caps = captions_tokens.copy()
        for token_key in ['input_ids', 'attention_mask']:
            _caps[token_key] = torch.cat([
                rearrange(_caps[token_key], 'b t l -> (b t) l'),
                repeat(query_tokens[token_key], 'b l -> (b t) l', t=T)],
                dim=1)  # [B x T, 2L]
        z_caps = self.lm.forward(_caps)  # [B x T, D]
        z_caps = rearrange(z_caps, '(b t) d -> b t d', b=B)
        return {
            'z_ctx': z_caps,  # [B, T, D]
            'm_ctx': m_ctx,  # [B, T]
            'z_q': z_caps[:, 0],  # [B, D], TODO: 필요 없긴 함
        }

    def forward_head(
        self,
        z_ctx,  # [B, T, D]
        m_ctx,  # [B, T]
        z_q,  # [B, D]
        gt_segment=None,  # [B, 2]
        **kwargs
    ) -> HeadOutputDict:
        return self.head.forward(
            z_ctx=z_ctx,
            m_ctx=m_ctx,
            z_q=z_q,
            gt_segment=gt_segment,
            **kwargs)


TEXTONLYNLQMODELS.append(TextOnlyCrossNLQModel)


if __name__ == '__main__':
    for cls in TEXTONLYNLQMODELS:
        cls.test_me()
