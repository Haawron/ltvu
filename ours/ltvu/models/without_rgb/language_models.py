from typing import Sequence
from collections import namedtuple

import torch
import torch.distributed
import torch.nn as nn
from einops import rearrange, repeat
from transformers import (
    AutoTokenizer, AutoModel,
    PreTrainedTokenizer, PreTrainedTokenizerFast
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer

from ltvu.models.without_rgb.heads import TXActionFormerHead, ActionFormerHead, SimilarityHead
from ltvu.models.egovlpv1.model import TextOnlyFrozenInTime


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


TEXTONLYNLQMODELS = []


class TextOnlyNLQModelBase(nn.Module):
    @classmethod
    def test_me(cls):
        from pprint import pprint
        print(f'\n======== Testing {cls.__name__} ========')
        from ltvu.data_loader.egonlq import EgoNLQRawDataModule
        model = cls('google/flan-t5-base').cuda()
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
        head_kws = dict(),
        **kwargs,
    ):
        super().__init__()
        self.max_ctx_len = max_ctx_len
        if 'egovlp' in model_name.lower():
            pass
        else:
            self.lm = SentenceTransformer(model_name)
            self.lm.forward_ = self.lm.forward
            self.lm.forward = lambda *args, **kwargs: self.lm.forward_(
                *args, **kwargs).sentence_embedding
            self.tokenizer = self.lm.tokenizer
            self.lm_dim = self.lm.get_sentence_embedding_dimension()

        if head_name == 'af':
            self.head = TXActionFormerHead(
                self.lm_dim, max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                **head_kws)
        elif head_name == 'af-notx':
            self.head = ActionFormerHead(
                self.lm_dim, max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                **head_kws)
        elif head_name == 'sim':
            self.head = SimilarityHead(
                self.lm_dim, max_ctx_len, feature_grid_stride_sec=caption_stride_sec,
                pred_width_sec=pred_width_sec,
                **head_kws)

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


class TextOnlyNLQModel(TextOnlyNLQModelBase):
    def forward_lm(
        self,
        captions_tokens: BatchEncoding,  # [B, T, L]
        query_tokens: BatchEncoding,  # [B, L]
        **kwargs
    ) -> LanguageModelOutputDict:
        B = captions_tokens.input_ids.shape[0]
        m_ctx = captions_tokens.attention_mask.any(dim=-1)  # [B, T]
        _caps = captions_tokens.copy()
        # T-padded entries have no non-pad tokens
        # and the attention_mask donesn't have any 1's
        # Language models output NaNs for those entries
        # So, we replace the first token with the UNK token
        # for those T-padded entries
        _caps['input_ids'][..., 0] += ~m_ctx * self.tokenizer.unk_token_id
        _caps['attention_mask'][..., 0] = 1
        _caps['input_ids'] = rearrange(
            _caps.input_ids, 'b t l -> (b t) l')
        _caps['attention_mask'] = rearrange(
            _caps.attention_mask, 'b t l -> (b t) l')
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


TEXTONLYNLQMODELS.append(TextOnlyNLQModel)


class TextOnlyNLQModel2(TextOnlyNLQModelBase):
    def forward_lm(
        self,
        captions_tokens: BatchEncoding,  # [B, T, L]
        query_tokens: BatchEncoding,  # [B, L]
        **kwargs
    ) -> LanguageModelOutputDict:
        B, T, L = captions_tokens.input_ids.shape
        m_ctx = captions_tokens.attention_mask.any(dim=-1)  # [B, T]
        _caps = captions_tokens.copy()
        # _caps['input_ids'][..., 0] += ~m_ctx * self.tokenizer.unk_token_id
        # _caps['attention_mask'][..., 0] = 1
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


TEXTONLYNLQMODELS.append(TextOnlyNLQModel2)


if __name__ == '__main__':
    for cls in TEXTONLYNLQMODELS:
        cls.test_me()
