import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerCrossAttention
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class T5WithCAEncoderAndDecoder(T5ForConditionalGeneration):
    def __init__(self, config, cross_attention_layer_idxs):
        super().__init__(config)
        self.cross_attention_layer_idxs = cross_attention_layer_idxs
        assert 0 <= min(cross_attention_layer_idxs)
        assert max(cross_attention_layer_idxs) < self.config.num_layers
        for i in cross_attention_layer_idxs:
            self.encoder.block[i].layer.insert(1, T5LayerCrossAttention(config))
            self.encoder.block[i].is_decoder = True


if __name__ == '__main__':
    def test():
        import lightning as pl
        pl.seed_everything()

        print('Testing T5EncoderWithCrossAttention ...')
        cross_attention_layer_idxs = [2, 5]
        model = T5WithCAEncoderAndDecoder.from_pretrained(
            'google/flan-t5-base', cross_attention_layer_idxs, local_files_only=True)
        config = model.config
        print(model)

        B, L, T, D = 4, 22, 128, config.d_model
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
        q_token = tokenizer(
            ['Hello, my dog is cute'] * B,
            padding='max_length',
            max_length=L,
            return_tensors='pt',
        )['input_ids']
        q_feat = model.encoder.embed_tokens(q_token)  # [B, L, D]
        v_feat = torch.randn(B, T, D)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.ones(B, L + T)
        # Setting requires_grad=True to test it's properly propagated
        key_value_states = torch.randn(B, 48, D, requires_grad=True)
        key_value_states_attention_mask = torch.zeros(B, 48)

        print(lm_input.shape)
        print(lm_mask.shape)
        out = model.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask,
            encoder_hidden_states=key_value_states,
            encoder_attention_mask=key_value_states_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        print(len(out.hidden_states))
        print(len(out.attentions))
        for i in range(len(out.hidden_states)):
            if i > 0:
                print(f'{i:2d}-th attention {list(out.attentions[i-1].shape)}')
            print(
                f'{i:2d}-th hidden    {list(out.hidden_states[i].shape)}'
                + (' (embed)' if i == 0 else ''))

        print('Testing backward')
        out.last_hidden_state.sum().backward()
        grad = key_value_states.grad
        print(grad); assert grad is not None
        print('Passed!')

        answer_tokens = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=out),
            attention_mask=lm_mask,
            max_new_tokens=32
        )
        print(answer_tokens)

        # ignore encoder_hidden_states and watch whether the output differs
        out2 = model.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        print(out.hidden_states[-1])
        print(out2.hidden_states[-1])
        assert not torch.allclose(out.hidden_states[-1], out2.hidden_states[-1])

        print('All passed!')
    test()
