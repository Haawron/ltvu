import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerCrossAttention
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class T5EncoderWithCrossAttention(T5Stack):
    def __init__(self, config, embed_tokens, cross_attention_layers):
        super().__init__(config, embed_tokens)
        self.cross_attention_layers = cross_attention_layers
        self.cross_attention = nn.ModuleList([T5LayerCrossAttention(config) for _ in cross_attention_layers])
        # TODO: CA 대신 T5 Block 이용

    def invert_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_length)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1(valid) -> 0., 0(pad) -> large negative
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        key_value_states=None,
        key_value_states_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        if key_value_states is not None and key_value_states_attention_mask is None:
            key_value_states_attention_mask = torch.ones(key_value_states.size()[:2], device=key_value_states.device)
        if key_value_states_attention_mask is not None:
            key_value_states_attention_mask = self.invert_attention_mask(key_value_states_attention_mask)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
                **kwargs)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if i in self.cross_attention_layers:
                index = self.cross_attention_layers.index(i)
                cross_attention_outputs = self.cross_attention[index](
                    hidden_states,
                    key_value_states=key_value_states,
                    attention_mask=key_value_states_attention_mask)
                hidden_states = cross_attention_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

class T5WithCAEncoderAndDecoder(T5ForConditionalGeneration):
    def __init__(self, config, cross_attention_layers):
        super().__init__(config)
        self.encoder = T5EncoderWithCrossAttention(config, self.shared, cross_attention_layers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        encoder_key_value_states=None,
        key_value_states_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if encoder_outputs are not given
        if encoder_outputs is None:
            encoder_outputs = self.encoder.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                key_value_states=encoder_key_value_states,
                key_value_states_attention_mask=key_value_states_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Decode
        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # Compute the logits
        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)

        if not return_dict:
            return (logits,) + decoder_outputs[1:] + encoder_outputs

        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


if __name__ == '__main__':
    def test():
        print('Testing T5EncoderWithCrossAttention ...')
        cross_attention_layers = [2, 5]
        model = T5WithCAEncoderAndDecoder.from_pretrained(
            'google/flan-t5-base', cross_attention_layers, local_files_only=True)
        config = model.config
        print(model)

        B, L, T, D = 4, 22, 128, config.d_model
        q_token = torch.randint(0, config.vocab_size, (B, L))
        q_feat = model.encoder.embed_tokens(q_token)
        v_feat = torch.randn(B, T, D)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.ones(B, L + T)
        key_value_states = torch.randn(B, 48, D)
        key_value_states_attention_mask = torch.ones(B, 48)

        print(lm_input.shape)
        print(lm_mask.shape)
        out = model.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask,
            key_value_states=key_value_states,
            key_value_states_attention_mask=key_value_states_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        print(len(out.hidden_states))
        print(len(out.attentions))

        answer_tokens = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=out),
            attention_mask=lm_mask,
            max_new_tokens=32
        )
        print(answer_tokens)

        # max_seq_len = 10
        # ca_seq_len = 20
        # input_ids = torch.randint(0, config.vocab_size, (1, max_seq_len))
        # decoder_input_ids = torch.randint(0, config.vocab_size, (1, max_seq_len))
        # z_k = torch.randn(1, ca_seq_len, config.d_model)
        # z_v = torch.randn(1, ca_seq_len, config.d_model)

        # outputs = model(
        #     input_ids=input_ids,
        #     decoder_input_ids=decoder_input_ids,
        #     z_k=z_k,
        #     z_v=z_v,
        #     output_attentions=True,
        #     output_hidden_states=True,
        #     return_dict=True
        # )

        # print(outputs.logits)
        # print(outputs.encoder_hidden_states)
        # print(outputs.decoder_hidden_states)

        # loss = outputs.logits.mean()
        # loss.backward()
        # print(loss)
        print('Testing T5EncoderWithCrossAttention ... Done')

    test()
