import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1200):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DETRDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_queries, dropout=0.1):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, memory, memory_key_padding_mask=None):
        B, L, D = memory.size()
        memory = memory.permute(1, 0, 2)  # [L, B, D]

        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, D]
        memory = self.positional_encoding(memory)
        output = self.transformer_decoder.forward(
            query_pos,
            memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask
        )
        output = self.linear(output)  # [num_queries, B, D]
        output = output.permute(1, 0, 2)  # [B, num_queries, D]
        return output


class TransposedDETRDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_memories=5, dropout=0.1):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model)
        self.memory_embed = nn.Embedding(num_memories, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, context, context_key_padding_mask=None):
        B, L, D = context.size()
        context = context.permute(1, 0, 2)  # [L, B, D]

        memory_pos = self.memory_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [N_mem, B, D]
        context = self.positional_encoding(context)
        output = self.transformer_decoder.forward(
            context,
            memory_pos,
            tgt_key_padding_mask=context_key_padding_mask,
            memory_key_padding_mask=None
        )
        output = self.linear(output)  # [N_mem, B, D]
        output = output.permute(1, 0, 2)  # [B, N_mem, D]
        return output


if __name__ == '__main__':
    def test():
        # Example usage
        d_model = 256
        nhead = 8
        num_layers = 3
        dim_feedforward = 2048
        num_queries = 100

        decoder = DETRDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_queries=num_queries
        )
        print(decoder)

        # Example input tensor (memory) with random values
        # Shape: (batch_size, channels, height, width)
        memory = torch.rand((32, d_model, 20, 20))

        # Forward pass
        output = decoder(memory)
        print(output.shape)  # Expected shape: (num_queries, batch_size, d_model)
    test()
