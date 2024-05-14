import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicTransformerEncoder(nn.Module):
    def __init__(self,
        d_input,
        d_output,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = 'gelu',
        num_layers: int = 6,
        dropout: float = 0.1,
        droppath: float = 0.1,
        prenorm: bool = True,
        max_len: int = 14400,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model

        self.in_proj = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                drop_path_rate=droppath, batch_first=True, norm_first=prenorm),
            num_layers,
            nn.LayerNorm(d_model),
        )
        self.out_proj = nn.Linear(d_input, d_output)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0)

    def forward(self, x, mask=None, padding_mask=None):
        x = self.in_proj(x) * math.sqrt(self.d_input)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask, padding_mask)
        output = self.out_proj(output)
        return output


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, drop_path_rate=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        branch = self.self_attn.forward(
            x, x, x, is_causal=is_causal,
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.drop_path(self.dropout1(branch))
        branch = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.drop_path(self.dropout2(branch))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=(1 << 14)):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(max_len) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DropPath(nn.Module):
    def __init__(self, p_drop: float = 0.):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.p_drop < 1e-12 or not self.training:
            return x
        p_keep = 1 - self.p_drop
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, ..., 1]
        random_tensor = p_keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(p_keep) * random_tensor
        return output


if __name__ == '__main__':
    d_model = 256
    ctx_len = 128
    x = torch.randn(2, ctx_len, d_model).cuda()
    y = torch.randn(2, d_model).cuda()
    model = BasicTransformerEncoder(
        d_input=d_model, d_output=d_model, d_model=d_model, dim_feedforward=2*d_model,
        nhead=8, num_layers=2, dropout=.0, droppath=.0,
        activation='gelu', prenorm=True, max_len=ctx_len
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for i in range(500):
        optimizer.zero_grad()
        output = model(x).mean(dim=1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'loss: {loss.item():9.7f}', flush=True)
