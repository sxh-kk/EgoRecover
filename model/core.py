import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None):
        # mask B x 1 x 1 x N denotes which elements of kv_x are valid

        x = self.norm(x)
        if mask is not None:
            mask = mask.squeeze([1, 2]).bool().logical_not()
        out = self.attn(x, kv_x, kv_x, key_padding_mask=mask)[0]
        out = self.out_dropout(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x, context, mask=None, context_mask=None):
        x = self.attn1(x, kv_x=x, mask=mask) + x
        x = self.attn2(x, kv_x=context, mask=context_mask) + x
        x = self.ff(x) + x
        return x
