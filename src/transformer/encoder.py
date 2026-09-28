from typing import Literal

import torch.nn as nn

from .attention import Attention, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, norm: Literal["pre", "post"] = "post"):
        super().__init__()
        assert(norm == "pre" or norm == "post")

        self.norm = norm

        self.sa = Attention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.norm == "post":
          x = self.ln1(x + self.sa(x, x))
          x = self.ln2(x + self.ff(x))
        elif self.norm == "pre":
          x_norm = self.ln1(x)
          x = x + self.sa(x_norm, x_norm)

          x_norm = self.ln2(x)
          x = x + self.ff(x_norm)

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff):
        super().__init__()

        self.layers = nn.ModuleList(
            EncoderLayer(d_model, heads, d_ff)
            for _ in range(num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
