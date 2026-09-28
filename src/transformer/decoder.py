import torch
import torch.nn as nn

from .attention import Attention, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, max_seq_len, decoder_only=False):
        super().__init__()

        self.decoder_only = decoder_only

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)) == 0,
        )

        self.sa = Attention(d_model, heads)
        self.ln1 = nn.LayerNorm(d_model)

        if not decoder_only:
            self.ca = Attention(d_model, heads)
            self.ln2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, memory=None):
        x = self.ln1(x + self.sa(x, x, self.mask))

        if not self.decoder_only and memory is not None:
            x = self.ln2(x + self.ca(x, memory))

        x = self.ln3(x + self.ff(x))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        max_seq_len,
        decoder_only=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model,
                heads,
                d_ff,
                max_seq_len,
                decoder_only,
            )
            for _ in range(num_layers)
        )

    def forward(self, x, memory=None):
        for layer in self.layers:
            x = layer(x, memory)

        return x
