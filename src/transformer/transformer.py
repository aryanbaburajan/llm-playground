import torch
import torch.nn as nn
from torch.nn import functional as F

from .decoder import Decoder
from .encoder import Encoder
from .positional_encoding import PositionEmbedding
from .utils import TokenEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        max_seq_len,
        num_layers,
        heads,
        d_ff,
        decoder_only=False,
    ):
        super().__init__()

        self.decoder_only = decoder_only

        if not decoder_only:
            self.src_token_embedding = TokenEmbedding(
                src_vocab_size,
                d_model,
            )
            self.encoder = Encoder(
                num_layers,
                d_model,
                heads,
                d_ff,
            )

        self.tgt_token_embedding = TokenEmbedding(
            tgt_vocab_size,
            d_model,
        )
        self.position_embedding = PositionEmbedding(
            max_seq_len,
            d_model,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            heads,
            d_ff,
            max_seq_len,
            decoder_only,
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, target, source=None):
        assert self.decoder_only is (source is None)

        if not self.decoder_only:
            x = self.src_token_embedding(source)
            x = self.position_embedding(x)
            memory = self.encoder(x)
        else:
            memory = None

        x = self.tgt_token_embedding(target)
        x = self.position_embedding(x)
        x = self.decoder(x, memory)
        x = self.output_proj(x)

        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            # logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
