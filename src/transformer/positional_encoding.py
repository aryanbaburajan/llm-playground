import math

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        self.max_seq_len = max_seq_len

        position_encoding = torch.empty(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(d_model // 2):
                angle = pos / math.pow(10000, (2 * i) / d_model)
                position_encoding[pos, 2 * i] = math.sin(angle)
                position_encoding[pos, 2 * i + 1] = math.cos(angle)

        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x):
        _, seq_len, _ = x.shape

        assert seq_len <= self.max_seq_len

        return x + self.position_encoding[:seq_len]
