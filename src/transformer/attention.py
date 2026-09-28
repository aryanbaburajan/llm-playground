import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        assert d_model % heads == 0

        self.heads = heads
        self.head_dim = d_model // heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, query_input, key_value_input, mask=None):
        B, Q_T, d_model = query_input.shape
        _, KV_T, _ = key_value_input.shape

        q = self.query(query_input)
        k = self.key(key_value_input)
        v = self.value(key_value_input)

        q = q.view(B, Q_T, self.heads, self.head_dim)
        k = k.view(B, KV_T, self.heads, self.head_dim)
        v = v.view(B, KV_T, self.heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = q @ k.transpose(-2, -1)
        wei = wei / math.sqrt(self.head_dim)

        if mask is not None:
            wei = wei.masked_fill(mask[:Q_T, :KV_T], float("-inf"))

        wei = self.softmax(wei)

        x = wei @ v
        x = x.transpose(1, 2)
        x = x.reshape(B, Q_T, d_model)

        return self.concat(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)
