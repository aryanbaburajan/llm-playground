# transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryanbaburajan/transformer/blob/main/transformer.ipynb)

A from-scratch implementation of the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Transformer, with multi-head attention, sinusoidal positional embeddings, encoder-decoder layers, training, and text generation.

- [x] `TokenEmbedding`: token embedding with the original embedding scale.
- [x] `PositionEmbedding`: sinusoidal positional encoding.
- [x] `Attention`: multi-head attention, including optional masking.
- [x] `FeedForward`: two-layer ReLU feed-forward network.
- [x] `EncoderLayer` and `Encoder`: stacked encoder self-attention and feed-forward layers.
- [x] `DecoderLayer` and `Decoder`: masked decoder self-attention, optional cross-attention, and feed-forward layers.
- [x] `Transformer`: encoder-decoder or decoder-only assembly, output projection, and generation method.
- [ ] `Training`: attain satisfactory training on tinyshakespeare
