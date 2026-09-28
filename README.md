# Attention Is All You Need

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryanbaburajan/transformer/blob/main/notebooks/transformer_demo.ipynb)

A from-scratch implementation of the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Transformer, with multi-head attention, sinusoidal positional embeddings, encoder-decoder layers, training, and text generation.

## Implemented components

- [x] `TokenEmbedding` — token embedding with the original embedding scale.
- [x] `PositionEmbedding` — sinusoidal positional encoding.
- [x] `Attention` — multi-head attention, including optional masking.
- [x] `FeedForward` — two-layer ReLU feed-forward network.
- [x] `EncoderLayer` and `Encoder` — stacked encoder self-attention and feed-forward layers.
- [x] `DecoderLayer` and `Decoder` — masked decoder self-attention, optional cross-attention, and feed-forward layers.
- [x] `Transformer` — encoder-decoder or decoder-only assembly, output projection, and generation method.

## Layout

- `src/transformer/`: extracted model implementation.
- `scripts/train.py`: the notebook's setup and training cells in their original order.
- `scripts/evaluate.py`: the notebook's generation cell. The original workflow has no checkpointing, so this imports and runs `train.py` to obtain its in-memory model.
- `notebooks/transformer_demo.ipynb`: the original experimental workflow using the extracted package.

## Run

```bash
pip install -e .
python scripts/train.py
```

Run generation with `python scripts/evaluate.py`. This follows the notebook's in-memory workflow and consequently runs training first.

## Google Colab

Clone this repository in a Colab cell, change into its directory, and run `pip install -e .`. The first cell of `notebooks/transformer_demo.ipynb` includes commented setup commands with a repository URL placeholder; replace it with your repository URL before running it in Colab.
