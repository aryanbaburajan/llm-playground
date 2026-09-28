# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# with open("input.txt") as f:
#   text = f.read()
text = "never gonna give you up"
vocab = "abcdefghijklmnopqrstuvwxyz "

vocab = list(set(vocab))
vocab_size = len(vocab)

import torch
from torch.nn import functional as F
from tqdm import tqdm

from transformer import Transformer
import transformer.transformer as transformer_module

batch_size = 2
d_model = 512
heads = 8
d_ff = 2048
num_layers = 6
src_len = 4
tgt_len = 6
max_seq_len = 1024
src_vocab_size = vocab_size
tgt_vocab_size = vocab_size
temperature = 0.7


target = torch.randint(0, vocab_size, (batch_size, tgt_len))

transformer = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    max_seq_len,
    num_layers,
    heads,
    d_ff,
    decoder_only=True,
)

x = transformer(target)

stoi = {s: i for i,s in enumerate(vocab)}
itos = {i: s for i,s in enumerate(vocab)}

def encode(string):
  return [stoi[c] for c in list(string)]

def decode(tokens):
  return "".join([itos[t] for t in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 22
transformer_module.block_size = block_size
learning_rate = 1e-3
max_iters = 2000
eval_iters = 20
eval_interval = 50

train_data = data

def get_batch():
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformer.to(device)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
  out = {}
  transformer.eval()
  losses = torch.zeros(eval_iters)
  for k in range(eval_iters):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    logits = transformer(x)
    logits = logits.transpose(-2, -1)
    loss = F.cross_entropy(logits, y)
    losses[k] = loss
  out = losses.mean()
  transformer.train()
  return out

pbar = tqdm(range(max_iters))

for steps in pbar:
  if steps % eval_interval == 0:
    loss = estimate_loss()
    print(f"eval {loss:.4f}")
    # pbar.set_description(f"train {loss:.4f}")

  x, y = get_batch()
  x, y = x.to(device), y.to(device)

  logits = transformer(x)
  logits = logits.transpose(-2, -1)
  loss = F.cross_entropy(logits, y)
  print(f"train {loss:.4f}")
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  pbar.set_postfix(loss=loss.item())
