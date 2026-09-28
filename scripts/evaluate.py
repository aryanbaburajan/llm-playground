# The original notebook has no checkpoint save/load step. Importing train therefore
# preserves its original execution order: training creates the in-memory model used below.
from train import decode, device, encode, torch, transformer

t = "never "
x = torch.tensor([encode(t)], dtype=torch.long).to(device)
y = transformer.generate(x, max_new_tokens=30)
print(decode(y.tolist()[0]))
