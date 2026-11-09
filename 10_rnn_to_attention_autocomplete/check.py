"""Run:  python check.py      (about 1 minute)"""
import numpy as np
import torch

from data import Vocab, corpus
from solution import TinyGPT, make_model, memory_test

text = corpus()
vocab = Vocab(text)

# 1. encode then decode gives back the same text
encoded = vocab.encode("on my way x")
decoded = vocab.decode(encoded)
assert decoded == "on my way x", "encode/decode do not match"
print("OK   encode/decode round trip")

# 2. causal mask: changing the LAST character must not change earlier predictions
torch.manual_seed(0)
gpt = TinyGPT(len(vocab), size=32, max_length=32)
gpt.eval()
original = torch.tensor([list(vocab.encode("are you free for dinner"))])
changed = original.clone()
changed[0, -1] = vocab.stoi["x"]
with torch.no_grad():
    scores_original = gpt(original)
    scores_changed = gpt(changed)
earlier_original = scores_original[0, :-1]
earlier_changed = scores_changed[0, :-1]
assert torch.allclose(earlier_original, earlier_changed, atol=1e-6), "the future leaked backwards"
print("OK   causal mask blocks the future")

# 3. every model turns [batch, length] into [batch, length, vocab]
for kind in ["rnn", "gru", "gpt"]:
    m = make_model(kind, 19, 16, 32)
    out = m(torch.zeros(2, 10, dtype=torch.long))
    assert out.shape == (2, 10, 19), kind + " has the wrong output shape"
print("OK   output shapes")

# 4. attention remembers across a long gap
accuracy, _ = memory_test("gpt", distance=60, steps=800)
print("OK   Tiny GPT memory test at distance 60:", round(accuracy * 100), "%")
assert accuracy > 0.9, "Tiny GPT should remember the code"

print()
print("All checks passed.")
