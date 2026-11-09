"""Run:  python check.py
Checks that the PyTorch network learns and that eval mode works."""
import numpy as np
import torch

from data import load

X_train, X_val, X_test, y_train, y_val, y_test = load()

# 1. the PyTorch MLP must learn
from solution import MLP, accuracy, make_loader, train
torch.manual_seed(0)
mlp = MLP()
train(mlp, make_loader(X_train, y_train, shuffle=True), make_loader(X_val, y_val), epochs=40)
torch_accuracy = accuracy(mlp, make_loader(X_test, y_test))
print("OK   PyTorch MLP test accuracy", round(torch_accuracy * 100, 1), "%")
assert torch_accuracy > 0.92, "should be above 92%"

# 2. eval mode must switch dropout off (same input -> same output)
mlp.eval()
with torch.no_grad():
    x = torch.tensor(X_test[:20])
    first = mlp(x)
    second = mlp(x)
assert torch.equal(first, second), "outputs differ in eval mode; is dropout still on?"
print("OK   eval() mode gives the same output every time")

print()
print("All checks passed.")
