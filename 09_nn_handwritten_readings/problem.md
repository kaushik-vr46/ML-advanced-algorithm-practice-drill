# 09 — Reading handwritten meter readings

Your landlord's agent photographs the handwritten meter cards in every flat
and someone types the numbers in by hand. It is slow and they get it wrong.

**Read the handwritten digits automatically.**

The images are 8x8 pixel greyscale digits — small, so everything trains in
seconds on a laptop CPU.

---

## Why this problem, and not one of the Tier 1 problems

Every Tier 1 problem was a table: rows of measurements. On tables, gradient
boosting usually wins and a neural network is the wrong tool.

An image is different. Pixel 27 only means something because of where it sits
next to pixels 26, 28, 19 and 35. That **spatial structure** is what a
convolutional neural network is built to exploit — and what a tree model
throws away.

The final experiment proves it: shuffle the pixel order (identically for
every image) and see which models care.

## The algorithm: a neural network

Layers of `output = activation(input @ W + b)`, stacked. Trained exactly like
your linear regression in Tier 1 — gradient descent — except the gradient has
to flow backwards through several layers. That backwards flow is
**backpropagation**, and it is just the chain rule applied repeatedly.

## The solution

`solution.py` builds the network in **PyTorch**, then a convolutional
network, compared honestly against LightGBM.

## What you will learn

- Forward pass, backward pass, and why they mirror each other
- Softmax and cross-entropy for more than two classes
- Mini-batch training
- **Gradient checking** — how to know your backprop is correct
- PyTorch's core pattern: `nn.Module`, `DataLoader`, `zero_grad` /
  `backward` / `step`, `train()` vs `eval()`
- When a neural network actually beats a tree model, and when it does not

## Why it matters for your projects

Embeddings — the vectors at the heart of your RAG project — are the output of
a neural network. The two-tower model in RecSys project 01 is two of these
networks side by side. After this exercise, both stop being black boxes.

## Time: 90 minutes — the longest exercise in either tier
