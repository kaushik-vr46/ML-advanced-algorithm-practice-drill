# 10 — Phone autocomplete: from RNN to attention

Your phone guesses the next letter as you type. That is **next-token
prediction** — exactly the objective every LLM is trained on. The only
difference is scale.

This exercise walks the actual history of the idea:

    RNN  ->  why it fails on long text  ->  attention  ->  a tiny GPT

---

## Part 1 — the RNN

A recurrent network reads one character at a time and carries a **hidden
state**: a fixed-size vector summarising everything it has read so far.

    h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + b)
    prediction_t = softmax(h_t @ Why + c)

Everything the model remembers must squeeze through that one vector, and
gradients must flow backwards through every step to train it. Both are
problems, and you will measure both.

## Part 2 — the problem: vanishing gradients

Backprop through time multiplies by `Whh` once per step. Over 30 steps the
gradient reaching early characters shrinks toward zero, so the RNN cannot
learn that something early matters to something late. You will plot this.

## Part 3 — the fix that came next: gates (GRU)

GRUs and LSTMs add gates that let information pass through steps almost
unchanged. Better — but still one step at a time, still one fixed vector.

## Part 4 — the fix that won: attention

Instead of squeezing the past into one vector, let every position **look
directly** at every earlier position and decide what to pull in:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

Character 60 can read character 1 in a single step. No chain of
multiplications, so nothing vanishes. Add a causal mask so no position can
see the future, and you have the core of GPT.

## The solution

`solution.py` uses PyTorch: an RNN, a GRU and a hand-written one-block
GPT, trained on the same text, then a memory test that shows exactly why
transformers replaced RNNs.

## Why this matters for your projects

Your RAG project calls an LLM. After this exercise, "tokens", "context
window", "temperature", "causal attention" and "next-token prediction" stop
being vocabulary and become things you have built.

## Time: 2 hours. The longest exercise you have done, and the one that
connects everything to LLMs.
