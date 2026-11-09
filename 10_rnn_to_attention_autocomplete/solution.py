import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Vocab, corpus, recall_task


class RecurrentModel(nn.Module):

    def __init__(self, vocab_size, size=64, cell_type="rnn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, size)
        if cell_type == "rnn":
            self.recurrent = nn.RNN(size, size, batch_first=True)
        else:
            self.recurrent = nn.GRU(size, size, batch_first=True)
        self.output = nn.Linear(size, vocab_size)

    def forward(self, ids):
        x = self.embed(ids)
        memories, last_memory = self.recurrent(x)
        scores = self.output(memories)
        return scores


class SelfAttention(nn.Module):

    def __init__(self, size, max_length):
        super().__init__()
        self.query = nn.Linear(size, size, bias=False)
        self.key = nn.Linear(size, size, bias=False)
        self.value = nn.Linear(size, size, bias=False)
        self.output = nn.Linear(size, size)

        mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        self.register_buffer("mask", mask)
        self.last_weights = None

    def forward(self, x):
        length = x.shape[1]
        size = x.shape[2]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        k_flipped = k.transpose(1, 2)
        scores = q @ k_flipped / math.sqrt(size)

        mask = self.mask[:length, :length]
        future = ~mask
        scores = scores.masked_fill(future, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()

        mixed = weights @ v
        return self.output(mixed)


class TinyGPT(nn.Module):

    def __init__(self, vocab_size, size=64, max_length=128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, size)
        self.position_embed = nn.Embedding(max_length, size)

        self.norm1 = nn.LayerNorm(size)
        self.attention = SelfAttention(size, max_length)
        self.norm2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.GELU(),
            nn.Linear(4 * size, size),
        )
        self.norm_final = nn.LayerNorm(size)
        self.output = nn.Linear(size, vocab_size)

    def forward(self, ids):
        length = ids.shape[1]
        positions = torch.arange(length)

        x = self.char_embed(ids) + self.position_embed(positions)

        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = self.norm_final(x)
        return self.output(x)


def make_model(kind, vocab_size, size=64, max_length=128):
    if kind == "gpt":
        return TinyGPT(vocab_size, size, max_length)
    return RecurrentModel(vocab_size, size, cell_type=kind)


def random_batch(data, seq_len, batch_size, rng):
    xs = []
    ys = []
    for _ in range(batch_size):
        start = rng.integers(0, len(data) - seq_len - 1)
        xs.append(data[start:start + seq_len])
        ys.append(data[start + 1:start + seq_len + 1])
    x = torch.tensor(np.array(xs))
    y = torch.tensor(np.array(ys))
    return x, y


def text_loss(model, x, y):
    scores = model(x)
    vocab_size = scores.shape[-1]
    scores_flat = scores.reshape(-1, vocab_size)
    y_flat = y.reshape(-1)
    return F.cross_entropy(scores_flat, y_flat)


def validation_loss(model, data, seq_len=64):
    model.eval()
    rng = np.random.default_rng(123)
    x, y = random_batch(data, seq_len, 64, rng)
    with torch.no_grad():
        loss = text_loss(model, x, y)
    return loss.item()


def train_text_model(model, train_data, val_data, steps=1500, seq_len=64, batch_size=32):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    steps_logged = []
    losses_logged = []
    for step in range(steps):
        model.train()
        x, y = random_batch(train_data, seq_len, batch_size, rng)
        loss = text_loss(model, x, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            steps_logged.append(step)
            losses_logged.append(validation_loss(model, val_data))
    return steps_logged, losses_logged


def generate(model, vocab, prompt, length=60, temperature=0.8, max_length=128):
    model.eval()
    generator = torch.Generator().manual_seed(0)
    ids = list(vocab.encode(prompt))

    with torch.no_grad():
        for _ in range(length):
            recent = ids[-max_length:]
            x = torch.tensor([recent])
            scores = model(x)
            last_scores = scores[0, -1] / temperature
            probs = F.softmax(last_scores, dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator).item()
            ids.append(next_id)

    return vocab.decode(ids)


def memory_test(kind, distance, steps=1500):
    X_train, y_train, vocab_size = recall_task(4000, distance, seed=0)
    X_test, y_test, _ = recall_task(1000, distance, seed=99)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)

    torch.manual_seed(0)
    model = make_model(kind, vocab_size, 64, max_length=distance + 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(0)

    for _ in range(steps):
        batch = torch.randint(0, len(X_train), (64,), generator=generator)
        scores = model(X_train[batch])
        scores_at_question = scores[:, -1]
        loss = F.cross_entropy(scores_at_question, y_train[batch])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(X_test)
        predictions = scores[:, -1].argmax(dim=-1).numpy()
    accuracy = np.mean(predictions == y_test)
    return accuracy, model


def main():
    text = corpus()
    vocab = Vocab(text)
    data = vocab.encode(text)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    names = {"rnn": "RNN", "gru": "GRU", "gpt": "Tiny GPT"}

    print("PART A: predicting the next character")
    curves = {}
    models = {}
    for kind in names:
        model = make_model(kind, len(vocab))
        n_params = sum(p.numel() for p in model.parameters())
        steps_logged, losses_logged = train_text_model(model, train_data, val_data)
        curves[kind] = (steps_logged, losses_logged)
        models[kind] = model
        print("  ", names[kind], "| parameters:", n_params,
              "| final validation loss:", round(losses_logged[-1], 3))
    print("   random guessing would be", round(math.log(len(vocab)), 3))
    print()

    print("Tiny GPT continuing 'are you f':")
    for temperature in [0.2, 0.8, 1.5]:
        output = generate(models["gpt"], vocab, "are you f", 60, temperature)
        first_line = output.split("\n")[0]
        print("  temperature", temperature, ":", first_line)
    print()

    print("PART B: remember a code from N steps ago (guessing = 12.5%)")
    distances = [5, 20, 60]
    results = {"rnn": [], "gru": [], "gpt": []}
    gpt_long = None

    for distance in distances:
        print("  distance", distance)
        for kind in names:
            accuracy, model = memory_test(kind, distance)
            results[kind].append(accuracy)
            print("     ", names[kind], round(accuracy * 100), "%")
            if kind == "gpt" and distance == distances[-1]:
                gpt_long = model
    print()

    X, _, _ = recall_task(200, distances[-1], seed=7)
    with torch.no_grad():
        gpt_long(torch.tensor(X))
    weights = gpt_long.attention.last_weights
    weights_from_question = weights[:, -1]
    average = weights_from_question.mean(dim=0).numpy()

    print("Tiny GPT: the '?' puts", round(average[0] * 100), "% of its attention on position 0,")
    print("where the code is. Spread evenly, each position would get",
          round(100 / len(average), 1), "%.")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    for kind in names:
        steps_logged, losses_logged = curves[kind]
        plt.plot(steps_logged, losses_logged, label=names[kind])
    plt.title("Next-character loss (validation)")
    plt.xlabel("step")
    plt.legend()

    plt.subplot(1, 3, 2)
    positions = np.arange(len(distances))
    width = 0.27
    plt.bar(positions - width, results["rnn"], width, label="RNN")
    plt.bar(positions, results["gru"], width, label="GRU")
    plt.bar(positions + width, results["gpt"], width, label="Tiny GPT")
    plt.axhline(1 / 8, linestyle="--", color="grey", label="guessing")
    plt.xticks(positions, ["5 steps", "20 steps", "60 steps"])
    plt.ylim(0, 1.3)
    plt.title("Memory test accuracy")
    plt.legend(loc="upper center", ncol=4, fontsize=8)

    plt.subplot(1, 3, 3)
    plt.bar(range(len(average)), average)
    plt.title("Where the '?' looks (Tiny GPT)")
    plt.xlabel("position")
    plt.ylabel("attention weight")

    plt.tight_layout()
    plt.savefig("plots/rnn_vs_attention.png", dpi=110)
    print()
    print("Saved plots/rnn_vs_attention.png")


if __name__ == "__main__":
    main()
