import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

from data import load, shuffled_pixels

torch.manual_seed(0)


class MLP(nn.Module):

    def __init__(self, n_hidden=128):
        super().__init__()
        self.layer1 = nn.Linear(64, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class SmallCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.average(x)
        x = x.view(-1, 64)
        x = self.output(x)
        return x


def make_loader(X, y, shuffle=False):
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            scores = model(X_batch)
            predictions = scores.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)
    return correct / total


def train(model, train_loader, val_loader, epochs=80, patience=12):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    loss_history = []
    val_history = []
    best_accuracy = 0.0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            scores = model(X_batch)
            loss = loss_function(scores, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        average_loss = total_loss / len(train_loader.dataset)
        val_accuracy = accuracy(model, val_loader)
        loss_history.append(average_loss)
        val_history.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    print("    stopped after", epoch + 1, "epochs")
    return loss_history, val_history


def train_three_models(X_train, X_val, X_test, y_train, y_val, y_test):
    torch.manual_seed(0)
    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    print("  training LightGBM")
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=0, verbose=-1)
    gbm.fit(X_train, y_train)
    gbm_predictions = gbm.predict(X_test)
    gbm_accuracy = np.mean(gbm_predictions == y_test)

    print("  training MLP")
    mlp = MLP()
    mlp_history = train(mlp, train_loader, val_loader)
    mlp_accuracy = accuracy(mlp, test_loader)

    print("  training CNN")
    cnn = SmallCNN()
    cnn_history = train(cnn, train_loader, val_loader)
    cnn_accuracy = accuracy(cnn, test_loader)

    results = {"LightGBM": gbm_accuracy, "MLP": mlp_accuracy, "CNN": cnn_accuracy}
    return results, mlp_history, cnn_history, cnn


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load()

    print("Normal images:")
    normal, mlp_history, cnn_history, cnn = train_three_models(
        X_train, X_val, X_test, y_train, y_val, y_test)
    print()
    print("Test accuracy:")
    for name in normal:
        print("  ", name, round(normal[name] * 100, 1), "%")
    print()

    torch.save(cnn.state_dict(), "cnn_weights.pt")
    reloaded = SmallCNN()
    reloaded.load_state_dict(torch.load("cnn_weights.pt"))
    reloaded_accuracy = accuracy(reloaded, make_loader(X_test, y_test))
    print("Saved cnn_weights.pt and loaded it into a fresh CNN.")
    print("Reloaded accuracy:", round(reloaded_accuracy * 100, 1), "% (should match the CNN above)")
    print()

    X_train_s, X_val_s, X_test_s = shuffled_pixels(X_train, X_val, X_test)
    print("Scrambled images:")
    scrambled, _, _, _ = train_three_models(
        X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
    print()

    print("Normal vs scrambled pixels:")
    for name in normal:
        before = round(normal[name] * 100, 1)
        after = round(scrambled[name] * 100, 1)
        print("  ", name, "  normal", before, "%   scrambled", after, "%")
    print()
    print("LightGBM and the MLP treat pixels as separate columns, so scrambling")
    print("hardly affects them. The CNN's filters expect neighbouring pixels to")
    print("belong together, so it drops. That is what 'using the image structure' means.")

    mlp_loss, mlp_val = mlp_history
    cnn_loss, cnn_val = cnn_history

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(mlp_loss, label="MLP")
    plt.plot(cnn_loss, label="CNN")
    plt.title("Training loss")
    plt.xlabel("epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mlp_val, label="MLP")
    plt.plot(cnn_val, label="CNN")
    plt.title("Validation accuracy (early stopping watches this)")
    plt.xlabel("epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/neural_networks.png", dpi=110)
    print()
    print("Saved plots/neural_networks.png")


if __name__ == "__main__":
    main()
