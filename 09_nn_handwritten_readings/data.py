"""Handwritten digits — bundled with scikit-learn, so no download needed."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load(seed=0):
    """Returns train / validation / test splits.

    X  : (n, 64) flattened pixels, scaled to 0-1
    y  : (n,) integer labels 0-9

    Three-way split: train to learn, validation to choose when to stop,
    test sealed until the very end.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0      # pixels are 0-16
    y = digits.target.astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=seed, stratify=y_tmp)

    return X_train, X_val, X_test, y_train, y_val, y_test


def shuffled_pixels(*arrays, seed=42):
    """Apply ONE fixed random permutation to the pixel order of every image.

    No information is destroyed: every image keeps all 64 pixel values.
    Only the spatial arrangement is scrambled.
    """
    perm = np.random.default_rng(seed).permutation(64)
    return tuple(a[:, perm] for a in arrays)
