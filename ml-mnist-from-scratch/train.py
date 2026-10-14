"""Mini-batch SGD training with optional L2, dropout, augmentation and best-epoch tracking."""
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from nn import backward, cross_entropy_loss, forward, init_params, sgd_step

FIG_DIR = "figures"


def evaluate(X, y, params):
    probs = forward(X, params)
    return cross_entropy_loss(probs, y), np.mean(probs.argmax(axis=1) == y)


def train(x_train, y_train, x_val, y_val, epochs=10, batch_size=64, lr=0.1,
          n_hidden=128, l2=0.0, dropout=0.0, augment=None, seed=0,
          val_subset=3000, verbose=True):
    """Train and record per-epoch train/validation loss and accuracy.

    l2:       weight decay strength (adds l2 * W to the weight gradients)
    dropout:  probability of silencing each hidden neuron during training
    augment:  optional function (X_batch, rng) -> X_batch applied to every training batch
    val_subset: curves use the first N validation images (speed); pass None for all.
    history["best_params"] holds the weights from the epoch with the lowest val loss
    (used for early stopping).
    """
    rng = np.random.default_rng(seed)
    params = init_params(rng, n_hidden=n_hidden)
    xv, yv = (x_val, y_val) if val_subset is None else (x_val[:val_subset], y_val[:val_subset])
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = (np.inf, 0, None)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            idx = order[start:start + batch_size]
            xb = x_train[idx] if augment is None else augment(x_train[idx], rng)
            grads = backward(xb, y_train[idx], params, dropout=dropout, rng=rng)
            if l2:
                for key in ("W1", "W2"):  # weight decay on weights, not biases
                    grads[key] += l2 * params[key]
            sgd_step(params, grads, lr)

        tr_loss, tr_acc = evaluate(x_train[:5000], y_train[:5000], params)
        va_loss, va_acc = evaluate(xv, yv, params)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(float(v))
        if va_loss < best[0]:
            best = (va_loss, epoch, {k: v.copy() for k, v in params.items()})
        if verbose:
            print(f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f}")

    history["best_epoch"] = best[1]
    history["best_params"] = best[2]
    return params, history


def plot_history(history, path, title=""):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="validation")
    ax1.set(title="Loss", xlabel="epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="validation")
    ax2.set(title="Accuracy", xlabel="epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), _ = load_mnist()  # test set deliberately untouched
    params, history = train(x_train, y_train, x_val, y_val)
    plot_history(history, os.path.join(FIG_DIR, "baseline_training.png"),
                 title="Baseline: 50k samples, 128 hidden neurons")
