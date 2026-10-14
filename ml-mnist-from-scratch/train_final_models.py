"""Stage 4 training: the same network, with and without data augmentation.

Evaluates on clean validation data and on a distorted validation set that stands in
for 'real-world' handwriting. The MNIST test set and the handwriting set are NOT used here.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from augment import augment_batch, make_shifted_set
from data import load_mnist
from train import FIG_DIR, evaluate, plot_history, train

EPOCHS = 15


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    (x_train, y_train), (x_val, y_val), _ = load_mnist()
    x_val_shift = make_shifted_set(x_val)
    results = {}

    for name, aug in (("baseline", None), ("augmented", augment_batch)):
        params, hist = train(x_train, y_train, x_val, y_val, epochs=EPOCHS, lr=0.1,
                             n_hidden=128, augment=aug, verbose=False)
        plot_history(hist, os.path.join(FIG_DIR, f"{name}_training.png"), title=f"{name} (128 hidden)")
        clean = float(evaluate(x_val, y_val, params)[1])
        shifted = float(evaluate(x_val_shift, y_val, params)[1])
        results[name] = {"val_clean": clean, "val_shifted": shifted}
        print(f"{name:10s} clean val acc {clean:.4f}   distorted val acc {shifted:.4f}", flush=True)
        np.savez(os.path.join("models", f"{name}.npz"), **params)

    with open("results/augmentation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figures: what augmentation does, and the accuracy comparison.
    rng = np.random.default_rng(1)
    sample = x_train[:10]
    fig, axes = plt.subplots(4, 10, figsize=(10, 4.4))
    for r in range(4):
        batch = sample if r == 0 else augment_batch(sample, rng)
        for c in range(10):
            axes[r, c].imshow(batch[c].reshape(28, 28), cmap="gray")
            axes[r, c].axis("off")
    axes[0, 0].set_title("original", fontsize=9, loc="left")
    axes[1, 0].set_title("augmented (random draws)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "augmentation_examples.png"), dpi=120)
    plt.close(fig)

    plot_comparison(results)


def plot_comparison(results):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(2)
    for i, name in enumerate(("baseline", "augmented")):
        vals = [results[name]["val_clean"], results[name]["val_shifted"]]
        bars = ax.bar(xs + i * 0.35, vals, 0.35, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(xs + 0.175, ["clean validation", "distorted validation"])
    ax.set(ylim=(0, 1.05), ylabel="accuracy", title="Distribution shift and augmentation")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "augmentation_comparison.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
