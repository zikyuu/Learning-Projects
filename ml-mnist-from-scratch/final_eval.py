"""Final evaluation. Run after all tuning is finished.

- MNIST test set: scored once; the numbers are cached in results/test.json and never recomputed.
- Handwriting set (optional): scored every run. Never used for training or model selection.

Usage: python3 final_eval.py [path/to/processed.npz] [tag]
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from nn import forward
from train import FIG_DIR, evaluate

MODELS = ("baseline", "augmented")
TEST_CACHE = "results/test.json"


def load_model(name):
    with np.load(os.path.join("models", f"{name}.npz")) as f:
        return {k: f[k] for k in f.files}


def accuracy(params, x, y):
    return float(np.mean(forward(x, params).argmax(axis=1) == y))


def main():
    models = {name: load_model(name) for name in MODELS}

    if os.path.exists(TEST_CACHE):
        with open(TEST_CACHE) as f:
            test = json.load(f)
        print("MNIST test accuracy (cached, scored once):", test)
    else:
        _, _, (x_test, y_test) = load_mnist()
        test = {name: accuracy(p, x_test, y_test) for name, p in models.items()}
        with open(TEST_CACHE, "w") as f:
            json.dump(test, f, indent=2)
        print("MNIST test accuracy (scored now, once):", test)

    path = sys.argv[1] if len(sys.argv) > 1 else "handwriting/processed.npz"
    tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    if not os.path.exists(path):
        print("no handwriting set found; skipping")
        return
    with np.load(path) as f:
        xh, yh = f["x"], f["y"]
    out = {"n": int(len(xh))}
    for name, p in models.items():
        out[name] = accuracy(p, xh, yh)
        per_digit = [float(np.mean(forward(xh[yh == d], p).argmax(axis=1) == d)) for d in range(10)]
        out[f"{name}_per_digit"] = per_digit
        print(f"{name:10s} handwriting accuracy {out[name]:.3f} on {len(xh)} images")
    os.makedirs("results", exist_ok=True)
    with open(f"results/handwriting{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure: accuracy comparison + misclassified examples from the baseline model.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.3]})
    bars = axes[0].bar(range(len(MODELS)), [out[m] for m in MODELS], color=["#888780", "#378ADD"])
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set_xticks(range(len(MODELS)), MODELS)
    axes[0].set(ylim=(0, 1.1), ylabel="accuracy", title=f"My handwriting ({len(xh)} images)")
    width = 0.38
    xs = np.arange(10)
    for i, m in enumerate(MODELS):
        axes[1].bar(xs + i * width, out[f"{m}_per_digit"], width, label=m)
    axes[1].set(xlabel="digit", ylabel="accuracy", ylim=(0, 1.25), title="Per-digit accuracy")
    axes[1].set_xticks(xs + width / 2, [str(d) for d in xs])
    axes[1].legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"handwriting_results{tag}.png"), dpi=120)
    plt.close(fig)

    wrong = np.where(forward(xh, models["baseline"]).argmax(axis=1) != yh)[0][:20]
    if len(wrong):
        fig, axes = plt.subplots(2, 10, figsize=(10, 2.8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), wrong):
            pred = forward(xh[i:i + 1], models["baseline"]).argmax()
            ax.imshow(xh[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"{yh[i]}→{pred}", fontsize=9)
        fig.suptitle("Baseline mistakes on my handwriting (true → predicted)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"handwriting_mistakes{tag}.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
