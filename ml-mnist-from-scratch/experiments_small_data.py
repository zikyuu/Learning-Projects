"""Stages 2-3: overfit on purpose (500 samples), then fix it one change at a time."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from nn import forward
from train import FIG_DIR, evaluate, plot_history, train

N_SMALL = 500
EPOCHS = 200
BIG = 512
RESULTS = "results"


def summarize(name, params, history, x_val, y_val):
    """Final-epoch numbers on the full validation set, plus the train/val gap."""
    full_loss, full_acc = evaluate(x_val, y_val, params)
    row = {
        "name": name,
        "train_acc": history["train_acc"][-1],
        "val_acc": float(full_acc),
        "val_loss": float(full_loss),
        "min_val_loss": min(history["val_loss"]),
        "best_epoch": history["best_epoch"],
        "gap": history["train_acc"][-1] - float(full_acc),
    }
    print(f"{name:32s} train {row['train_acc']:.3f}  val {row['val_acc']:.3f}  "
          f"gap {row['gap']:.3f}  val loss {row['val_loss']:.3f}", flush=True)
    return row


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    (x_train, y_train), (x_val, y_val), _ = load_mnist()
    xs, ys = x_train[:N_SMALL], y_train[:N_SMALL]
    rows, hists = [], {}

    def run(name, tag, **kw):
        kw.setdefault("n_hidden", BIG)
        params, hist = train(xs, ys, x_val, y_val, epochs=EPOCHS, lr=0.1, verbose=False, **kw)
        rows.append(summarize(name, params, hist, x_val, y_val))
        hists[tag] = hist
        plot_history(hist, os.path.join(FIG_DIR, f"{tag}.png"), title=name)
        return params, hist

    # Stage 2: break it on purpose.
    _, base = run(f"overfit: {BIG} hidden, 500 samples", "overfit_500_samples")

    # Early stopping: same run, but keep the weights from the lowest-val-loss epoch.
    best_params = base["best_params"]
    es_loss, es_acc = evaluate(x_val, y_val, best_params)
    es_train_acc = evaluate(xs, ys, best_params)[1]
    rows.append({"name": f"early stopping (epoch {base['best_epoch']})",
                 "train_acc": float(es_train_acc), "val_acc": float(es_acc),
                 "val_loss": float(es_loss), "best_epoch": base["best_epoch"],
                 "gap": float(es_train_acc - es_acc)})
    print(f"{rows[-1]['name']:32s} train {es_train_acc:.3f}  val {es_acc:.3f}  "
          f"gap {es_train_acc - es_acc:.3f}  val loss {es_loss:.3f}", flush=True)

    # Stage 3: one change at a time.
    run("smaller model (16 hidden)", "fix_small_model", n_hidden=16)
    for l2 in (1e-3, 1e-2, 5e-2):
        run(f"L2 weight decay {l2:g}", f"fix_l2_{l2:g}", l2=l2)
    for p in (0.2, 0.5):
        run(f"dropout {p}", f"fix_dropout_{p}", dropout=p)

    # More data (same big model).
    data_rows = []
    for n, ep in ((500, 100), (2000, 60), (10000, 30), (50000, 10)):
        params, hist = train(x_train[:n], y_train[:n], x_val, y_val, epochs=ep, lr=0.1,
                             n_hidden=BIG, verbose=False)
        r = summarize(f"more data: {n} samples", params, hist, x_val, y_val)
        r["n"] = n
        data_rows.append(r)

    # Comparison figure: validation loss curves for each fix.
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label in (("overfit_500_samples", "no fix (512 hidden)"),
                       ("fix_small_model", "16 hidden"),
                       ("fix_l2_0.01", "L2 0.01"),
                       ("fix_dropout_0.5", "dropout 0.5")):
        ax.plot(hists[tag]["val_loss"], label=label)
    ax.set(title="Validation loss, 500 training samples", xlabel="epoch", ylabel="validation loss",
           ylim=(0, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fixes_val_loss_comparison.png"), dpi=120)
    plt.close(fig)

    # More-data figure.
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [r["n"] for r in data_rows]
    ax.semilogx(ns, [r["train_acc"] for r in data_rows], "o-", label="train")
    ax.semilogx(ns, [r["val_acc"] for r in data_rows], "o-", label="validation")
    ax.set(title="Effect of more training data", xlabel="training samples", ylabel="accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "more_data.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(RESULTS, "small_data.json"), "w") as f:
        json.dump({"rows": rows, "data_rows": data_rows}, f, indent=2)


if __name__ == "__main__":
    main()
