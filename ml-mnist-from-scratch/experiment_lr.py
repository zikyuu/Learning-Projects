"""Stage 3: learning rate too low / right / too high. One epoch of training per setting."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from nn import backward, cross_entropy_loss, forward, init_params, sgd_step
from train import FIG_DIR

LRS = (0.001, 0.01, 0.1, 1.0, 5.0)


def run(lr, x, y, x_val, y_val, batch_size=64, log_every=10, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    order = rng.permutation(len(x))
    curve = []
    for step, start in enumerate(range(0, len(x), batch_size)):
        idx = order[start:start + batch_size]
        if step % log_every == 0:
            curve.append(float(cross_entropy_loss(forward(x[idx], params), y[idx])))
        sgd_step(params, backward(x[idx], y[idx], params), lr)
    probs = forward(x_val, params)
    acc = float(np.mean(probs.argmax(axis=1) == y_val))
    return curve, acc


def main():
    (x, y), (x_val, y_val), _ = load_mnist()
    fig, ax = plt.subplots(figsize=(8, 5))
    results = []
    for lr in LRS:
        curve, acc = run(lr, x, y, x_val, y_val)
        curve = np.array(curve)
        print(f"lr {lr:<6g} final batch loss {curve[-1]:.3f}  val acc after 1 epoch {acc:.3f}", flush=True)
        results.append({"lr": lr, "final_batch_loss": float(curve[-1]), "val_acc": acc})
        ax.plot(np.arange(len(curve)) * 10, np.clip(curve, 0, 10), label=f"lr = {lr:g}")
    ax.set(title="Learning rate: training loss over one epoch", xlabel="step", ylabel="batch loss (clipped at 10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "learning_rate.png"), dpi=120)
    os.makedirs("results", exist_ok=True)
    with open("results/learning_rate.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
