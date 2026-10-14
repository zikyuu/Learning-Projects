"""Does the MNIST-style preprocessing (crop to ink, fit in 20x20, centre by mass) matter?

Compares it with a naive version that just shrinks each whole grid cell to 28x28.
Both models are only evaluated here; nothing is trained or tuned on the handwriting.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from final_eval import MODELS, accuracy, load_model
from preprocess_handwriting import crop_to_grid, flatten_background, process_sheet


def naive_process(path, n=10):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    ink = flatten_background(crop_to_grid(gray))
    h, w = ink.shape
    images, labels = [], []
    for r in range(n):
        for c in range(n):
            cell = ink[r * h // n:(r + 1) * h // n, c * w // n:(c + 1) * w // n]
            m = int(0.05 * cell.shape[0])
            cell = cell[m:-m, m:-m]
            cell = np.clip(cell / max(cell.max(), 1e-6), 0, 1)
            small = Image.fromarray((cell * 255).astype(np.uint8)).resize((28, 28), Image.LANCZOS)
            images.append(np.asarray(small, dtype=np.float32).ravel() / 255.0)
            labels.append(r)
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    sheet = sys.argv[1] if len(sys.argv) > 1 else "handwriting/raw/sheet.jpg"
    tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    models = {m: load_model(m) for m in MODELS}
    sets = {"MNIST-style preprocessing": process_sheet(sheet), "naive resize": naive_process(sheet)}
    results = {}
    for set_name, (x, y) in sets.items():
        results[set_name] = {m: accuracy(p, x, y) for m, p in models.items()}
        print(set_name, {m: round(a, 3) for m, a in results[set_name].items()})
    with open(f"results/preprocessing_ablation{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(2, 10, figsize=(10, 2.6))
    for ax in axes.ravel():
        ax.axis("off")
    for row, (set_name, (x, y)) in enumerate(sets.items()):
        for d in range(10):
            axes[row, d].imshow(x[y == d][0].reshape(28, 28), cmap="gray")
        axes[row, 0].set_title(set_name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(f"figures/preprocessing_ablation{tag}.png", dpi=120)
