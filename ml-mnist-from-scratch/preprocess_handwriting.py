"""Turn a photo/scan of a handwriting sheet into 28x28 MNIST-style images.

Sheet layout: a 10x10 grid, one row per digit (row 0 = digit 0 ... row 9 = digit 9),
ten samples per row. Crop the photo to the outer border of the grid before running.

Usage:  python3 preprocess_handwriting.py handwriting/raw/sheet.jpg [tag]
Output: handwriting/processed.npz (x: (100, 784) in [0,1], y: (100,)) and a preview PNG.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

OUT_DIR = "handwriting"


def otsu_threshold(img):
    """Otsu's method on a [0,1] image: the threshold that best separates ink from paper."""
    hist, edges = np.histogram(img, bins=256, range=(0, 1))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m0 = np.cumsum(p * centers) / np.maximum(w0, 1e-12)
    m_total = (p * centers).sum()
    m1 = (m_total - np.cumsum(p * centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return centers[np.argmax(between)]


def flatten_background(gray):
    """Divide out uneven lighting, then map to ink = 1, paper = 0."""
    bg = ndimage.gaussian_filter(gray, sigma=gray.shape[0] / 8)
    norm = np.clip(gray / np.maximum(bg, 1e-3), 0, 1)
    return 1 - norm


def cell_to_mnist(cell, margin=0.05):
    """One grid cell (ink = 1) -> 28x28 image: crop to ink, fit in 20x20, centre by mass."""
    h, w = cell.shape
    my, mx = int(h * margin), int(w * margin)
    cell = cell[my:h - my, mx:w - mx]  # drop the grid lines at the cell edges
    cell = ndimage.gaussian_filter(cell, 1.0)
    cell = cell / max(cell.max(), 1e-6)
    mask = cell > max(otsu_threshold(cell), 0.5)  # thin grey grid lines stay below this
    if mask.sum() < 15:  # empty or unreadable cell
        return None
    ys, xs = np.where(mask)
    crop = cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(crop.shape)
    new_size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS)) / 255.0
    canvas = np.zeros((28, 28))
    canvas[:small.shape[0], :small.shape[1]] = small
    cy, cx = ndimage.center_of_mass(canvas)  # MNIST convention: centre of mass at the centre
    canvas = ndimage.shift(canvas, (13.5 - cy, 13.5 - cx), order=1, mode="constant")
    return np.clip(canvas, 0, 1)


def crop_to_grid(gray):
    """Crop to the outer grid lines: they are the long dark rows/columns (ink never spans a page)."""
    dark = gray < 0.85
    cols = np.where(dark.mean(axis=0) > 0.5)[0]
    rows = np.where(dark.mean(axis=1) > 0.5)[0]
    if len(cols) < 2 or len(rows) < 2:
        return gray  # no grid found; assume the photo is already cropped
    return gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def process_sheet(path, n_rows=10, n_cols=10):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    gray = crop_to_grid(gray)
    ink = flatten_background(gray)
    h, w = ink.shape
    images, labels = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            cell = ink[r * h // n_rows:(r + 1) * h // n_rows, c * w // n_cols:(c + 1) * w // n_cols]
            img = cell_to_mnist(cell)
            if img is None:
                print(f"warning: empty cell at row {r} (digit {r}), column {c}; skipped")
                continue
            images.append(img.ravel())
            labels.append(r)
    return np.array(images, dtype=np.float32), np.array(labels)


def save_preview(x, y, path):
    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    counts = np.zeros(10, dtype=int)
    for ax in axes.ravel():
        ax.axis("off")
    for img, label in zip(x, y):
        axes[label, counts[label]].imshow(img.reshape(28, 28), cmap="gray")
        counts[label] += 1
    fig.suptitle("Processed handwriting (row = digit)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    sheet = sys.argv[1]
    tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""  # optional tag keeps sets separate
    os.makedirs(OUT_DIR, exist_ok=True)
    x, y = process_sheet(sheet)
    np.savez(os.path.join(OUT_DIR, f"processed{tag}.npz"), x=x, y=y)
    save_preview(x, y, os.path.join(OUT_DIR, f"processed_preview{tag}.png"))
    print(f"saved {len(x)} images; per-digit counts: {np.bincount(y, minlength=10).tolist()}")
