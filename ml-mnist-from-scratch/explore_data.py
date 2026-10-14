"""Inspect the MNIST data: split sizes, raw pixel values, and a sample grid."""
import os

import matplotlib.pyplot as plt

from data import load_mnist

FIG_DIR = "figures"


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print("train:", x_train.shape, y_train.shape)
    print("val:  ", x_val.shape, y_val.shape)
    print("test: ", x_test.shape, y_test.shape)

    # One image is a flat vector of 784 pixel intensities in [0, 1].
    img = x_train[0]
    print(f"\nfirst label: {y_train[0]}, pixels: {img.shape[0]}, range: [{img.min()}, {img.max()}]")
    print("\nfirst image as a 28x28 grid (. dark, + dim, # bright):")
    for row in img.reshape(28, 28):
        print("".join("#" if p > 0.5 else "+" if p > 0.1 else "." for p in row))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, i in zip(axes.flat, range(10)):
        ax.imshow(x_train[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"label {y_train[i]}")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "mnist_samples.png")
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
