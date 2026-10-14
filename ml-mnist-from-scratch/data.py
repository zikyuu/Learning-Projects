"""MNIST loading + train/val/test split. Plain numpy, no ML libraries."""
import gzip
import os
import urllib.request

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # see nn.py

import numpy as np  # noqa: E402

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data")


def _download(filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def _read_images(path):
    with gzip.open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16)  # 16-byte header
    return raw.reshape(-1, 784)  # each image: 28*28 = 784 numbers


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)  # 8-byte header


def load_mnist():
    """Returns (x_train, y_train), (x_val, y_val), (x_test, y_test).

    Pixels are scaled from 0..255 to 0..1. Split: 50k train / 10k val / 10k test.
    The test set is the official MNIST test set; touch it only once, at the very end.
    """
    x_all = _read_images(_download(FILES["train_x"])).astype(np.float32) / 255.0
    y_all = _read_labels(_download(FILES["train_y"]))
    x_test = _read_images(_download(FILES["test_x"])).astype(np.float32) / 255.0
    y_test = _read_labels(_download(FILES["test_y"]))

    # Fixed shuffle so the split is the same every run.
    idx = np.random.default_rng(0).permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    return (
        (x_all[:50000], y_all[:50000]),
        (x_all[50000:], y_all[50000:]),
        (x_test, y_test),
    )
