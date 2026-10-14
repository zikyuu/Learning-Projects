"""Numerical gradient check: compare backprop gradients with finite differences."""
import numpy as np

from data import load_mnist
from nn import backward, cross_entropy_loss, forward, init_params


def loss_fn(X, y, params):
    return cross_entropy_loss(forward(X, params), y)


def main():
    (x_train, y_train), _, _ = load_mnist()
    X, y = x_train[:32].astype(np.float64), y_train[:32]
    rng = np.random.default_rng(0)
    params = init_params(rng)
    grads = backward(X, y, params)

    eps = 1e-5
    for name in params:
        worst = 0.0
        for _ in range(10):  # spot-check 10 random entries per parameter
            idx = tuple(rng.integers(0, s) for s in params[name].shape)
            orig = params[name][idx]
            params[name][idx] = orig + eps
            plus = loss_fn(X, y, params)
            params[name][idx] = orig - eps
            minus = loss_fn(X, y, params)
            params[name][idx] = orig
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name][idx]
            denom = max(abs(numeric), abs(analytic), 1e-12)
            worst = max(worst, abs(numeric - analytic) / denom)
        print(f"{name}: worst relative error = {worst:.2e}")


if __name__ == "__main__":
    main()
