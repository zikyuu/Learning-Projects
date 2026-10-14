"""A two-layer neural network (784 -> hidden -> 10) in plain numpy.

Parameters live in a dict: {"W1", "b1", "W2", "b2"}. Gradients returned by
`backward` use the same keys and shapes.
"""
import os

# Default BLAS threading is ~10x slower for these small matrices; must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np  # noqa: E402


def relu(z):
    return np.maximum(0, z)


def neuron(x, w, b):
    """A single neuron: weighted sum of inputs plus bias, through ReLU."""
    return relu(np.dot(x, w) + b)


def layer(X, W, b):
    """A layer of neurons. X (N, in) @ W (in, out) + b (out,) -> (N, out)."""
    return relu(np.dot(X, W) + b)


def softmax(logits):
    """Row-wise softmax. Subtracting the row max avoids overflow in exp."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """Mean of -log(probability assigned to the correct class)."""
    n = probs.shape[0]
    p_correct = probs[np.arange(n), y]
    return -np.sum(np.log(p_correct + 1e-12)) / n  # epsilon avoids log(0)


def init_params(rng, n_in=784, n_hidden=128, n_out=10):
    """He initialization: random weights scaled by sqrt(2 / fan_in), zero biases.

    Random values break symmetry so neurons can specialize; the scaling keeps
    activations a sensible size from layer to layer.
    """
    return {
        "W1": rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in),
        "b1": np.zeros(n_hidden),
        "W2": rng.standard_normal((n_hidden, n_out)) * np.sqrt(2 / n_hidden),
        "b2": np.zeros(n_out),
    }


def forward(X, params):
    """X (N, 784) -> class probabilities (N, 10)."""
    H = layer(X, params["W1"], params["b1"])
    logits = H @ params["W2"] + params["b2"]  # raw scores, no ReLU
    return softmax(logits)


def backward(X, y, params, dropout=0.0, rng=None):
    """Gradients of the mean cross-entropy loss w.r.t. every parameter.

    With dropout > 0 (training only), each hidden neuron is silenced with that
    probability and the survivors are scaled by 1/(1-dropout) ("inverted dropout"),
    so no rescaling is needed at test time.
    """
    n = X.shape[0]

    # Forward again, keeping the intermediates the backward pass needs.
    z1 = X @ params["W1"] + params["b1"]
    H = relu(z1)
    mask = 1.0
    if dropout > 0:
        mask = (rng.random(H.shape) >= dropout) / (1 - dropout)
        H = H * mask
    probs = softmax(H @ params["W2"] + params["b2"])

    # Softmax + cross-entropy: d(loss)/d(logits) = probs - one_hot(y), batch-averaged.
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n

    dW2 = H.T @ dlogits
    db2 = dlogits.sum(axis=0)

    # Blame flows to the hidden layer, but only through neurons that were active.
    dz1 = (dlogits @ params["W2"].T) * mask * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def sgd_step(params, grads, lr):
    """Gradient descent update, in place: param <- param - lr * grad."""
    for key in params:
        params[key] -= lr * grads[key]


if __name__ == "__main__":
    from data import load_mnist

    (x_train, y_train), _, _ = load_mnist()
    params = init_params(np.random.default_rng(0))

    probs = forward(x_train[:64], params)
    print("probs shape:", probs.shape)
    print("initial loss:", cross_entropy_loss(probs, y_train[:64]), "(chance = ln 10 = 2.303)")
    print("initial accuracy:", np.mean(probs.argmax(axis=1) == y_train[:64]), "(chance = 0.1)")
