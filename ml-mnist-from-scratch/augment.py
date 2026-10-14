"""Data augmentation for 28x28 digit images stored as flat 784-vectors in [0, 1]."""
import numpy as np
from scipy import ndimage


def _augment_one(img, rng, max_rot, max_shift, max_scale, p_thick, p_thin):
    angle = np.deg2rad(rng.uniform(-max_rot, max_rot))
    scale = 1 + rng.uniform(-max_scale, max_scale)
    shift = rng.uniform(-max_shift, max_shift, size=2)
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    matrix = np.array([[cos, -sin], [sin, cos]])
    centre = np.array([13.5, 13.5])
    offset = centre - matrix @ centre - shift  # rotate/scale about the image centre
    out = ndimage.affine_transform(img, matrix, offset=offset, order=1, mode="constant")
    r = rng.random()
    if r < p_thick:
        out = ndimage.grey_dilation(out, size=(2, 2))  # thicker strokes
    elif r < p_thick + p_thin:
        out = ndimage.grey_erosion(out, size=(2, 2))   # thinner strokes
    return out


def augment_batch(X, rng, max_rot=15, max_shift=3, max_scale=0.15, p_thick=0.25, p_thin=0.15):
    """Random rotation, shift, scale and stroke thickness for a batch (N, 784)."""
    out = np.empty_like(X)
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        out[i] = _augment_one(img, rng, max_rot, max_shift, max_scale, p_thick, p_thin).ravel()
    return out


def make_shifted_set(X, seed=123):
    """A fixed, stronger-than-training distortion of a clean set.

    Stand-in for 'real-world' handwriting: same digits, but rotated, shifted,
    rescaled and re-stroked in ways the plain model never saw.
    """
    rng = np.random.default_rng(seed)
    return augment_batch(X, rng, max_rot=20, max_shift=3, max_scale=0.2, p_thick=0.4, p_thin=0.2)
