import numpy as np
import pytest


@pytest.fixture(scope="session")
def clustered():
    """Small clustered dataset (Gaussian blobs) with held-out queries."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((20, 16)) * 4
    X = (centers[rng.integers(0, 20, 3000)] + rng.standard_normal((3000, 16))).astype(np.float32)
    Q = (centers[rng.integers(0, 20, 100)] + rng.standard_normal((100, 16))).astype(np.float32)
    return X, Q


def recall(I, gt):
    return sum(len(set(a[a >= 0]) & set(b)) for a, b in zip(I, gt)) / gt.size
