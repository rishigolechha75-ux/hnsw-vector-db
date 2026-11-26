"""Exact (brute-force) k-NN search: the ground truth and the baseline to beat."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def topk_sq_l2(X: np.ndarray, Q: np.ndarray, k: int, chunk: int = 256,
               ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k by squared L2 using one matrix multiply per query chunk.

    Args:
        X: [n, d] database vectors.
        Q: [nq, d] queries.
        k: neighbours per query (clipped to n; missing slots are -1 / inf).
        ids: optional [n] labels to return instead of row positions.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q[None, :]
    nq, n = Q.shape[0], X.shape[0]
    out_i = np.full((nq, k), -1, np.int64)
    out_d = np.full((nq, k), np.inf, np.float64)
    if n == 0:
        return out_i, out_d
    kk = min(k, n)
    xn = (X * X).sum(1)
    for s in range(0, nq, chunk):
        q = Q[s:s + chunk]
        d = xn[None, :] - 2.0 * (q @ X.T) + (q * q).sum(1)[:, None]
        if kk < n:
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        else:
            part = np.tile(np.arange(n), (q.shape[0], 1))
        pd = np.take_along_axis(d, part, 1)
        order = np.argsort(pd, axis=1)
        top = np.take_along_axis(part, order, 1)
        out_i[s:s + chunk, :kk] = top if ids is None else ids[top]
        out_d[s:s + chunk, :kk] = np.maximum(np.take_along_axis(pd, order, 1), 0.0)
    return out_i, out_d


class FlatIndex:
    """Stores vectors in a growable array and answers queries exactly."""

    def __init__(self, dim: int):
        self.dim = dim
        self._X = np.zeros((0, dim), np.float32)

    @property
    def count(self) -> int:
        return self._X.shape[0]

    def add(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32).reshape(-1, self.dim)
        start = self.count
        self._X = np.concatenate([self._X, X])
        return np.arange(start, self.count, dtype=np.int64)

    def search(self, Q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        return topk_sq_l2(self._X, Q, k)

    def memory_bytes(self) -> int:
        return int(self._X.nbytes)
