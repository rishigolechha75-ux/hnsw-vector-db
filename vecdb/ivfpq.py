"""Inverted-file index with product quantisation (IVF-PQ).

Based on Jegou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search"
(IEEE TPAMI 2011). The index has two levels:

1. A coarse k-means quantiser with ``nlist`` centroids partitions the space into
   Voronoi cells. Each vector is stored in the inverted list of its nearest cell.
2. The *residual* (vector minus its cell centroid) is split into ``m`` sub-vectors,
   and each sub-vector is replaced by the id of its nearest centroid in a 256-entry
   per-subspace codebook. A vector therefore costs ``m`` bytes instead of 4*dim.

At query time we probe the ``nprobe`` closest cells. For each probed cell we build
a lookup table ``LUT[j, c] = ||(q - centroid)_j - codebook[j, c]||^2`` and the
approximate distance to every encoded vector is just ``sum_j LUT[j, code_j]``
(asymmetric distance computation, ADC). Optionally the best ``rerank`` candidates
are re-scored with the exact float vectors.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances [n, k] via the ||x||^2 - 2xc + ||c||^2 expansion."""
    d = (X * X).sum(1)[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def assign(X: np.ndarray, C: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Index of the nearest centroid for each row of X (chunked to bound memory)."""
    out = np.empty(X.shape[0], np.int64)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = _sq_dists(X[s:s + chunk], C).argmin(1)
    return out


def kmeans(X: np.ndarray, k: int, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Lloyd's k-means with k-means++ initialisation. Returns float32 [k, d] centroids.

    Empty clusters are re-seeded with the point currently farthest from its centroid,
    a standard fix that keeps all ``k`` codewords useful.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"need at least k={k} training points, got {n}")
    rng = np.random.default_rng(seed)
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    closest = ((X - C[0]) ** 2).sum(1)
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            C[i] = X[rng.integers(n)]
        else:
            C[i] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, ((X - C[i]) ** 2).sum(1))
    for _ in range(n_iter):
        a = assign(X, C)
        counts = np.bincount(a, minlength=k)
        sums = np.zeros_like(C, dtype=np.float64)
        np.add.at(sums, a, X)
        nonempty = counts > 0
        C[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        empty = np.nonzero(~nonempty)[0]
        if empty.size:
            err = ((X - C[a]) ** 2).sum(1)
            far = np.argsort(-err)[: empty.size]
            C[empty] = X[far]
    return C


@njit(cache=True, inline="always")
def _heap_insert(bd, bi, size, d, i):
    """Insert (d, i) into a bounded max-heap stored in arrays; returns new size.

    Keeps the ``bd.shape[0]`` smallest distances seen so far in O(log cap) per item,
    which avoids sorting every candidate in the probed lists.
    """
    cap = bd.shape[0]
    if size < cap:
        j = size
        size += 1
        while j > 0:
            parent = (j - 1) // 2
            if bd[parent] >= d:
                break
            bd[j] = bd[parent]
            bi[j] = bi[parent]
            j = parent
        bd[j] = d
        bi[j] = i
        return size
    if d >= bd[0]:
        return size
    j = 0
    while True:
        left = 2 * j + 1
        if left >= cap:
            break
        child = left
        if left + 1 < cap and bd[left + 1] > bd[left]:
            child = left + 1
        if bd[child] <= d:
            break
        bd[j] = bd[child]
        bi[j] = bi[child]
        j = child
    bd[j] = d
    bi[j] = i
    return size


@njit(cache=True, fastmath=True)
def _ivfpq_search(Q, coarse, codebooks, term2, offsets, codes, ids, raw, k, nprobe,
                  rerank, stats):
    """ADC search with precomputed tables.

    ||q - yc - yr||^2 = ||q - yc||^2 + (||yr||^2 + 2<yc, yr>) - 2<q, yr>
                         term 1         term 2 (per cell, precomputed)   term 3 (per query)

    Every term splits over the m sub-spaces, so the per-cell lookup table is
    ``term2[p] - 2 * term3`` and costs m*256 additions instead of m*256*dsub
    multiply-adds (Jegou et al. 2011, Sec. IV-A; also what FAISS does).
    """
    nq, d = Q.shape
    nlist = coarse.shape[0]
    m, ksub, dsub = codebooks.shape
    out_i = np.full((nq, k), -1, np.int64)
    out_d = np.full((nq, k), np.inf, np.float64)
    term3 = np.empty((m, ksub), np.float32)
    lut = np.empty((m, ksub), np.float32)
    npr = min(nprobe, nlist)
    keep_cap = max(k, rerank)
    pd = np.empty(npr, np.float64)
    pi = np.empty(npr, np.int64)
    bd = np.empty(keep_cap, np.float64)
    bi = np.empty(keep_cap, np.int64)
    for qi in range(nq):
        q = Q[qi]
        # nprobe closest cells (bounded heap, no full sort of nlist distances)
        ps = 0
        for c in range(nlist):
            s = 0.0
            for t in range(d):
                diff = q[t] - coarse[c, t]
                s += diff * diff
            ps = _heap_insert(pd, pi, ps, s, c)
        for j in range(m):
            for c in range(ksub):
                s = 0.0
                for t in range(dsub):
                    s += q[j * dsub + t] * codebooks[j, c, t]
                term3[j, c] = s
        size = 0
        for pp in range(ps):
            p = pi[pp]
            base = pd[pp]
            for j in range(m):
                for c in range(ksub):
                    lut[j, c] = term2[p, j, c] - 2.0 * term3[j, c]
            for r in range(offsets[p], offsets[p + 1]):
                s = base
                for j in range(m):
                    s += lut[j, codes[r, j]]
                size = _heap_insert(bd, bi, size, s, ids[r])
            stats[0] += offsets[p + 1] - offsets[p]
        if size == 0:
            continue
        order = np.argsort(bd[:size])
        if rerank > 0:
            exact = np.empty(size, np.float64)
            for j in range(size):
                v = bi[order[j]]
                s = 0.0
                for t in range(d):
                    diff = raw[v, t] - q[t]
                    s += diff * diff
                exact[j] = s
            o2 = np.argsort(exact)
            for j in range(min(k, size)):
                out_i[qi, j] = bi[order[o2[j]]]
                out_d[qi, j] = exact[o2[j]]
        else:
            for j in range(min(k, size)):
                out_i[qi, j] = bi[order[j]]
                out_d[qi, j] = max(bd[order[j]], 0.0)
    return out_i, out_d


class IVFPQIndex:
    """IVF-PQ index (squared L2). Ids are assigned sequentially by :meth:`add`.

    Args:
        dim: vector dimensionality (must be divisible by ``m``).
        nlist: number of coarse cells.
        m: number of PQ sub-quantisers (bytes per code).
        keep_raw: also keep float vectors so :meth:`search` can re-rank exactly.
    """

    KSUB = 256

    def __init__(self, dim: int, nlist: int = 256, m: int = 5, keep_raw: bool = True,
                 seed: int = 0):
        if dim % m != 0:
            raise ValueError(f"dim {dim} must be divisible by m {m}")
        self.dim, self.nlist, self.m = dim, nlist, m
        self.dsub = dim // m
        self.keep_raw = keep_raw
        self.seed = seed
        self.coarse: Optional[np.ndarray] = None
        self.codebooks: Optional[np.ndarray] = None
        self._codes = np.zeros((0, m), np.uint8)
        self._lists = np.zeros(0, np.int64)
        self._raw = np.zeros((0, dim), np.float32)
        self.offsets = np.zeros(nlist + 1, np.int64)
        self.list_codes = np.zeros((0, m), np.uint8)
        self.list_ids = np.zeros(0, np.int64)
        self.stats = np.zeros(1, np.int64)

    @property
    def is_trained(self) -> bool:
        return self.coarse is not None

    @property
    def count(self) -> int:
        return int(self._codes.shape[0])

    def train(self, X: np.ndarray, sample: int = 50_000) -> None:
        """Fit the coarse quantiser and the per-subspace PQ codebooks."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        if X.shape[0] > sample:
            X = X[rng.choice(X.shape[0], sample, replace=False)]
        self.coarse = kmeans(X, self.nlist, seed=self.seed)
        resid = X - self.coarse[assign(X, self.coarse)]
        books = np.empty((self.m, self.KSUB, self.dsub), np.float32)
        for j in range(self.m):
            sl = resid[:, j * self.dsub:(j + 1) * self.dsub]
            books[j] = kmeans(sl, self.KSUB, n_iter=15, seed=self.seed + j + 1)
        self.codebooks = books
        self._precompute_term2()

    def _precompute_term2(self) -> None:
        """term2[p, j, c] = ||b_jc||^2 + 2 <coarse_p (sub-space j), b_jc>."""
        C = self.coarse.reshape(self.nlist, self.m, self.dsub)
        norms = (self.codebooks ** 2).sum(-1)  # [m, ksub]
        cross = np.einsum("pjt,jct->pjc", C, self.codebooks)
        self.term2 = np.ascontiguousarray(norms[None] + 2.0 * cross, dtype=np.float32)

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cell ids, PQ codes) for X."""
        if not self.is_trained:
            raise RuntimeError("train() first")
        X = np.ascontiguousarray(X, dtype=np.float32)
        cells = assign(X, self.coarse)
        resid = X - self.coarse[cells]
        codes = np.empty((X.shape[0], self.m), np.uint8)
        for j in range(self.m):
            sl = resid[:, j * self.dsub:(j + 1) * self.dsub]
            codes[:, j] = assign(sl, self.codebooks[j])
        return cells, codes

    def decode(self, cells: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from (cell, code) pairs."""
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.m)]
        return self.coarse[cells] + np.concatenate(parts, axis=1)

    def add(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        cells, codes = self.encode(X)
        start = self.count
        self._codes = np.concatenate([self._codes, codes])
        self._lists = np.concatenate([self._lists, cells])
        if self.keep_raw:
            self._raw = np.concatenate([self._raw, X])
        # Rebuild CSR layout: vectors grouped by cell so a probe scans contiguous memory.
        order = np.argsort(self._lists, kind="stable")
        self.list_ids = order.astype(np.int64)
        self.list_codes = np.ascontiguousarray(self._codes[order])
        counts = np.bincount(self._lists, minlength=self.nlist)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return np.arange(start, self.count, dtype=np.int64)

    def search(self, Q: np.ndarray, k: int = 10, nprobe: int = 8,
               rerank: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("train() first")
        if rerank and not self.keep_raw:
            raise ValueError("rerank needs keep_raw=True")
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]
        raw = self._raw if self.keep_raw and self._raw.shape[0] else np.zeros((1, self.dim), np.float32)
        return _ivfpq_search(Q, self.coarse, self.codebooks, self.term2, self.offsets,
                             self.list_codes, self.list_ids, raw, int(k), int(nprobe),
                             int(rerank), self.stats)

    def memory_bytes(self, include_raw: bool = False) -> int:
        """Codes + ids + codebooks (+ raw vectors if requested)."""
        b = (self.count * (self.m + 8) + self.coarse.nbytes + self.codebooks.nbytes
             + self.term2.nbytes)
        if include_raw:
            b += self._raw.nbytes
        return int(b)
