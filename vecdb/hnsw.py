"""Hierarchical Navigable Small World (HNSW) graph index.

Implements the algorithm from Malkov & Yashunin, "Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs" (2016,
arXiv:1603.09320). The hot loops (graph construction, beam search, neighbour
selection) are compiled with Numba; the Python class only manages memory,
level sampling and persistence.

Memory layout (similar to hnswlib):

* ``vecs``      float32 [capacity, dim]   raw vectors
* ``links0``    int32   [capacity, 2*M]   layer-0 adjacency lists
* ``n0``        int32   [capacity]        layer-0 degree per node
* ``upper``     int32   [slots, M]        adjacency lists for layers >= 1
* ``nup``       int32   [slots]           degree per upper slot
* ``upper_off`` int32   [capacity]        first upper slot of each node (-1 if level 0)

A node with level L owns slots ``upper_off[n] .. upper_off[n] + L - 1`` (one per
layer 1..L). Because only ~1/M of nodes reach layer 1, this is far smaller than a
dense ``[capacity, max_level, M]`` array.

All distances are squared Euclidean. Cosine similarity is handled one level up by
normalising vectors (for unit vectors, ||a-b||^2 = 2 - 2cos(a, b)).
"""
from __future__ import annotations

import heapq
import json
import math
import os
from typing import Optional, Tuple

import numpy as np
from numba import njit

# --------------------------------------------------------------------------------------
# Numba kernels
# --------------------------------------------------------------------------------------


@njit(cache=True, fastmath=True, inline="always")
def _dist(vecs, i, q):
    """Squared L2 distance between stored vector ``i`` and query ``q``."""
    s = 0.0
    for t in range(q.shape[0]):
        diff = vecs[i, t] - q[t]
        s += diff * diff
    return s


@njit(cache=True, inline="always")
def _accept(node, use_filter, allowed, skip_deleted, deleted):
    """Whether ``node`` may appear in the result set (traversal ignores this)."""
    if skip_deleted and deleted[node]:
        return False
    if use_filter and not allowed[node]:
        return False
    return True


@njit(cache=True)
def _neighbors(node, layer, links0, n0, upper, nup, upper_off):
    """View of the adjacency list of ``node`` at ``layer``."""
    if layer == 0:
        return links0[node, : n0[node]]
    slot = upper_off[node] + layer - 1
    return upper[slot, : nup[slot]]


@njit(cache=True)
def _search_layer(q, ep_ids, ef, layer, vecs, links0, n0, upper, nup, upper_off,
                  visited, tagbox, use_filter, allowed, skip_deleted, deleted, stats):
    """Beam search on one layer (Algorithm 2 of the paper).

    Returns ``(dists, ids)`` of up to ``ef`` accepted nodes, sorted ascending.

    ``visited`` is a per-index int32 array; instead of clearing it for every search
    we bump a generation tag (``tagbox[0]``) and treat ``visited[i] == tag`` as
    "seen", which keeps each search O(nodes touched) rather than O(N).

    When filtering (or skipping tombstones), nodes that fail the predicate are still
    traversed so the graph stays connected, but they are not added to the result
    heap. The early-exit test then requires ``ef`` accepted results, which is why a
    very selective filter makes the search wander over much of the graph.
    """
    tagbox[0] += 1
    tag = tagbox[0]
    cand = [(0.0, np.int64(0))]  # min-heap of (dist, id); seeded for type inference
    cand.pop()
    res = [(0.0, np.int64(0))]  # max-heap via (-dist, id)
    res.pop()
    lower = np.inf
    filtering = use_filter or skip_deleted

    for j in range(ep_ids.shape[0]):
        e = np.int64(ep_ids[j])
        if visited[e] == tag:
            continue
        visited[e] = tag
        d = _dist(vecs, e, q)
        stats[0] += 1
        heapq.heappush(cand, (d, e))
        if _accept(e, use_filter, allowed, skip_deleted, deleted):
            heapq.heappush(res, (-d, e))
            if len(res) > ef:
                heapq.heappop(res)
            lower = -res[0][0]

    while len(cand) > 0:
        d, c = heapq.heappop(cand)
        if d > lower and (len(res) >= ef or not filtering):
            break
        nbrs = _neighbors(c, layer, links0, n0, upper, nup, upper_off)
        for t in range(nbrs.shape[0]):
            nb = np.int64(nbrs[t])
            if visited[nb] == tag:
                continue
            visited[nb] = tag
            dn = _dist(vecs, nb, q)
            stats[0] += 1
            if len(res) < ef or dn < lower:
                heapq.heappush(cand, (dn, nb))
                if _accept(nb, use_filter, allowed, skip_deleted, deleted):
                    heapq.heappush(res, (-dn, nb))
                    if len(res) > ef:
                        heapq.heappop(res)
                    lower = -res[0][0]

    n = len(res)
    out_d = np.empty(n, np.float64)
    out_i = np.empty(n, np.int64)
    for j in range(n - 1, -1, -1):
        nd, ni = heapq.heappop(res)
        out_d[j] = -nd
        out_i[j] = ni
    return out_d, out_i


@njit(cache=True)
def _select_heuristic(cand_d, cand_i, M, vecs):
    """Neighbour selection heuristic (Algorithm 4).

    Candidates arrive sorted by distance to the base node. A candidate is kept only
    if it is closer to the base than to every neighbour already kept. This spreads
    edges across directions, which preserves connectivity between clusters, the
    main reason HNSW beats a plain k-NN graph on clustered data.
    """
    n = cand_i.shape[0]
    sel = np.empty(min(M, n), np.int64)
    ns = 0
    for j in range(n):
        if ns >= M:
            break
        c = cand_i[j]
        good = True
        for s in range(ns):
            if _dist(vecs, c, vecs[sel[s]]) < cand_d[j]:
                good = False
                break
        if good:
            sel[ns] = c
            ns += 1
    return sel[:ns]


@njit(cache=True)
def _add_link(src, dst, layer, mmax, vecs, links0, n0, upper, nup, upper_off):
    """Add edge src->dst; if src's list overflows, re-prune it with the heuristic."""
    if layer == 0:
        links = links0
        cnts = n0
        r = src
    else:
        links = upper
        cnts = nup
        r = upper_off[src] + layer - 1
    cnt = cnts[r]
    for j in range(cnt):
        if links[r, j] == dst:
            return
    if cnt < mmax:
        links[r, cnt] = dst
        cnts[r] = cnt + 1
        return
    m = cnt + 1
    ci = np.empty(m, np.int64)
    cd = np.empty(m, np.float64)
    for j in range(cnt):
        ci[j] = links[r, j]
    ci[cnt] = dst
    base = vecs[src]
    for j in range(m):
        cd[j] = _dist(vecs, ci[j], base)
    order = np.argsort(cd)
    sel = _select_heuristic(cd[order], ci[order], mmax, vecs)
    for j in range(sel.shape[0]):
        links[r, j] = sel[j]
    cnts[r] = sel.shape[0]


@njit(cache=True)
def _greedy_descend(q, cur, top_layer, stop_layer, vecs, links0, n0, upper, nup,
                    upper_off, stats):
    """Greedy walk from ``top_layer`` down to ``stop_layer + 1`` (ef = 1)."""
    curd = _dist(vecs, cur, q)
    stats[0] += 1
    for lc in range(top_layer, stop_layer, -1):
        changed = True
        while changed:
            changed = False
            nbrs = _neighbors(cur, lc, links0, n0, upper, nup, upper_off)
            for t in range(nbrs.shape[0]):
                nb = nbrs[t]
                d = _dist(vecs, nb, q)
                stats[0] += 1
                if d < curd:
                    curd = d
                    cur = nb
                    changed = True
    return cur


@njit(cache=True)
def _insert_range(start, end, levels, vecs, links0, n0, upper, nup, upper_off,
                  M, M0, efc, visited, tagbox, state, stats):
    """Insert nodes ``start..end-1`` (Algorithm 1). ``state = [entry, max_level]``."""
    dummy = np.zeros(1, np.uint8)
    for node in range(start, end):
        lvl = levels[node]
        if state[0] < 0:
            state[0] = node
            state[1] = lvl
            continue
        q = vecs[node]
        maxl = state[1]
        cur = _greedy_descend(q, state[0], maxl, lvl, vecs, links0, n0, upper, nup,
                              upper_off, stats)
        eps = np.empty(1, np.int64)
        eps[0] = cur
        for lc in range(min(lvl, maxl), -1, -1):
            cd, ci = _search_layer(q, eps, efc, lc, vecs, links0, n0, upper, nup,
                                   upper_off, visited, tagbox, False, dummy, False,
                                   dummy, stats)
            sel = _select_heuristic(cd, ci, M, vecs)
            if lc == 0:
                for j in range(sel.shape[0]):
                    links0[node, j] = sel[j]
                n0[node] = sel.shape[0]
                mmax = M0
            else:
                slot = upper_off[node] + lc - 1
                for j in range(sel.shape[0]):
                    upper[slot, j] = sel[j]
                nup[slot] = sel.shape[0]
                mmax = M
            for j in range(sel.shape[0]):
                _add_link(sel[j], node, lc, mmax, vecs, links0, n0, upper, nup,
                          upper_off)
            eps = ci
        if lvl > maxl:
            state[0] = node
            state[1] = lvl


@njit(cache=True)
def _search_batch(Q, k, ef, vecs, links0, n0, upper, nup, upper_off, visited, tagbox,
                  state, use_filter, allowed, skip_deleted, deleted, stats):
    """Answer a batch of queries; unfilled slots are id -1, distance +inf."""
    nq = Q.shape[0]
    out_i = np.full((nq, k), -1, np.int64)
    out_d = np.full((nq, k), np.inf, np.float64)
    if state[0] >= 0:
        beam = max(ef, k)
        for qi in range(nq):
            q = Q[qi]
            cur = _greedy_descend(q, state[0], state[1], 0, vecs, links0, n0, upper,
                                  nup, upper_off, stats)
            eps = np.empty(1, np.int64)
            eps[0] = cur
            cd, ci = _search_layer(q, eps, beam, 0, vecs, links0, n0, upper, nup,
                                   upper_off, visited, tagbox, use_filter, allowed,
                                   skip_deleted, deleted, stats)
            m = min(k, ci.shape[0])
            for j in range(m):
                out_i[qi, j] = ci[j]
                out_d[qi, j] = cd[j]
    return out_i, out_d


# --------------------------------------------------------------------------------------
# Python wrapper
# --------------------------------------------------------------------------------------


def _grow(arr: np.ndarray, new_rows: int, fill) -> np.ndarray:
    out = np.full((new_rows,) + arr.shape[1:], fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


class HNSWIndex:
    """HNSW index over float32 vectors with squared-L2 distance.

    Args:
        dim: vector dimensionality.
        M: max out-degree on layers >= 1 (layer 0 allows ``2*M``).
        ef_construction: beam width used while inserting.
        capacity: initial number of preallocated rows (grows by doubling).
        seed: RNG seed for level sampling, making builds reproducible.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 capacity: int = 1024, seed: int = 42):
        if dim <= 0 or M < 2 or ef_construction < 1:
            raise ValueError("dim must be > 0, M >= 2, ef_construction >= 1")
        self.dim = int(dim)
        self.M = int(M)
        self.M0 = 2 * self.M
        self.ef_construction = int(ef_construction)
        self.level_mult = 1.0 / math.log(self.M)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        cap = max(int(capacity), 1)
        self.count = 0
        self.vecs = np.zeros((cap, self.dim), np.float32)
        self.links0 = np.full((cap, self.M0), -1, np.int32)
        self.n0 = np.zeros(cap, np.int32)
        self.levels = np.zeros(cap, np.int32)
        self.upper_off = np.full(cap, -1, np.int32)
        self.deleted = np.zeros(cap, np.uint8)
        self.visited = np.zeros(cap, np.int32)
        self.upper = np.full((max(cap // 8, 16), self.M), -1, np.int32)
        self.nup = np.zeros(self.upper.shape[0], np.int32)
        self.upper_used = 0
        self.tagbox = np.zeros(1, np.int64)
        self.state = np.array([-1, -1], np.int64)  # entry point, max level
        self.stats = np.zeros(1, np.int64)  # distance computations (search only)
        self._build_stats = np.zeros(1, np.int64)

    # ---------------------------------------------------------------- capacity
    def _ensure_capacity(self, n_rows: int) -> None:
        cap = self.vecs.shape[0]
        if n_rows <= cap:
            return
        new = cap
        while new < n_rows:
            new *= 2
        self.vecs = _grow(self.vecs, new, 0.0)
        self.links0 = _grow(self.links0, new, -1)
        self.n0 = _grow(self.n0, new, 0)
        self.levels = _grow(self.levels, new, 0)
        self.upper_off = _grow(self.upper_off, new, -1)
        self.deleted = _grow(self.deleted, new, 0)
        self.visited = _grow(self.visited, new, 0)

    def _ensure_upper(self, n_slots: int) -> None:
        cap = self.upper.shape[0]
        if n_slots <= cap:
            return
        new = cap
        while new < n_slots:
            new *= 2
        self.upper = _grow(self.upper, new, -1)
        self.nup = _grow(self.nup, new, 0)

    # ---------------------------------------------------------------- build
    def add(self, X: np.ndarray) -> np.ndarray:
        """Insert vectors; returns their internal ids (0..n-1 in insertion order)."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {X.shape[1]}")
        n = X.shape[0]
        start, end = self.count, self.count + n
        if n == 0:
            return np.arange(start, start, dtype=np.int64)
        self._ensure_capacity(end)
        # Level ~ floor(-ln(U) * mL): exponentially decaying layer occupancy.
        u = self._rng.random(n)
        lv = np.floor(-np.log(np.maximum(u, 1e-12)) * self.level_mult).astype(np.int32)
        self.levels[start:end] = lv
        offs = np.full(n, -1, np.int32)
        pos = self.upper_used
        for j in np.nonzero(lv > 0)[0]:
            offs[j] = pos
            pos += int(lv[j])
        self._ensure_upper(pos)
        self.upper_off[start:end] = offs
        self.upper_used = pos
        self.vecs[start:end] = X
        _insert_range(start, end, self.levels, self.vecs, self.links0, self.n0,
                      self.upper, self.nup, self.upper_off, self.M, self.M0,
                      self.ef_construction, self.visited, self.tagbox, self.state,
                      self._build_stats)
        self.count = end
        return np.arange(start, end, dtype=np.int64)

    def mark_deleted(self, ids) -> None:
        """Tombstone nodes: they stay in the graph for routing but are never returned."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.count):
            raise IndexError("id out of range")
        self.deleted[ids] = 1

    # ---------------------------------------------------------------- query
    def search(self, Q: np.ndarray, k: int = 10, ef: int = 64,
               allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN search.

        Args:
            Q: [nq, dim] or [dim] queries.
            k: neighbours to return.
            ef: beam width at layer 0 (higher = better recall, slower).
            allowed: optional bool/uint8 mask of length ``count``; only nodes with a
                truthy entry are returned (filtered search).

        Returns:
            ``(ids, dists)`` each [nq, k]; missing entries are -1 / inf.
        """
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]
        if Q.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {Q.shape[1]}")
        if k < 1:
            raise ValueError("k must be >= 1")
        use_filter = allowed is not None
        if use_filter:
            mask = np.zeros(self.vecs.shape[0], np.uint8)
            a = np.asarray(allowed).astype(np.uint8)
            mask[: a.shape[0]] = a
        else:
            mask = np.zeros(1, np.uint8)
        skip_deleted = bool(self.deleted[: self.count].any())
        return _search_batch(Q, int(k), int(ef), self.vecs, self.links0, self.n0,
                             self.upper, self.nup, self.upper_off, self.visited,
                             self.tagbox, self.state, use_filter, mask, skip_deleted,
                             self.deleted, self.stats)

    # ---------------------------------------------------------------- introspection
    @property
    def max_level(self) -> int:
        return int(self.state[1])

    def memory_bytes(self) -> int:
        """Bytes used by vectors + graph for the live rows (excluding slack)."""
        n = self.count
        vec = n * self.dim * 4
        graph = n * (self.M0 * 4 + 4 + 4 + 4 + 1) + self.upper_used * (self.M * 4 + 4)
        return int(vec + graph)

    def graph_stats(self) -> dict:
        n = self.count
        deg0 = self.n0[:n]
        return {
            "count": n,
            "max_level": self.max_level,
            "mean_degree_l0": float(deg0.mean()) if n else 0.0,
            "nodes_above_l0": int((self.levels[:n] > 0).sum()),
            "deleted": int(self.deleted[:n].sum()),
            "memory_bytes": self.memory_bytes(),
        }

    # ---------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Write the live part of the index to a single ``.npz`` file."""
        n, s = self.count, self.upper_used
        meta = {"dim": self.dim, "M": self.M, "ef_construction": self.ef_construction,
                "seed": self.seed, "count": n, "upper_used": s,
                "state": self.state.tolist()}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, vecs=self.vecs[:n], links0=self.links0[:n], n0=self.n0[:n],
                     levels=self.levels[:n], upper_off=self.upper_off[:n],
                     deleted=self.deleted[:n], upper=self.upper[:s], nup=self.nup[:s],
                     rng=np.frombuffer(json.dumps(self._rng.bit_generator.state).encode(),
                                       dtype=np.uint8),
                     meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        z = np.load(path)
        meta = json.loads(z["meta"].tobytes().decode())
        n, s = meta["count"], meta["upper_used"]
        idx = cls(meta["dim"], meta["M"], meta["ef_construction"],
                  capacity=max(n, 1), seed=meta["seed"])
        idx._ensure_upper(max(s, 1))
        idx.vecs[:n] = z["vecs"]
        idx.links0[:n] = z["links0"]
        idx.n0[:n] = z["n0"]
        idx.levels[:n] = z["levels"]
        idx.upper_off[:n] = z["upper_off"]
        idx.deleted[:n] = z["deleted"]
        idx.upper[:s] = z["upper"]
        idx.nup[:s] = z["nup"]
        idx.count, idx.upper_used = n, s
        idx.state[:] = meta["state"]
        idx._rng.bit_generator.state = json.loads(z["rng"].tobytes().decode())
        return idx
