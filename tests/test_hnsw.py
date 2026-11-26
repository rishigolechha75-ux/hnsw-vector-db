import numpy as np
import pytest

from vecdb.flat import topk_sq_l2
from vecdb.hnsw import HNSWIndex

from .conftest import recall


@pytest.fixture(scope="module")
def built(clustered):
    X, Q = clustered
    idx = HNSWIndex(X.shape[1], M=12, ef_construction=100, capacity=64, seed=3)
    idx.add(X[:1000])
    idx.add(X[1000:])  # second batch exercises capacity growth
    return idx, X, Q


def test_recall_high_and_monotonic_in_ef(built):
    idx, X, Q = built
    gt, _ = topk_sq_l2(X, Q, 10)
    recalls = [recall(idx.search(Q, 10, ef)[0], gt) for ef in (10, 40, 200)]
    assert recalls[-1] >= 0.98
    assert recalls[0] <= recalls[1] + 0.02 <= recalls[2] + 0.04


def test_distances_sorted_and_exact(built):
    idx, X, Q = built
    I, D = idx.search(Q[:5], 10, 64)
    assert (np.diff(D, axis=1) >= 0).all()
    true = ((X[I[0]] - Q[0]) ** 2).sum(1)
    np.testing.assert_allclose(D[0], true, rtol=1e-4, atol=1e-4)


def test_graph_invariants(built):
    idx, X, _ = built
    n = idx.count
    assert n == len(X)
    for node in range(n):
        row = idx.links0[node, : idx.n0[node]]
        assert idx.n0[node] <= idx.M0
        assert node not in row
        assert len(set(row.tolist())) == len(row)
        assert (row >= 0).all() and (row < n).all()
    assert (idx.nup[: idx.upper_used] <= idx.M).all()
    # every node with level > 0 owns exactly `level` upper slots
    lv = idx.levels[:n]
    assert idx.upper_used == lv.sum()
    assert ((idx.upper_off[:n] >= 0) == (lv > 0)).all()


def test_every_node_reachable_at_layer0(built):
    idx, _, _ = built
    n = idx.count
    seen = np.zeros(n, bool)
    stack = [int(idx.state[0])]
    seen[stack[0]] = True
    while stack:
        u = stack.pop()
        for v in idx.links0[u, : idx.n0[u]]:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    assert seen.mean() > 0.995


def test_self_query_finds_itself(built):
    idx, X, _ = built
    I, D = idx.search(X[:50], 1, 32)
    assert (I[:, 0] == np.arange(50)).mean() >= 0.98


def test_filter_only_returns_allowed(built):
    idx, X, Q = built
    allowed = np.zeros(len(X), bool)
    allowed[::7] = True
    I, _ = idx.search(Q, 10, 64, allowed=allowed)
    assert allowed[I[I >= 0]].all()
    ids = np.nonzero(allowed)[0]
    gt, _ = topk_sq_l2(X[ids], Q, 10, ids=ids)
    assert recall(I, gt) >= 0.95


def test_filter_matching_fewer_than_k(built):
    idx, X, Q = built
    allowed = np.zeros(len(X), bool)
    allowed[[5, 17, 300]] = True
    I, D = idx.search(Q[:3], 10, 32, allowed=allowed)
    for row, drow in zip(I, D):
        assert set(row[:3]) == {5, 17, 300}
        assert (row[3:] == -1).all() and np.isinf(drow[3:]).all()


def test_deleted_never_returned():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((800, 8)).astype(np.float32)
    idx = HNSWIndex(8, M=8, ef_construction=64)
    idx.add(X)
    dead = np.arange(0, 800, 3)
    idx.mark_deleted(dead)
    I, _ = idx.search(X[:100], 10, 64)
    assert not np.isin(I, dead).any()
    with pytest.raises(IndexError):
        idx.mark_deleted([800])


def test_empty_and_single_element():
    idx = HNSWIndex(4)
    I, D = idx.search(np.ones(4), 3)
    assert (I == -1).all()
    idx.add(np.array([[1, 0, 0, 0]], np.float32))
    I, D = idx.search(np.ones(4), 3)
    assert I[0, 0] == 0 and (I[0, 1:] == -1).all()
    assert D[0, 0] == pytest.approx(3.0)


def test_duplicate_vectors():
    X = np.ones((200, 4), np.float32)
    idx = HNSWIndex(4, M=4, ef_construction=16)
    idx.add(X)
    I, D = idx.search(X[0], 5, 16)
    assert (I >= 0).all() and np.allclose(D, 0)


def test_input_validation():
    idx = HNSWIndex(4)
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 5)))
    with pytest.raises(ValueError):
        idx.search(np.ones(5))
    with pytest.raises(ValueError):
        idx.search(np.ones(4), k=0)
    with pytest.raises(ValueError):
        HNSWIndex(0)
    assert idx.add(np.zeros((0, 4))).size == 0


def test_save_load_roundtrip(tmp_path, built):
    idx, X, Q = built
    p = str(tmp_path / "i.npz")
    idx.save(p)
    other = HNSWIndex.load(p)
    a = idx.search(Q, 10, 50)
    b = other.search(Q, 10, 50)
    np.testing.assert_array_equal(a[0], b[0])
    # level RNG state is restored, so continued inserts match too
    extra = np.random.default_rng(9).standard_normal((50, X.shape[1])).astype(np.float32)
    idx2 = HNSWIndex.load(p)
    idx2.add(extra)
    other.add(extra)
    np.testing.assert_array_equal(idx2.levels[: idx2.count], other.levels[: other.count])


def test_deterministic_build(clustered):
    X, Q = clustered
    a = HNSWIndex(16, 8, 50, seed=1)
    b = HNSWIndex(16, 8, 50, seed=1)
    a.add(X[:500])
    b.add(X[:500])
    np.testing.assert_array_equal(a.links0[:500], b.links0[:500])
