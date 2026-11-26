import numpy as np
import pytest

from vecdb.flat import topk_sq_l2
from vecdb.ivfpq import IVFPQIndex, assign, kmeans

from .conftest import recall


@pytest.fixture(scope="module")
def pq(clustered):
    X, Q = clustered
    idx = IVFPQIndex(16, nlist=32, m=4)
    idx.train(X)
    idx.add(X[:2000])
    idx.add(X[2000:])
    return idx, X, Q


def test_kmeans_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [-10, 10]], np.float32)
    X = np.concatenate([c + rng.standard_normal((200, 2)) for c in centers]).astype(np.float32)
    C = kmeans(X, 3, seed=1)
    d = ((C[:, None] - centers[None]) ** 2).sum(-1)
    assert d.min(1).max() < 0.5
    with pytest.raises(ValueError):
        kmeans(X[:2], 3)


def test_kmeans_handles_duplicates_without_nan():
    X = np.zeros((100, 3), np.float32)
    X[:50] = 1
    C = kmeans(X, 4)
    assert np.isfinite(C).all()


def test_adc_distance_equals_decoded_distance(pq):
    idx, X, Q = pq
    I, D = idx.search(Q[:5], 5, nprobe=32)
    cells, codes = idx.encode(X)
    R = idx.decode(cells, codes)
    for qi in range(5):
        true = ((R[I[qi]] - Q[qi]) ** 2).sum(1)
        np.testing.assert_allclose(D[qi], true, rtol=1e-3, atol=1e-3)


def test_quantisation_beats_coarse_only(pq):
    idx, X, _ = pq
    cells, codes = idx.encode(X)
    err_pq = ((idx.decode(cells, codes) - X) ** 2).sum(1).mean()
    err_coarse = ((idx.coarse[assign(X, idx.coarse)] - X) ** 2).sum(1).mean()
    assert err_pq < 0.5 * err_coarse


def test_recall_grows_with_nprobe_and_rerank(pq):
    idx, X, Q = pq
    gt, _ = topk_sq_l2(X, Q, 10)
    r1 = recall(idx.search(Q, 10, 1)[0], gt)
    r32 = recall(idx.search(Q, 10, 32)[0], gt)
    rr = recall(idx.search(Q, 10, 32, rerank=100)[0], gt)
    assert r1 <= r32 <= rr
    assert rr >= 0.95
    assert idx.count == len(X) and idx.offsets[-1] == len(X)


def test_errors():
    idx = IVFPQIndex(16, nlist=4, m=4)
    with pytest.raises(RuntimeError):
        idx.search(np.ones(16))
    with pytest.raises(ValueError):
        IVFPQIndex(10, m=4)
    no_raw = IVFPQIndex(4, nlist=2, m=2, keep_raw=False)
    no_raw.train(np.random.default_rng(0).standard_normal((600, 4)))
    with pytest.raises(ValueError):
        no_raw.search(np.ones(4), rerank=5)
