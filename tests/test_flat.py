import numpy as np

from vecdb.flat import FlatIndex, topk_sq_l2


def test_matches_naive_sort():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    Q = rng.standard_normal((20, 8)).astype(np.float32)
    I, D = topk_sq_l2(X, Q, 7)
    full = ((Q[:, None] - X[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(I, np.argsort(full, 1)[:, :7])
    np.testing.assert_allclose(D, np.sort(full, 1)[:, :7], rtol=1e-4, atol=1e-4)


def test_k_larger_than_n_pads():
    X = np.eye(3, dtype=np.float32)
    I, D = topk_sq_l2(X, X[:1], 5)
    assert list(I[0, :3]) == [0, 1, 2] or I[0, 0] == 0
    assert (I[0, 3:] == -1).all() and np.isinf(D[0, 3:]).all()


def test_empty_database():
    I, D = topk_sq_l2(np.zeros((0, 4), np.float32), np.ones((2, 4), np.float32), 3)
    assert (I == -1).all() and np.isinf(D).all()


def test_custom_ids_and_index():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    I, _ = topk_sq_l2(X, X[2:3], 1, ids=np.arange(100, 106))
    assert I[0, 0] == 102
    f = FlatIndex(2)
    f.add(X)
    assert f.count == 6 and f.search(X[5], 1)[0][0, 0] == 5
