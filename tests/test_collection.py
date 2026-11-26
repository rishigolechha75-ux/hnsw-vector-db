import numpy as np
import pytest

from vecdb.collection import Collection, CollectionError


def _data(n=600, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _fill(col, X):
    ids = [f"p{i}" for i in range(len(X))]
    col.upsert(ids, X, [{"color": ["red", "green", "blue"][i % 3], "n": i % 10} for i in range(len(X))])
    return ids


def test_upsert_search_cosine(tmp_path):
    X = _data()
    col = Collection(str(tmp_path / "c"), dim=8)
    _fill(col, X)
    res = col.search(X[10] * 3.0, k=5)  # scale should not matter for cosine
    assert res["plan"] == "hnsw"
    assert res["hits"][0]["id"] == "p10"
    assert res["hits"][0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert res["hits"][0]["payload"]["color"] == "green"
    scores = [h["score"] for h in res["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_l2_metric_scores_are_distances(tmp_path):
    X = _data()
    col = Collection(str(tmp_path / "c"), dim=8, metric="l2")
    _fill(col, X)
    hit = col.search(X[3] + 0.05, k=1)["hits"][0]
    assert hit["id"] == "p3"
    assert hit["score"] == pytest.approx(np.sqrt(8 * 0.0025), rel=1e-4)


def test_upsert_replaces_and_delete(tmp_path):
    X = _data()
    col = Collection(str(tmp_path / "c"), dim=8)
    _fill(col, X)
    col.upsert(["p1"], X[200:201], [{"color": "black"}])
    assert len(col) == len(X)
    hits = col.search(X[200], k=2)["hits"]
    assert {h["id"] for h in hits} == {"p1", "p200"}
    assert col.get("p1")["payload"] == {"color": "black"}
    assert col.delete(["p200", "missing"]) == 1
    assert all(h["id"] != "p200" for h in col.search(X[200], k=10)["hits"])
    assert col.get("p200") is None and len(col) == len(X) - 1


def test_filters_and_planner(tmp_path):
    X = _data(n=2000)
    col = Collection(str(tmp_path / "c"), dim=8, exact_max_points=200)
    _fill(col, X)
    r = col.search(X[0], k=10, filter={"color": "red"})
    assert r["plan"] == "hnsw_filtered"
    assert all(h["payload"]["color"] == "red" for h in r["hits"]) and len(r["hits"]) == 10
    r = col.search(X[0], k=10, filter={"color": "red", "n": {"$in": [0, 3]}})
    assert r["plan"] == "exact_filtered"  # 134 matches <= 200
    assert all(h["payload"]["color"] == "red" and h["payload"]["n"] in (0, 3) for h in r["hits"])
    r = col.search(X[0], k=5, filter={"n": {"$eq": 7}, "color": "blue"})
    assert all(h["payload"]["n"] == 7 for h in r["hits"])
    assert col.search(X[0], k=5, filter={"color": "purple"})["hits"] == []
    with pytest.raises(CollectionError):
        col.search(X[0], filter={"n": {"$gt": 3}})
    with pytest.raises(CollectionError):
        col.search(X[0], filter={"n": [1, 2]})


def test_filter_type_distinctions(tmp_path):
    col = Collection(str(tmp_path / "c"), dim=2)
    col.upsert(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]], [{"v": 1}, {"v": True}, {"v": "1"}])
    for val, want in ((1, "a"), (True, "b"), ("1", "c")):
        hits = col.search([1, 1], k=3, filter={"v": val})["hits"]
        assert [h["id"] for h in hits] == [want]


def test_exact_mode_matches_hnsw_top1(tmp_path):
    X = _data()
    col = Collection(str(tmp_path / "c"), dim=8)
    _fill(col, X)
    a = col.search(X[42], k=3, exact=True)
    assert a["plan"] == "exact" and a["hits"][0]["id"] == "p42"


def test_validation(tmp_path):
    with pytest.raises(CollectionError):
        Collection(str(tmp_path / "x"))
    with pytest.raises(CollectionError):
        Collection(str(tmp_path / "y"), dim=4, metric="dot")
    col = Collection(str(tmp_path / "c"), dim=4)
    with pytest.raises(CollectionError):
        col.upsert(["a"], [[1, 2, 3]])
    with pytest.raises(CollectionError):
        col.upsert(["a"], [[0, 0, 0, 0]])
    with pytest.raises(CollectionError):
        col.upsert(["a"], [[np.nan, 0, 0, 1]])
    with pytest.raises(CollectionError):
        col.upsert(["a", "b"], [[1, 2, 3, 4]])
    with pytest.raises(CollectionError):
        col.search([1, 2, 3, 4], k=0)
    assert col.search([1, 2, 3, 4])["hits"] == []
    with pytest.raises(CollectionError):
        Collection(str(tmp_path / "c"), dim=5)


def test_crash_recovery_from_wal(tmp_path):
    X = _data()
    path = str(tmp_path / "c")
    col = Collection(path, dim=8)
    _fill(col, X)
    col.delete(["p5"])
    col.upsert(["p6"], X[100:101], [{"color": "x"}])
    before = col.search(X[7], k=10)
    # simulate a crash: drop the object without close()/snapshot()
    col.wal._f.flush()
    del col
    again = Collection(path)
    assert again.recovery["replayed"] == len(X) + 2
    assert again.search(X[7], k=10) == before
    assert again.get("p5") is None and again.get("p6")["payload"] == {"color": "x"}


def test_snapshot_then_more_writes_then_recover(tmp_path):
    X = _data()
    path = str(tmp_path / "c")
    col = Collection(path, dim=8)
    _fill(col, X[:400])
    lsn = col.snapshot()
    assert lsn == 400 and col.wal.size_bytes() == 0
    col.upsert([f"q{i}" for i in range(400, 600)], X[400:], [{"n": 1}] * 200)
    col.delete(["p0"])
    col.close()
    again = Collection(path)
    assert again.recovery == {"snapshot_lsn": 400, "replayed": 201, "skipped": 0,
                              "truncated_bytes": 0}
    assert len(again) == 599
    assert again.search(X[450], k=1)["hits"][0]["id"] == "q450"
    assert again.search(X[1], k=3, filter={"color": "green"})["hits"][0]["id"] == "p1"
    # new writes continue the LSN sequence
    again.upsert(["z"], X[:1])
    assert again.lsn == 602


def test_replay_skips_records_already_in_snapshot(tmp_path):
    """Crash after CURRENT was updated but before the WAL was truncated."""
    X = _data(n=50)
    path = str(tmp_path / "c")
    col = Collection(path, dim=8)
    _fill(col, X)
    wal_bytes = open(col.wal.path, "rb").read()
    col.snapshot()
    col.close()
    with open(col.wal.path, "wb") as f:
        f.write(wal_bytes)  # stale log reappears
    again = Collection(path)
    assert again.recovery["skipped"] == 50 and again.recovery["replayed"] == 0
    assert len(again) == 50 and again.index.count == 50
