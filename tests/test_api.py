import numpy as np
import pytest
from fastapi.testclient import TestClient

from vecdb.server import create_app


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        yield c, tmp_path


def _points(n=300, d=6):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, d))
    return X, [{"id": i, "vector": X[i].tolist(), "payload": {"g": i % 4}} for i in range(n)]


def test_end_to_end(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/collections", json={"name": "docs", "dim": 6})
    assert r.status_code == 201 and r.json()["metric"] == "cosine"
    X, pts = _points()
    r = c.put("/collections/docs/points", json={"points": pts})
    assert r.json() == {"upserted": 300, "lsn": 300}
    r = c.post("/collections/docs/search", json={"vector": X[9].tolist(), "k": 3})
    body = r.json()
    assert body["plan"] == "hnsw" and body["hits"][0]["id"] == "9"
    r = c.post("/collections/docs/search",
               json={"vector": X[9].tolist(), "k": 5, "filter": {"g": {"$in": [2, 3]}}})
    assert all(h["payload"]["g"] in (2, 3) for h in r.json()["hits"])
    assert c.get("/collections/docs/points/9").json()["payload"] == {"g": 1}
    assert c.post("/collections/docs/points/delete", json={"ids": [9]}).json() == {"deleted": 1}
    assert c.get("/collections/docs/points/9").status_code == 404
    assert c.post("/collections/docs/snapshot").json() == {"snapshot_lsn": 301}
    info = c.get("/collections/docs").json()
    assert info["points"] == 299 and info["wal_bytes"] == 0
    assert c.get("/collections").json()["collections"][0]["points"] == 299


def test_restart_reloads_collections(client):
    c, path = client
    c.post("/collections", json={"name": "a", "dim": 6, "metric": "l2"})
    X, pts = _points(50)
    c.put("/collections/a/points", json={"points": pts})
    with TestClient(create_app(str(path))) as c2:
        r = c2.post("/collections/a/search", json={"vector": X[3].tolist(), "k": 1})
        assert r.json()["hits"][0]["id"] == "3"
        assert r.json()["hits"][0]["score"] == pytest.approx(0.0, abs=1e-3)


def test_errors(client):
    c, _ = client
    assert c.post("/collections/nope/search", json={"vector": [1.0]}).status_code == 404
    assert c.post("/collections", json={"name": "bad name!", "dim": 3}).status_code == 400
    assert c.post("/collections", json={"name": "x", "dim": 0}).status_code == 422
    assert c.post("/collections", json={"name": "x", "dim": 3, "metric": "dot"}).status_code == 422
    c.post("/collections", json={"name": "x", "dim": 3})
    assert c.post("/collections", json={"name": "x", "dim": 3}).status_code == 409
    bad = {"points": [{"id": 1, "vector": [1.0, 2.0]}]}
    assert c.put("/collections/x/points", json=bad).status_code == 400
    assert c.put("/collections/x/points", json={"points": []}).status_code == 422
    r = c.post("/collections/x/search", json={"vector": [1, 0, 0], "filter": {"a": {"$gt": 1}}})
    assert r.status_code == 400
    assert c.post("/collections/x/search", json={"vector": [1, 0, 0], "k": 0}).status_code == 422
