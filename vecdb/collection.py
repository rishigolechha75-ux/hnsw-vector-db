"""A durable collection: HNSW index + payloads + write-ahead log + snapshots.

Write path:  validate -> append to WAL (flush) -> apply to in-memory index.
Read path:   build filter mask from the payload index -> choose a plan -> search.
Recovery:    load latest snapshot (if any) -> replay WAL records with lsn > snapshot lsn.

The query planner picks between two strategies for filtered queries:

* ``hnsw_filtered``: walk the full graph but only admit matching points into the
  result set. Cheap when many points match, but when few do the beam has to wander
  through lots of non-matching nodes before it collects ``ef`` matches.
* ``exact_filtered``: brute-force over just the matching points. Cost is linear in
  the number of matches, so it wins when the filter is selective.

``exact_max_points`` is the number of matching points at or below which the exact
plan is used. An absolute count (rather than a selectivity fraction) is used because
brute-force cost grows with the number of matches, while filtered-HNSW cost depends
mostly on selectivity and only weakly (~log n) on collection size. The default is
taken from the measured crossover in reports/*/filtered.json (see README).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .flat import topk_sq_l2
from .hnsw import HNSWIndex
from .wal import OP_DELETE, OP_UPSERT, Record, WriteAheadLog

METRICS = ("cosine", "l2")
SCALARS = (str, int, float, bool)
EXACT_MAX_POINTS_DEFAULT = 10_000


def _key(v: Any) -> str:
    """Canonical payload-index key; json keeps 1, 1.0, True and "1" distinct."""
    return json.dumps(v)


class CollectionError(ValueError):
    """Raised for invalid user input (bad dimension, unknown metric, bad filter)."""


class Collection:
    """Open (or create) a collection stored in directory ``path``.

    Args:
        path: directory holding ``config.json``, ``wal.log`` and snapshots.
        dim: required when creating; ignored (must match) when reopening.
        metric: ``"cosine"`` (vectors are L2-normalised) or ``"l2"``.
        M, ef_construction: HNSW build parameters.
        fsync: fsync the WAL on every write batch.
        exact_max_points: filtered queries matching at most this many points use
            brute force over the matches instead of the graph.
    """

    def __init__(self, path: str, dim: Optional[int] = None, metric: str = "cosine",
                 M: int = 16, ef_construction: int = 200, fsync: bool = False,
                 exact_max_points: int = EXACT_MAX_POINTS_DEFAULT):
        self.path = path
        os.makedirs(path, exist_ok=True)
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if dim is not None and dim != cfg["dim"]:
                raise CollectionError(f"collection has dim {cfg['dim']}, not {dim}")
        else:
            if dim is None or dim <= 0:
                raise CollectionError("dim is required to create a collection")
            if metric not in METRICS:
                raise CollectionError(f"metric must be one of {METRICS}")
            cfg = {"dim": int(dim), "metric": metric, "M": int(M),
                   "ef_construction": int(ef_construction)}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
        self.config = cfg
        self.dim = cfg["dim"]
        self.metric = cfg["metric"]
        self.exact_max_points = int(exact_max_points)
        self._lock = threading.RLock()
        self._reset_state()
        self.wal = WriteAheadLog(os.path.join(path, "wal.log"), fsync=fsync)
        self.recovery = self._recover()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.index = HNSWIndex(self.dim, self.config["M"], self.config["ef_construction"])
        self.int_to_ext: List[str] = []
        self.ext_to_int: Dict[str, int] = {}
        self.payloads: List[dict] = []
        self.payload_index: Dict[str, Dict[str, List[int]]] = {}
        self.lsn = 0
        self.snapshot_lsn = 0

    def _index_payload(self, internal: int, payload: dict) -> None:
        for field, value in payload.items():
            if isinstance(value, SCALARS):
                self.payload_index.setdefault(field, {}).setdefault(_key(value), []).append(internal)

    def _prepare(self, vectors) -> np.ndarray:
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise CollectionError(f"vectors must have dimension {self.dim}")
        if not np.isfinite(X).all():
            raise CollectionError("vectors must be finite")
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            if (norms == 0).any():
                raise CollectionError("zero vector cannot be normalised for cosine")
            X = X / norms
        return np.ascontiguousarray(X, dtype=np.float32)

    # ------------------------------------------------------------------ apply (no WAL)
    def _apply_upserts(self, ids: List[str], X: np.ndarray, payloads: List[dict]) -> None:
        base = self.index.count
        tombstones = []
        for j, ext in enumerate(ids):
            old = self.ext_to_int.get(ext)
            if old is not None:
                tombstones.append(old)
            self.ext_to_int[ext] = base + j
            self.int_to_ext.append(ext)
            self.payloads.append(payloads[j])
            self._index_payload(base + j, payloads[j])
        self.index.add(X)
        if tombstones:
            self.index.mark_deleted(tombstones)

    def _apply_delete(self, ext: str) -> bool:
        internal = self.ext_to_int.pop(ext, None)
        if internal is None:
            return False
        self.index.mark_deleted([internal])
        return True

    def _apply_records(self, records: Sequence[Record]) -> None:
        batch_ids, batch_vecs, batch_pl = [], [], []

        def flush():
            if batch_ids:
                self._apply_upserts(list(batch_ids), np.stack(batch_vecs), list(batch_pl))
                batch_ids.clear(); batch_vecs.clear(); batch_pl.clear()

        for r in records:
            if r.op == OP_UPSERT:
                batch_ids.append(r.header["id"])
                batch_vecs.append(r.vector)
                batch_pl.append(r.header.get("payload") or {})
            elif r.op == OP_DELETE:
                flush()
                self._apply_delete(r.header["id"])
            self.lsn = max(self.lsn, r.lsn)
        flush()

    # ------------------------------------------------------------------ recovery
    def _snapshot_files(self, lsn: int):
        return (os.path.join(self.path, f"snap-{lsn}.npz"),
                os.path.join(self.path, f"snap-{lsn}.json"))

    def _recover(self) -> dict:
        current = os.path.join(self.path, "CURRENT")
        if os.path.exists(current):
            with open(current) as f:
                snap_lsn = int(f.read().strip())
            npz, meta_path = self._snapshot_files(snap_lsn)
            self.index = HNSWIndex.load(npz)
            with open(meta_path) as f:
                meta = json.load(f)
            self.int_to_ext = meta["int_to_ext"]
            self.payloads = meta["payloads"]
            deleted = self.index.deleted
            self.ext_to_int = {e: i for i, e in enumerate(self.int_to_ext) if not deleted[i]}
            for i, p in enumerate(self.payloads):
                self._index_payload(i, p)
            self.lsn = self.snapshot_lsn = snap_lsn
        replay = self.wal.replay()
        fresh = [r for r in replay.records if r.lsn > self.snapshot_lsn]
        self._apply_records(fresh)
        return {"snapshot_lsn": self.snapshot_lsn, "replayed": len(fresh),
                "skipped": len(replay.records) - len(fresh),
                "truncated_bytes": replay.truncated_bytes}

    # ------------------------------------------------------------------ public API
    def upsert(self, ids: Sequence, vectors, payloads: Optional[Sequence[dict]] = None) -> int:
        """Insert or replace points. Returns the number written."""
        with self._lock:
            ids = [str(i) for i in ids]
            X = self._prepare(vectors)
            if len(ids) != X.shape[0]:
                raise CollectionError("ids and vectors must have the same length")
            payloads = [dict(p or {}) for p in (payloads or [{}] * len(ids))]
            if len(payloads) != len(ids):
                raise CollectionError("payloads and ids must have the same length")
            recs = []
            for j, ext in enumerate(ids):
                self.lsn += 1
                recs.append(Record(self.lsn, OP_UPSERT, {"id": ext, "payload": payloads[j]}, X[j]))
            self.wal.append(recs)
            self._apply_upserts(ids, X, payloads)
            return len(ids)

    def delete(self, ids: Sequence) -> int:
        """Delete points by id. Returns how many existed."""
        with self._lock:
            ids = [str(i) for i in ids]
            recs = []
            for ext in ids:
                self.lsn += 1
                recs.append(Record(self.lsn, OP_DELETE, {"id": ext}))
            self.wal.append(recs)
            return sum(self._apply_delete(e) for e in ids)

    def get(self, ext_id) -> Optional[dict]:
        with self._lock:
            i = self.ext_to_int.get(str(ext_id))
            if i is None:
                return None
            return {"id": self.int_to_ext[i], "vector": self.index.vecs[i].tolist(),
                    "payload": self.payloads[i]}

    def filter_mask(self, flt: Optional[dict]) -> Optional[np.ndarray]:
        """Turn ``{"field": value | {"$eq": v} | {"$in": [..]}}`` (AND across fields)
        into a boolean mask over internal ids. ``None`` means no filter."""
        if not flt:
            return None
        n = self.index.count
        mask = np.ones(n, bool)
        for field, cond in flt.items():
            if isinstance(cond, dict):
                if set(cond) == {"$eq"}:
                    values = [cond["$eq"]]
                elif set(cond) == {"$in"} and isinstance(cond["$in"], list):
                    values = cond["$in"]
                else:
                    raise CollectionError(f"unsupported condition for {field!r}: {cond}")
            else:
                values = [cond]
            if not all(isinstance(v, SCALARS) for v in values):
                raise CollectionError("filter values must be scalars")
            m = np.zeros(n, bool)
            postings = self.payload_index.get(field, {})
            for v in values:
                hits = postings.get(_key(v))
                if hits:
                    m[np.asarray(hits, np.int64)] = True
            mask &= m
        return mask

    def search(self, vector, k: int = 10, ef: int = 64, filter: Optional[dict] = None,
               exact: bool = False) -> dict:
        """Nearest neighbours of ``vector``.

        Returns ``{"plan": str, "hits": [{"id", "score", "payload"}]}``. ``score`` is
        cosine similarity for ``cosine`` collections and Euclidean distance for ``l2``.
        """
        with self._lock:
            if k < 1:
                raise CollectionError("k must be >= 1")
            q = self._prepare(vector)
            n = self.index.count
            live = self.index.deleted[:n] == 0
            mask = self.filter_mask(filter)
            if mask is not None:
                mask &= live
            n_live = int(live.sum())
            if n_live == 0:
                return {"plan": "empty", "hits": []}
            if exact:
                plan = "exact"
            elif mask is None:
                plan = "hnsw"
            elif int(mask.sum()) <= self.exact_max_points:
                plan = "exact_filtered"
            else:
                plan = "hnsw_filtered"
            if plan.startswith("exact"):
                cand = np.nonzero(mask if mask is not None else live)[0]
                I, D = topk_sq_l2(self.index.vecs[cand], q, k, ids=cand)
            else:
                I, D = self.index.search(q, k=k, ef=ef, allowed=mask)
            hits = []
            for i, d in zip(I[0], D[0]):
                if i < 0:
                    break
                score = 1.0 - d / 2.0 if self.metric == "cosine" else float(np.sqrt(d))
                hits.append({"id": self.int_to_ext[i], "score": float(score),
                             "payload": self.payloads[i]})
            return {"plan": plan, "hits": hits}

    def snapshot(self) -> int:
        """Persist full state and truncate the WAL. Returns the snapshot LSN."""
        with self._lock:
            lsn = self.lsn
            npz, meta_path = self._snapshot_files(lsn)
            self.index.save(npz)
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"int_to_ext": self.int_to_ext, "payloads": self.payloads}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, meta_path)
            cur_tmp = os.path.join(self.path, "CURRENT.tmp")
            with open(cur_tmp, "w") as f:
                f.write(str(lsn))
                f.flush()
                os.fsync(f.fileno())
            os.replace(cur_tmp, os.path.join(self.path, "CURRENT"))  # atomic commit point
            for name in os.listdir(self.path):
                if name.startswith("snap-") and not name.startswith(f"snap-{lsn}."):
                    os.remove(os.path.join(self.path, name))
            self.wal.reset()
            self.snapshot_lsn = lsn
            return lsn

    def stats(self) -> dict:
        with self._lock:
            return {"dim": self.dim, "metric": self.metric, "points": len(self.ext_to_int),
                    "lsn": self.lsn, "snapshot_lsn": self.snapshot_lsn,
                    "wal_bytes": self.wal.size_bytes(), "index": self.index.graph_stats(),
                    "exact_max_points": self.exact_max_points}

    def close(self) -> None:
        self.wal.close()

    def __len__(self) -> int:
        return len(self.ext_to_int)
