"""Reproducible benchmarks on GloVe-25 (angular). Writes reports/<tag>/*.json.

    python benchmarks/run_benchmarks.py --n 100000 --queries 1000

Everything runs single-threaded (BLAS, Numba and FAISS) so numbers compare
algorithms, not core counts. Each timed search is repeated ``--repeats`` times and
the fastest run is kept, which filters out scheduler noise but is still wall clock.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vecdb.collection import Collection  # noqa: E402
from vecdb.datasets import load_glove25  # noqa: E402
from vecdb.flat import topk_sq_l2  # noqa: E402
from vecdb.hnsw import HNSWIndex  # noqa: E402
from vecdb.ivfpq import IVFPQIndex  # noqa: E402

try:
    import faiss  # type: ignore

    faiss.omp_set_num_threads(1)
except ImportError:  # optional reference implementation
    faiss = None

EFS = [10, 16, 32, 64, 128, 256, 512]
NPROBES = [1, 2, 4, 8, 16, 32, 64]


def recall_at_k(I: np.ndarray, gt: np.ndarray) -> float:
    k = gt.shape[1]
    hits = sum(len(set(a[a >= 0]) & set(b)) for a, b in zip(I[:, :k], gt))
    return hits / gt.size


def timed(fn, repeats: int):
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def warm_up_jit(dim: int) -> None:
    """Compile Numba kernels on toy data so build times exclude JIT compilation."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, dim)).astype(np.float32)
    h = HNSWIndex(dim, 8, 32)
    h.add(X)
    h.search(X[:2], 5, 16)
    h.search(X[:2], 5, 16, allowed=np.ones(300, bool))
    p = IVFPQIndex(dim, nlist=4, m=5 if dim % 5 == 0 else 1)
    p.train(np.tile(X, (2, 1)))
    p.add(X)
    p.search(X[:2], 5, 2, rerank=10)


def bench_flat(X, Q, gt, k, repeats):
    t, (I, _) = timed(lambda: topk_sq_l2(X, Q, k), repeats)
    return {"method": "exact (numpy)", "param": None, "recall": recall_at_k(I, gt),
            "qps": len(Q) / t, "memory_bytes": int(X.nbytes)}


def bench_hnsw(X, Q, gt, k, repeats, M, efc):
    idx = HNSWIndex(X.shape[1], M, efc, capacity=len(X))
    t0 = time.perf_counter()
    idx.add(X)
    build = time.perf_counter() - t0
    rows = []
    for ef in EFS:
        s0 = int(idx.stats[0])
        idx.search(Q, k, ef)
        comps = (int(idx.stats[0]) - s0) / len(Q)
        t, (I, _) = timed(lambda: idx.search(Q, k, ef), repeats)
        rows.append({"method": "vecdb HNSW", "param": ef, "recall": recall_at_k(I, gt),
                     "qps": len(Q) / t, "dist_comps": comps})
    info = {"build_s": build, **idx.graph_stats()}
    return idx, rows, info


def bench_faiss_hnsw(X, Q, gt, k, repeats, M, efc):
    idx = faiss.IndexHNSWFlat(X.shape[1], M)
    idx.hnsw.efConstruction = efc
    t0 = time.perf_counter()
    idx.add(X)
    build = time.perf_counter() - t0
    rows = []
    for ef in EFS:
        idx.hnsw.efSearch = max(ef, k)
        t, (_, I) = timed(lambda: idx.search(Q, k), repeats)
        rows.append({"method": "FAISS HNSWFlat", "param": ef, "recall": recall_at_k(I, gt),
                     "qps": len(Q) / t})
    return rows, {"build_s": build}


def bench_ivfpq(X, Q, gt, k, repeats, nlist, m):
    idx = IVFPQIndex(X.shape[1], nlist=nlist, m=m, keep_raw=True)
    t0 = time.perf_counter()
    idx.train(X)
    train = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.add(X)
    add = time.perf_counter() - t0
    rows = []
    for rerank, label in ((0, "vecdb IVF-PQ"), (100, "vecdb IVF-PQ + rerank100")):
        for nprobe in NPROBES:
            t, (I, _) = timed(lambda: idx.search(Q, k, nprobe, rerank), repeats)
            rows.append({"method": label, "param": nprobe, "recall": recall_at_k(I, gt),
                         "qps": len(Q) / t})
    info = {"train_s": train, "add_s": add, "code_bytes_per_vector": m,
            "memory_bytes_codes": idx.memory_bytes(False),
            "memory_bytes_with_raw": idx.memory_bytes(True)}
    return rows, info


def bench_faiss_ivfpq(X, Q, gt, k, repeats, nlist, m):
    quant = faiss.IndexFlatL2(X.shape[1])
    idx = faiss.IndexIVFPQ(quant, X.shape[1], nlist, m, 8)
    t0 = time.perf_counter()
    idx.train(X[np.random.default_rng(0).choice(len(X), min(len(X), 50_000), replace=False)])
    idx.add(X)
    build = time.perf_counter() - t0
    rows = []
    for nprobe in NPROBES:
        idx.nprobe = nprobe
        t, (_, I) = timed(lambda: idx.search(Q, k), repeats)
        rows.append({"method": "FAISS IVF-PQ", "param": nprobe, "recall": recall_at_k(I, gt),
                     "qps": len(Q) / t})
    return rows, {"build_s": build}


def bench_filtered(idx: HNSWIndex, X, Q, k, repeats, ef=64):
    """HNSW-with-filter vs brute force over matching points, across selectivities."""
    rng = np.random.default_rng(7)
    u = rng.random(len(X))
    rows = []
    for sel in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        mask = u < sel
        ids = np.nonzero(mask)[0]
        gt, _ = topk_sq_l2(X[ids], Q, k, ids=ids)
        s0 = int(idx.stats[0])
        idx.search(Q, k, ef, allowed=mask)
        comps = (int(idx.stats[0]) - s0) / len(Q)
        th, (Ih, _) = timed(lambda: idx.search(Q, k, ef, allowed=mask), repeats)
        te, (Ie, _) = timed(lambda: topk_sq_l2(X[ids], Q, k, ids=ids), repeats)
        rows.append({"selectivity": sel, "matching": int(mask.sum()),
                     "hnsw_recall": recall_at_k(Ih, gt), "hnsw_qps": len(Q) / th,
                     "hnsw_dist_comps": comps,
                     "exact_recall": recall_at_k(Ie, gt), "exact_qps": len(Q) / te})
    return rows


def bench_deletes(X, Q, k, M, efc, ef=64):
    """Recall when a fraction of points is tombstoned (graph is not repaired)."""
    rows = []
    rng = np.random.default_rng(11)
    idx = HNSWIndex(X.shape[1], M, efc, capacity=len(X))
    idx.add(X)
    order = rng.permutation(len(X))
    done = 0
    for frac in [0.0, 0.1, 0.3, 0.5]:
        target = int(frac * len(X))
        if target > done:
            idx.mark_deleted(order[done:target])
            done = target
        live = np.nonzero(idx.deleted[: len(X)] == 0)[0]
        gt, _ = topk_sq_l2(X[live], Q, k, ids=live)
        I, _ = idx.search(Q, k, ef)
        rows.append({"deleted_fraction": frac, "recall": recall_at_k(I, gt)})
    return rows


def bench_durability(X, tmp_root, batch=1000, n=20_000):
    """WAL write throughput and recovery time: WAL replay vs snapshot load."""
    X = X[:n]
    out = {"points": n, "batch": batch}
    from vecdb.wal import OP_UPSERT, Record, WriteAheadLog
    # (fsync, records per append): batching amortises one sync over many records
    for fsync, b, count in ((False, batch, n), (True, batch, n), (True, 1, 2000)):
        wal = WriteAheadLog(os.path.join(tmp_root, f"wal-{fsync}-{b}.log"), fsync=fsync)
        wal.replay()
        recs = [Record(i + 1, OP_UPSERT, {"id": str(i), "payload": {}}, X[i]) for i in range(count)]
        t0 = time.perf_counter()
        for s in range(0, count, b):
            wal.append(recs[s:s + b])
        out[f"wal_points_per_s_fsync_{fsync}_batch_{b}"] = count / (time.perf_counter() - t0)
        wal.close()
    d = os.path.join(tmp_root, "recover")
    col = Collection(d, dim=X.shape[1])
    for s in range(0, n, batch):
        col.upsert([str(i) for i in range(s, s + batch)], X[s:s + batch],
                   [{"tag": i % 10} for i in range(s, s + batch)])
    col.close()
    t0 = time.perf_counter()
    c2 = Collection(d)
    out["recover_from_wal_s"] = time.perf_counter() - t0
    out["wal_bytes"] = c2.wal.size_bytes()
    c2.snapshot()
    c2.close()
    t0 = time.perf_counter()
    c3 = Collection(d)
    out["recover_from_snapshot_s"] = time.perf_counter() - t0
    out["recovered_points"] = len(c3)
    c3.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--nlist", type=int, default=0, help="0 = 4*sqrt(n) rounded to pow2")
    ap.add_argument("--pq-m", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--skip-extras", action="store_true",
                    help="only the recall/QPS sweep (for large n)")
    args = ap.parse_args()

    X, Q = load_glove25(args.n, args.queries)
    out_dir = os.path.join(args.out, f"glove25_n{len(X)}")
    os.makedirs(out_dir, exist_ok=True)
    k = args.k
    nlist = args.nlist or int(2 ** round(np.log2(4 * np.sqrt(len(X)))))
    print(f"base={X.shape} queries={Q.shape} nlist={nlist}", flush=True)

    warm_up_jit(X.shape[1])
    gt, _ = topk_sq_l2(X, Q, k)

    results = {"dataset": "glove-25-angular (ann-benchmarks)", "n_base": len(X),
               "n_queries": len(Q), "k": k, "M": args.M, "ef_construction": args.efc,
               "nlist": nlist, "pq_m": args.pq_m, "repeats": args.repeats,
               "machine": {"platform": platform.platform(), "processor": platform.processor(),
                           "python": platform.python_version()},
               "threads": 1, "sweeps": [], "build": {}}

    print("exact ...", flush=True)
    results["sweeps"].append(bench_flat(X, Q, gt, k, args.repeats))

    print("vecdb hnsw ...", flush=True)
    idx, rows, info = bench_hnsw(X, Q, gt, k, args.repeats, args.M, args.efc)
    results["sweeps"] += rows
    results["build"]["vecdb_hnsw"] = info
    print(info, flush=True)

    print("vecdb ivfpq ...", flush=True)
    rows, info = bench_ivfpq(X, Q, gt, k, args.repeats, nlist, args.pq_m)
    results["sweeps"] += rows
    results["build"]["vecdb_ivfpq"] = info

    if faiss is not None:
        print("faiss ...", flush=True)
        rows, info = bench_faiss_hnsw(X, Q, gt, k, args.repeats, args.M, args.efc)
        results["sweeps"] += rows
        results["build"]["faiss_hnsw"] = info
        rows, info = bench_faiss_ivfpq(X, Q, gt, k, args.repeats, nlist, args.pq_m)
        results["sweeps"] += rows
        results["build"]["faiss_ivfpq"] = info

    with open(os.path.join(out_dir, "sweeps.json"), "w") as f:
        json.dump(results, f, indent=2)

    if not args.skip_extras:
        print("filtered ...", flush=True)
        filt = bench_filtered(idx, X, Q, k, args.repeats)
        with open(os.path.join(out_dir, "filtered.json"), "w") as f:
            json.dump({"ef": 64, "k": k, "n_base": len(X), "rows": filt}, f, indent=2)
        print("deletes ...", flush=True)
        dels = bench_deletes(X, Q, k, args.M, args.efc)
        with open(os.path.join(out_dir, "deletes.json"), "w") as f:
            json.dump({"ef": 64, "k": k, "rows": dels}, f, indent=2)
        print("durability ...", flush=True)
        tmp = tempfile.mkdtemp(prefix="vecdb-bench-")
        try:
            dur = bench_durability(X, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with open(os.path.join(out_dir, "durability.json"), "w") as f:
            json.dump(dur, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
