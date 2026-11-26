# Walkthrough

This document explains how vecdb works, file by file, and why it is built the way it
is. Read it top to bottom once, then use the interview section to test yourself.

## 1. The problem

Embedding models turn text, images and code into vectors such that "similar"
things land close together. Retrieval-augmented generation (RAG), semantic search,
recommendation and deduplication all reduce to one query: *given a vector q, find
the k stored vectors closest to it*.

Exact search compares q with every stored vector: O(n·d) per query. At a million
768-dimensional vectors that is ~768 million multiply-adds per query, too slow for
interactive traffic and wasteful at high QPS. Approximate nearest neighbour (ANN)
indexes trade a small, measurable loss in accuracy (recall) for orders of magnitude
less work. Vector databases (pgvector, Qdrant, Weaviate, Milvus, FAISS-based
services) wrap such an index with the things an application needs: ids, metadata
("payload") filters, deletes, persistence and an API.

vecdb implements all of those pieces from scratch so each one can be explained and
measured:

| Piece | File | Idea |
|---|---|---|
| Graph index | `vecdb/hnsw.py` | HNSW: multi-layer proximity graph, greedy beam search |
| Compressed index | `vecdb/ivfpq.py` | IVF-PQ: k-means cells + product-quantised codes |
| Exact baseline | `vecdb/flat.py` | one matrix multiply + partial sort |
| Durability | `vecdb/wal.py` | CRC-framed write-ahead log, torn-write recovery |
| Collection | `vecdb/collection.py` | ids, payload index, filter planner, snapshots |
| API | `vecdb/server.py` | FastAPI HTTP endpoints |
| Evaluation | `benchmarks/` | recall/QPS sweeps against exact search and FAISS |

## 2. Architecture and data flow

```
 HTTP (FastAPI)          vecdb/server.py
      |
      v
 Collection              vecdb/collection.py
   upsert/delete ---> WAL append (flush, optional F_FULLFSYNC) ---> apply in memory
   search  -------> filter mask (payload postings) -> planner
                         |                     |
                 hnsw / hnsw_filtered     exact / exact_filtered
                         |                     |
                     HNSWIndex             topk_sq_l2
   snapshot ------> index .npz + meta .json -> CURRENT (atomic rename) -> truncate WAL
   open    -------> load snapshot named by CURRENT -> replay WAL records with lsn > snapshot
```

**Write path.** `Collection.upsert` validates and normalises vectors, assigns each
record a log sequence number (LSN), appends all records to the WAL in one write,
and only then inserts them into the HNSW graph. If the process dies after the WAL
append, replay redoes the insert; if it dies before, the client never got an
acknowledgement. This "log first, then apply" rule is the core of every database
recovery scheme (ARIES, Postgres WAL, RocksDB).

**Read path.** A filter like `{"color": "red", "n": {"$in": [0, 3]}}` is turned into a
boolean mask using an inverted index (field -> value -> list of internal ids). The
planner counts matches and chooses a strategy (section 4.4), then the hits are
mapped back to external ids and payloads.

## 3. File-by-file tour

### `vecdb/hnsw.py`

*Layers.* Every inserted node draws a level `L = floor(-ln(U) * mL)` with
`mL = 1/ln(M)`. So all nodes are on layer 0, about 1/M of them on layer 1, 1/M² on
layer 2, and so on. Upper layers are sparse "express lanes"; layer 0 is the dense
local graph.

*Search (`_search_batch`).* Start at the global entry point on the top layer and walk
greedily (move to any neighbour that is closer, repeat) down to layer 1. On layer 0
run a beam search (`_search_layer`) of width `ef`: a min-heap of candidates to expand
and a max-heap of the best `ef` results found. Stop when the closest unexpanded
candidate is farther than the worst result. Return the top k. Larger `ef` explores
more nodes, giving higher recall at lower QPS; this is the main knob in the
benchmark sweep.

*Insert (`_insert_range`).* Search for the new node as if it were a query (beam width
`ef_construction`), then connect it to up to M neighbours chosen by the *selection
heuristic* (`_select_heuristic`): walk candidates from nearest to farthest and keep
one only if it is closer to the new node than to every neighbour already kept. This
avoids spending all M edges inside one tight cluster and keeps long-range links that
connect clusters. Each chosen neighbour gets a back-edge; if its list overflows
(M on upper layers, 2M on layer 0) it is re-pruned with the same heuristic
(`_add_link`).

*Memory layout.* Adjacency lists are fixed-width int32 rows (`links0` for layer 0,
`upper` for higher layers) with a separate degree array. Fixed-width rows mean no
Python objects per node, contiguous memory, and Numba can compile everything to
machine code. A node of level L owns L consecutive rows in `upper`, located via
`upper_off`. Since ~94% of nodes have level 0 (for M=16), this is much smaller than a
dense `[n, max_level, M]` tensor.

*Visited set.* Instead of allocating or clearing a visited array per query, each
search increments a generation counter and marks `visited[i] = tag`. Clearing is
then free. hnswlib uses the same trick.

*Deletes.* `mark_deleted` sets a tombstone. Deleted nodes still route traffic (their
edges stay) but never appear in results. This keeps the graph connected without the
complexity of repairing edges; the cost is that tombstones occupy memory until a
rebuild (not implemented, see limitations).

*Filters.* With a mask, non-matching nodes are traversed but not admitted to the
result heap, and the early-exit test only fires once `ef` matching results exist.
That is what makes very selective filters slow (the search wanders) and is why a
planner is needed.

### `vecdb/ivfpq.py`

*Coarse quantiser.* k-means with `nlist` centroids (`kmeans`, k-means++ seeding,
empty-cluster re-seeding) splits space into cells. Each vector lives in the inverted
list of its nearest cell, stored in CSR form (`offsets`, `list_ids`, `list_codes`)
so a probe scans contiguous memory.

*Product quantisation.* The residual `x - centroid` is split into `m` sub-vectors;
each is replaced by the index of its nearest of 256 sub-centroids. With m=5 a
25-d float vector (100 bytes) becomes 5 bytes: 20x compression.

*Asymmetric distance computation.* The query stays exact; only database vectors are
quantised. Distance to a code is a sum of `m` table lookups. The tables are built
cheaply using the decomposition

```
||q - yc - yr||^2 = ||q - yc||^2 + (||yr||^2 + 2<yc, yr>) - 2<q, yr>
```

where the middle term depends only on (cell, code) and is precomputed at training
time (`term2`), and the last term is computed once per query (`term3`). Building a
cell's table then costs m·256 additions instead of m·256·dsub multiply-adds. In
development this made the scan several times faster; the earlier slower version's
numbers were not kept, so only the final numbers appear in `reports/`.

*Re-ranking.* PQ distances are approximate, so recall plateaus (~0.55 here for 5-byte
codes on 25-d data; FAISS hits the same ceiling, which is a useful correctness
check). Keeping the raw vectors and re-scoring the best 100 PQ candidates exactly
recovers most of the loss, at the cost of storing floats again.

### `vecdb/flat.py`

Exact search using `||x||² - 2·x·q + ||q||²` so the heavy part is one BLAS matrix
multiply, then `argpartition` (O(n)) instead of a full sort. Used for ground truth,
as the baseline, and by the planner for selective filters.

### `vecdb/wal.py`

Each record is framed as `length | crc32 | body`. On startup `replay` reads frames
until one is incomplete or fails its checksum. That can only be a torn write from a
crash mid-append, so the file is truncated to the last good frame. Durability
options:

* `fsync=False`: data reaches the OS page cache. Survives a process crash, not a
  power loss.
* `fsync=True`: `F_FULLFSYNC` on macOS (plain `fsync` there does not flush the drive's
  write cache, so an early version that used plain `fsync` showed almost no cost for
  syncing, a sign it was not really durable), `fsync` on Linux. Batching many records per sync (group commit) is what makes this affordable.

### `vecdb/collection.py`

Maps external string ids to internal row numbers. Upserting an existing id
tombstones the old row and appends a new one (append-only, so the graph never needs
in-place vector updates). Snapshots write the index and metadata under a name that
includes the LSN, then atomically replace a tiny `CURRENT` file; that rename is the
commit point. Only after it succeeds is the WAL truncated. If the process dies
between the two, the next start sees WAL records whose LSN is already in the
snapshot and skips them (tested in
`test_replay_skips_records_already_in_snapshot`).

### `vecdb/server.py`, `vecdb/cli.py`

Thin FastAPI layer: Pydantic validates shapes, `CollectionError` becomes HTTP 400,
unknown collections 404, duplicates 409. Collections found on disk are reopened (and
recovered) at startup. Each collection serialises operations with a lock, because the
shared `visited` array in the index is not thread-safe.

### `benchmarks/run_benchmarks.py`, `benchmarks/plot.py`

Downloads GloVe-25 (checksum pinned), computes exact ground truth, then sweeps `ef`
(HNSW) and `nprobe` (IVF-PQ) for vecdb and the same configurations in FAISS. Also
runs the filtered-search, delete, and durability experiments. Everything is pinned
to one thread so the comparison is about algorithms, not cores.

### `tests/`

42 tests (unit + integration): graph invariants (degree bounds, no self-loops, layer
slot accounting, reachability), recall floors, filter correctness including fewer
matches than k, tombstones, save/load equivalence including RNG state, ADC distance
equal to decoded-vector distance, WAL torn-tail and bit-flip handling, crash
recovery, snapshot + replay, API errors and restart.

## 4. Design decisions and trade-offs

### 4.1 HNSW vs other ANN families

* **IVF (inverted file)**: simple, small, but recall depends on probing enough cells;
  bad at very high recall.
* **Graph (HNSW, Vamana/DiskANN, NSG)**: best recall/QPS trade-off in memory, but uses
  more memory (vectors + ~2M edges per node) and inserts are expensive.
* **Quantisation (PQ, ScaNN, RaBitQ)**: compresses vectors; usually combined with IVF or
  graphs and a re-ranking step.
* **LSH**: theoretical guarantees, but needs many tables to compete; rarely used now.

vecdb ships HNSW (the default in Qdrant, Weaviate, pgvector, Milvus) and IVF-PQ (the
classic memory-saving option) so the trade-off can be measured, not just claimed.

### 4.2 Numba instead of C++/Rust

Numba compiles the Python kernels to LLVM machine code, which gets close to
hand-written C++ without a build toolchain. Constraints: kernels must use NumPy
arrays and simple types, so the graph is stored as fixed-width int32 arrays rather
than Python lists. The measured gap to FAISS's C++ HNSW is in the README.

### 4.3 Tombstones instead of graph repair

Removing a node and re-linking its neighbours (as hnswlib's `replace_deleted` or
FreshDiskANN do) is complex and can degrade the graph. Tombstones are simple and
correct; the price is memory and wasted traversal. Production systems compact by
rebuilding segments in the background.

### 4.4 Filter planner

Two plans for filtered queries: walk the graph with a mask, or brute-force the
matches. Graph cost rises as selectivity falls (the beam must visit more
non-matching nodes), while brute-force cost is proportional to the number of
matches. The measured crossover moved with collection size (10-20% selectivity at
100k points, 5-10% at 1.18M), so a single fixed threshold is wrong somewhere.

`choose_filtered_plan` therefore estimates both costs:

* graph: `20 * ef * selectivity^-0.8` distance evaluations (the 20 and 0.8 are fitted
  to the measured distance counts in `filtered.json`),
* brute force: `matching` evaluations, each ~5.7x cheaper because they are sequential
  BLAS work rather than random memory access plus heap updates,

and picks the cheaper one. `benchmarks/eval_planner.py` grades it against the measured
QPS of both plans. Be honest about this in an interview: the constants are fitted on
the same rows they are graded on, and a plain "selectivity < 10%" rule scores almost
as well. Qdrant uses a similar idea (its "full scan threshold"); ACORN (SIGMOD 2024)
changes the graph itself so filtered traversal degrades less.

### 4.5 WAL + snapshot instead of a memory-mapped index

A WAL gives cheap durable writes; snapshots bound recovery time (replay rebuilds the
graph, which is slow: see the durability numbers). Memory-mapped segments
(Lance, Milvus) would avoid loading everything into RAM but add page-cache tuning and
format versioning; out of scope here.

## 5. How the evaluation works

* **Recall@10**: of the true 10 nearest neighbours (from exact search), how many did
  the index return. 0.95 means on average 9.5 of 10.
* **QPS**: queries per second answering 1000 queries in one batch call, single thread,
  best of 3 repeats.
* **Distance computations per query**: hardware-independent measure of work for HNSW.
* **Reading the recall/QPS chart**: up and to the right is better. A curve that is
  higher at the same recall is strictly better. Compare methods at a fixed recall
  target (e.g. 0.95), never at a fixed parameter value.
* **Caveats**: one dataset (25-d, cosine); random-uniform filters (real filters often
  correlate with vector position, which changes the picture); wall-clock numbers
  depend on the machine.

## 6. Interview questions

1. **Why is HNSW search roughly logarithmic?** Upper layers have exponentially fewer
   nodes, so greedy descent covers large distances in few hops (like a skip list);
   layer 0 then only needs a local beam search near the target.
2. **What does `ef` control, and how is it different from `ef_construction`?** `ef` is
   the beam width at query time (recall vs latency). `ef_construction` is the beam
   used to find neighbours when inserting; higher gives a better graph but slower
   builds. They are independent knobs.
3. **Why the neighbour selection heuristic rather than the M nearest?** Pure k-NN edges
   cluster: all M edges may point into the same dense region and disconnect clusters.
   The heuristic keeps a candidate only if no kept neighbour is closer to it, which
   diversifies directions and preserves navigability.
4. **How do you delete from HNSW?** Tombstone and skip in results while still routing
   through the node; periodically rebuild or repair. Hard delete requires re-linking
   neighbours.
5. **Why does filtered HNSW get slow for selective filters, and what do you do?** The
   beam must find `ef` matching nodes, so it visits many non-matching ones. Fall back
   to brute force over the matches when they are few (planner), or partition the index
   by the filter field.
6. **How does PQ compress a vector?** Split into m sub-vectors, replace each by the id
   of its nearest of 256 sub-centroids (1 byte). Memory goes from 4d bytes to m bytes.
7. **What is asymmetric distance computation?** Query stays exact; distances to
   centroids are tabulated once per query/cell, then each database code costs m
   lookups and adds.
8. **Why does IVF-PQ recall plateau, and how do you fix it?** Quantisation error limits
   how well distances can be ranked. More bytes per code (larger m), OPQ rotation,
   or re-ranking candidates with exact vectors.
9. **How do you choose `nlist`?** Rule of thumb ~4·sqrt(n): enough cells that each probe
   is cheap, few enough that each cell has a useful number of points.
10. **How do you know your ANN index is correct?** Compare to exact search (recall),
    check structural invariants, and compare against a reference implementation
    (FAISS) at the same parameters: matching recall means the algorithm is right; QPS
    differences are engineering.
11. **What does the WAL protect against, and what does fsync add?** Without the WAL a
    crash loses all un-snapshotted writes. Flushing to the page cache survives process
    crashes; fsync (F_FULLFSYNC on macOS) survives power loss.
12. **What is a torn write and how do you detect it?** A crash mid-append leaves a
    partial record. Length + CRC framing detects it; truncate to the last valid frame.
13. **Why does the snapshot store an LSN?** So replay after a crash between "snapshot
    committed" and "WAL truncated" skips records already included, making recovery
    idempotent.
14. **Why measure single-threaded?** Fair comparison of algorithms; multi-threading
    multiplies QPS roughly by core count for read-only workloads and hides per-query
    efficiency.
15. **How would you scale to 100M vectors?** Shard by id across nodes and fan out
    queries (merge top-k), use PQ or disk-based graphs (DiskANN) to cut RAM, build
    segments in the background, and replicate shards for availability.

## 7. Further study

* Malkov & Yashunin, *HNSW* (2016): https://arxiv.org/abs/1603.09320
* Jegou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search* (TPAMI 2011):
  https://ieeexplore.ieee.org/document/5432202
* Subramanya et al., *DiskANN* (NeurIPS 2019):
  https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html
* Aumuller, Bernhardsson, Faithfull, *ANN-Benchmarks* (2018): https://arxiv.org/abs/1807.05614
* Johnson, Douze, Jegou, *Billion-scale similarity search with GPUs* (FAISS, 2017):
  https://arxiv.org/abs/1702.08734
* Gao & Long, *RaBitQ* (SIGMOD 2024): https://arxiv.org/abs/2405.12497
* Patel et al., *ACORN: filtered vector search* (SIGMOD 2024): https://arxiv.org/abs/2403.04871
* Mohan et al., *ARIES* (write-ahead logging, 1992): https://dl.acm.org/doi/10.1145/128765.128770
* hnswlib source (reference implementation): https://github.com/nmslib/hnswlib
* Numba documentation: https://numba.readthedocs.io/
