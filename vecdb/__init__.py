"""vecdb: a small vector database built from scratch.

Public entry points:
    HNSWIndex   graph index (vecdb.hnsw)
    IVFPQIndex  compressed inverted-file index (vecdb.ivfpq)
    Collection  durable collection with payload filters (vecdb.collection)
"""
from .collection import Collection, CollectionError
from .hnsw import HNSWIndex
from .ivfpq import IVFPQIndex

__all__ = ["Collection", "CollectionError", "HNSWIndex", "IVFPQIndex"]
__version__ = "0.1.0"
