"""Download and load the GloVe-25 ANN benchmark set.

Source: ann-benchmarks (Aumuller, Bernhardsson, Faithfull), file
``glove-25-angular.hdf5`` = 1,183,514 GloVe Twitter word vectors (25-d) plus 10,000
held-out query vectors. GloVe vectors are released by Stanford NLP under the Public
Domain Dedication and License (PDDL 1.0): https://nlp.stanford.edu/projects/glove/
The file is downloaded at runtime and never committed.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np

GLOVE25_URL = "http://ann-benchmarks.com/glove-25-angular.hdf5"
GLOVE25_SHA256 = "51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: str = "data", url: str = GLOVE25_URL) -> str:
    """Fetch the dataset into ``data_dir`` if not already present; return its path."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, os.path.basename(url))
    if not os.path.exists(path):
        tmp = path + ".part"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    if GLOVE25_SHA256 and sha256(path) != GLOVE25_SHA256:
        raise IOError(f"checksum mismatch for {path}; delete it and retry")
    return path


def normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def load_glove25(n_base: int, n_queries: int, data_dir: str = "data", seed: int = 0):
    """Return (base, queries) as unit-normalised float32 arrays.

    ``n_base`` rows are sampled without replacement from the 1.18M training vectors
    (or all of them if ``n_base`` is larger); queries come from the official test split.
    Zero vectors (none in this file, but guarded) are dropped: cosine is undefined for them.
    """
    import h5py

    path = download(data_dir)
    with h5py.File(path, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
    train = train[np.linalg.norm(train, axis=1) > 0]
    test = test[np.linalg.norm(test, axis=1) > 0]
    rng = np.random.default_rng(seed)
    if n_base < train.shape[0]:
        train = train[np.sort(rng.choice(train.shape[0], n_base, replace=False))]
    test = test[:n_queries]
    return normalize(train), normalize(test)
