"""HTTP API (FastAPI) exposing collections stored under a data directory."""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .collection import Collection, CollectionError

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CreateCollection(BaseModel):
    name: str
    dim: int = Field(gt=0, le=4096)
    metric: Literal["cosine", "l2"] = "cosine"
    M: int = Field(16, ge=2, le=128)
    ef_construction: int = Field(200, ge=8, le=2000)


class Point(BaseModel):
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any] = Field(default_factory=dict)


class UpsertPoints(BaseModel):
    points: List[Point] = Field(min_length=1, max_length=10_000)


class DeletePoints(BaseModel):
    ids: List[Union[str, int]] = Field(min_length=1)


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = Field(10, ge=1, le=1000)
    ef: int = Field(64, ge=1, le=5000)
    filter: Optional[Dict[str, Any]] = None
    exact: bool = False


def create_app(data_dir: Optional[str] = None) -> FastAPI:
    """Build the app. Collections found in ``data_dir`` are reopened (and recovered)."""
    data_dir = data_dir or os.environ.get("VECDB_DATA", "./vecdb-data")
    os.makedirs(data_dir, exist_ok=True)
    app = FastAPI(title="vecdb", version="0.1.0")
    cols: Dict[str, Collection] = {}
    registry_lock = threading.Lock()

    for name in sorted(os.listdir(data_dir)):
        if os.path.exists(os.path.join(data_dir, name, "config.json")):
            cols[name] = Collection(os.path.join(data_dir, name))

    def get(name: str) -> Collection:
        col = cols.get(name)
        if col is None:
            raise HTTPException(404, f"collection {name!r} not found")
        return col

    @app.get("/health")
    def health():
        return {"status": "ok", "collections": len(cols)}

    @app.post("/collections", status_code=201)
    def create(req: CreateCollection):
        if not NAME_RE.match(req.name):
            raise HTTPException(400, "name must match [A-Za-z0-9_-]{1,64}")
        with registry_lock:
            if req.name in cols:
                raise HTTPException(409, f"collection {req.name!r} exists")
            cols[req.name] = Collection(os.path.join(data_dir, req.name), dim=req.dim,
                                        metric=req.metric, M=req.M,
                                        ef_construction=req.ef_construction)
        return {"name": req.name, **cols[req.name].config}

    @app.get("/collections")
    def list_collections():
        return {"collections": [{"name": n, **c.config, "points": len(c)} for n, c in cols.items()]}

    @app.get("/collections/{name}")
    def info(name: str):
        return {"name": name, **get(name).stats()}

    @app.put("/collections/{name}/points")
    def upsert(name: str, req: UpsertPoints):
        col = get(name)
        try:
            n = col.upsert([p.id for p in req.points], [p.vector for p in req.points],
                           [p.payload for p in req.points])
        except CollectionError as e:
            raise HTTPException(400, str(e))
        return {"upserted": n, "lsn": col.lsn}

    @app.get("/collections/{name}/points/{point_id}")
    def get_point(name: str, point_id: str):
        p = get(name).get(point_id)
        if p is None:
            raise HTTPException(404, "point not found")
        return p

    @app.post("/collections/{name}/points/delete")
    def delete(name: str, req: DeletePoints):
        return {"deleted": get(name).delete(req.ids)}

    @app.post("/collections/{name}/search")
    def search(name: str, req: SearchRequest):
        try:
            return get(name).search(req.vector, k=req.k, ef=req.ef, filter=req.filter,
                                    exact=req.exact)
        except CollectionError as e:
            raise HTTPException(400, str(e))

    @app.post("/collections/{name}/snapshot")
    def snapshot(name: str):
        return {"snapshot_lsn": get(name).snapshot()}

    @app.on_event("shutdown")
    def _close():
        for c in cols.values():
            c.close()

    return app
