"""FastAPI service. Request = node features + edges + target node ids (ids = row numbers of `features`)."""
import os
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from package import ModelLoadError, Predictor
from registry import Registry


class PredictRequest(BaseModel):
    features: list[list[float]]
    edges: list[list[int]] = []
    target_ids: list[int]


class RequestError(ValueError):
    pass


def validate(req, n_features):
    """Defined behaviour: every problem -> RequestError -> HTTP 422 with a message. Nothing is silently fixed."""
    n = len(req.features)
    if n == 0 or any(len(r) != n_features for r in req.features):
        raise RequestError(f"features must be an N x {n_features} matrix")
    x = torch.tensor(req.features, dtype=torch.float32)
    if not torch.isfinite(x).all() or (x < 0).any():
        raise RequestError("features must be finite and non-negative")
    if any(len(e) != 2 or not (0 <= e[0] < n and 0 <= e[1] < n) for e in req.edges):
        raise RequestError(f"invalid edge endpoint: edges are [u, v] pairs with ids in [0, {n - 1}]")
    ids = req.target_ids
    if not ids:
        raise RequestError("target_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise RequestError("duplicate target_ids")
    if any(not 0 <= i < n for i in ids):
        raise RequestError(f"unknown target id: valid ids are 0..{n - 1}")
    e = torch.tensor(req.edges, dtype=torch.long).T if req.edges else torch.empty(2, 0, dtype=torch.long)
    e = e[:, e[0] != e[1]]                                   # self-loops are added by the model itself
    e = torch.unique(torch.cat([e, e.flip(0)], 1), dim=1)    # edges are undirected; duplicates removed
    return x, e, torch.tensor(ids)


def create_app(registry_root=None):
    root = registry_root or os.environ.get("REGISTRY_ROOT", "registry")
    app, state = FastAPI(title="Cora GCN service"), {"predictor": None, "error": None}

    def load():
        state["predictor"], state["error"] = None, None
        try:
            reg = Registry(root)
            if reg.approved is None:
                raise ModelLoadError("no approved model in registry")
            state["predictor"] = Predictor(reg.path(reg.approved))
        except Exception as e:
            state["error"] = str(e)

    def health():
        if state["predictor"] is None:
            return JSONResponse({"status": "unavailable", "error": state["error"]}, 503)
        return {"status": "ok", "model_version": state["predictor"].version}

    load()
    app.get("/health")(health)

    @app.post("/reload")                                      # call after promote / rollback
    def reload():
        load()
        return health()

    @app.post("/predict")
    def predict(req: PredictRequest):
        p = state["predictor"]
        if p is None:
            return JSONResponse({"error": f"model unavailable: {state['error']}"}, 503)
        try:
            x, e, ids = validate(req, p.n_features)
        except RequestError as ex:
            return JSONResponse({"error": str(ex)}, 422)
        return {"model_version": p.version, "predictions": p.predict(x, e, ids),
                "context": {"n_nodes": x.size(0), "n_edges": e.size(1) // 2}}

    return app


app = create_app()