"""Run:  pytest tests -q     (tiny synthetic graph, no downloads)"""
import json, os, sys
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fastapi.testclient import TestClient
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

from client import build_request, context_graph
from data import drop_edges, make_snapshot, training_graph
from model import GCN, normalize_rows, train_model
from monitor import calibrate, detect, linear_cka, permute_hidden, snapshot_metrics
from package import ModelLoadError, Predictor, save_package, val_accuracy
from registry import Registry, gate
from serve import PredictRequest, create_app, validate

D, C = 12, 3
CFG = dict(hidden=8, dropout=0.5, lr=0.01, wd=5e-4, epochs=15, n_labels=None, n_classes=C)
LABELS = ["a", "b", "c"]


@pytest.fixture
def graph():
    g = torch.Generator().manual_seed(0)
    n = 60
    x = (torch.rand(n, D, generator=g) < 0.3).float(); x[:, 0] = 1                     # no all-zero rows
    e = to_undirected(remove_self_loops(torch.randint(0, n, (2, 150), generator=g))[0])
    m = lambda s: torch.zeros(n, dtype=torch.bool).index_fill_(0, torch.arange(s, s + 20), True)
    return Data(x=x, y=torch.randint(0, C, (n,), generator=g), edge_index=e,
                train_mask=m(0), val_mask=m(20), test_mask=m(40))


@pytest.fixture
def trained(graph, tmp_path):
    model, val, _ = train_model(graph, CFG, seed=0)
    return model, save_package(tmp_path / "pkg", model, CFG, D, LABELS, "ds123")


@pytest.fixture
def served(tmp_path, trained):
    reg = Registry(tmp_path / "reg")
    v = reg.register(trained[1], {"val_acc": 0.7}); reg.promote(v)
    return TestClient(create_app(str(tmp_path / "reg"))), reg, v


# ---------- packaging ----------
def test_reload_reproduces_predictions(graph, trained):
    model, pkg = trained
    ref = model(normalize_rows(graph.x), graph.edge_index)
    p1, p2 = Predictor(pkg), Predictor(pkg)
    assert torch.equal(p1.logits(graph.x, graph.edge_index), ref)
    assert torch.equal(p1.logits(graph.x, graph.edge_index), p2.logits(graph.x, graph.edge_index))


def test_package_contains_schema_labels_preprocessing(trained):
    meta = json.loads((trained[1] / "meta.json").read_text())
    assert meta["feature_schema"]["n_features"] == D and meta["labels"] == LABELS
    assert meta["architecture"]["d_out"] == C and "row_normalize" in meta["preprocessing"] and meta["dataset_id"] == "ds123"


def test_preprocessing_is_applied_inside_package(graph, trained):
    p = Predictor(trained[1])
    assert torch.allclose(p.logits(graph.x, graph.edge_index), p.logits(5 * graph.x, graph.edge_index), atol=1e-5)


def test_corrupt_package_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        Predictor(tmp_path / "nothing_here")


# ---------- gate ----------
@pytest.mark.parametrize("cand,appr,has,kw,expected", [
    (None, 0.8, True, {}, False),                      # missing candidate metric
    (float("nan"), 0.8, True, {}, False),
    (0.30, None, False, {}, False),                    # first model below floor
    (0.70, None, False, {}, True),                     # first model OK
    (0.80, 0.80, True, {}, False),                     # tie -> blocked
    (0.802, 0.80, True, {}, False),                    # gain below threshold
    (0.79, 0.80, True, {}, False),                     # regression
    (0.81, 0.80, True, {}, True),                      # improvement
    (0.81, None, True, {}, False),                     # approved metric missing
    (0.80, 0.80, True, {"min_delta": -0.01}, True),    # CI mode: tie allowed
    (0.77, 0.80, True, {"min_delta": -0.01}, False),   # CI mode: real regression blocked
])
def test_gate(cand, appr, has, kw, expected):
    assert gate(cand, appr, has, **kw).promote is expected


# ---------- registry / rollback ----------
def test_register_promote_block_rollback(tmp_path, trained):
    reg = Registry(tmp_path / "r")
    metrics = {}
    ev = lambda path: metrics[path.name]
    v1, v2, v3 = (reg.register(trained[1], {}) for _ in range(3))
    assert (v1, v2, v3) == ("v1", "v2", "v3")
    metrics.update({"v1": 0.60, "v2": 0.80, "v3": 0.50})
    assert reg.try_promote(v1, ev).promote and reg.approved == "v1"      # first model
    assert reg.try_promote(v2, ev).promote and reg.approved == "v2"      # improvement
    assert not reg.try_promote(v3, ev).promote and reg.approved == "v2"  # regression blocked, approved unchanged
    assert reg.rollback() == "v1" and Registry(tmp_path / "r").approved == "v1"   # persisted
    with pytest.raises(RuntimeError):
        reg.rollback()                                                    # nothing older to restore


def test_failed_evaluation_counts_as_missing_metric(tmp_path, trained):
    reg = Registry(tmp_path / "r"); v = reg.register(trained[1], {})
    def boom(path): raise ValueError("cannot evaluate")
    assert not reg.try_promote(v, boom).promote and reg.approved is None


def test_registered_version_is_stamped(tmp_path, trained):
    reg = Registry(tmp_path / "r"); v = reg.register(trained[1], {})
    assert Predictor(reg.path(v)).version == v


def test_val_accuracy_matches_manual(graph, trained):
    p = Predictor(trained[1])
    manual = (p.logits(graph.x, graph.edge_index).argmax(1)[graph.val_mask] == graph.y[graph.val_mask]).float().mean().item()
    assert val_accuracy(trained[1], graph) == manual


# ---------- unseen nodes ----------
def test_training_graph_excludes_unseen_nodes_and_edges(graph):
    unseen = torch.tensor([3, 7, 11])
    g2 = training_graph(graph, unseen)
    touching = torch.isin(graph.edge_index, unseen).any(0)
    assert g2.num_nodes == graph.num_nodes - 3
    assert g2.edge_index.size(1) == int((~touching).sum())
    assert int(g2.edge_index.max()) < g2.num_nodes


def test_three_hop_context_reproduces_full_graph_predictions(graph, trained):
    p = Predictor(trained[1])
    ids = torch.tensor([3, 7, 11, 20])
    full = p.logits(graph.x, graph.edge_index)[ids]
    xs, ei, local, _ = context_graph(graph.x, graph.edge_index, ids, hops=3)
    assert torch.allclose(p.logits(xs, ei)[local], full, atol=1e-5)


# ---------- API ----------
def test_api_ok_and_matches_offline(graph, trained, served):
    client, reg, v = served
    ids = [3, 7, 11]
    payload, subset = build_request(graph.x, graph.edge_index, ids)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200 and r.json()["model_version"] == v
    offline = Predictor(reg.path(v))
    x, e, loc = validate(PredictRequest(**payload), D)
    same_input = offline.predict(x, e, loc)                                   # identical input/context/model
    full = offline.predict(graph.x, graph.edge_index, torch.tensor(ids))      # whole graph
    for api, a, b in zip(r.json()["predictions"], same_input, full):
        assert api["label"] == a["label"] == b["label"]
        assert api["confidence"] == pytest.approx(a["confidence"], abs=1e-9)
        assert api["confidence"] == pytest.approx(b["confidence"], abs=1e-5)


def test_health_reports_version(served):
    client, _, v = served
    assert client.get("/health").json() == {"status": "ok", "model_version": v}


def base_payload(**over):
    p = {"features": [[1.0] * D for _ in range(3)], "edges": [[0, 1]], "target_ids": [0]}
    p.update(over)
    return p


@pytest.mark.parametrize("bad", [
    dict(features=[[1.0] * (D - 1)] * 3),        # wrong feature dimension
    dict(target_ids=[0, 0]),                     # duplicate ids
    dict(target_ids=[9]),                        # unknown id
    dict(target_ids=[-1]),                       # negative id
    dict(target_ids=[]),                         # empty
    dict(edges=[[0, 9]]),                        # invalid endpoint
    dict(edges=[[0, -1]]),                       # negative endpoint
    dict(edges=[[0, 1, 2]]),                     # malformed edge
    dict(features=[[-1.0] + [1.0] * (D - 1)] * 3),   # negative feature
])
def test_api_rejects_bad_requests(served, bad):
    r = served[0].post("/predict", json=base_payload(**bad))
    assert r.status_code == 422


def test_api_isolated_node_is_served_and_flagged(served):
    r = served[0].post("/predict", json=base_payload(target_ids=[2]))
    assert r.status_code == 200 and r.json()["predictions"][0]["isolated"] is True


def test_api_without_model_is_503(tmp_path):
    c = TestClient(create_app(str(tmp_path / "does_not_exist")))
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json=base_payload()).status_code == 503


def test_api_corrupt_weights_is_503(tmp_path, trained):
    reg = Registry(tmp_path / "reg"); v = reg.register(trained[1], {}); reg.promote(v)
    (reg.path(v) / "weights.pt").write_bytes(b"garbage")
    assert TestClient(create_app(str(tmp_path / "reg"))).post("/predict", json=base_payload()).status_code == 503


def test_reload_after_rollback_serves_previous_version(tmp_path, trained):
    reg = Registry(tmp_path / "reg")
    for _ in range(2):
        reg.promote(reg.register(trained[1], {}))
    c = TestClient(create_app(str(tmp_path / "reg")))
    assert c.get("/health").json()["model_version"] == "v2"
    reg.rollback()
    assert c.post("/reload").json()["model_version"] == "v1"


# ---------- drift ----------
def test_drift_normal_vs_changed(graph, trained):
    p = Predictor(trained[1]); ref = (graph.x, graph.edge_index)
    normal = [snapshot_metrics(p, ref, make_snapshot(graph, 0.05, 0.0, seed=s)) for s in range(10)]
    thr = calibrate(normal)
    assert not any(detect(normal[0], thr).values())
    same = snapshot_metrics(p, ref, ref)
    assert same["degree_change"] == 0.0 and not any(detect(same, thr).values())
    drift = snapshot_metrics(p, ref, make_snapshot(graph, 0.8, 0.5, seed=1))
    assert detect(drift, thr)["degree_change"] and detect(drift, thr)["embedding_shift"]


def test_drop_edges_keeps_symmetry(graph):
    e = drop_edges(graph.edge_index, 0.5)
    s = {tuple(c) for c in e.T.tolist()}
    assert {(b, a) for a, b in s} == s and s <= {tuple(c) for c in graph.edge_index.T.tolist()}


def test_raw_embeddings_not_comparable_but_cka_is(graph):
    torch.manual_seed(0)
    m = GCN(D, 8, C).eval()
    x = normalize_rows(graph.x)
    m2 = permute_hidden(m, torch.randperm(8, generator=torch.Generator().manual_seed(1)))
    (o1, z1), (o2, z2) = m(x, graph.edge_index, return_emb=True), m2(x, graph.edge_index, return_emb=True)
    assert torch.allclose(o1, o2, atol=1e-5)                    # same predictions
    assert not torch.allclose(z1, z2)                           # different raw coordinates
    assert linear_cka(z1, z2) == pytest.approx(1.0, abs=1e-4)   # same geometry


# ---------- tracking ----------
def test_mlflow_logs_three_experiments(graph, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from tracking import run_experiment
    monkeypatch.chdir(tmp_path)
    runs = [run_experiment(f"exp{i}", graph, dict(CFG, epochs=3), seed=i, labels=LABELS, out_dir=tmp_path / "art") for i in range(3)]
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    df = mlflow.search_runs(experiment_names=["cora-gcn"])
    assert len(df) == 3 and {"params.seed", "tags.git_commit", "tags.dataset_id", "metrics.best_val_acc"} <= set(df.columns)
    assert any(a.path == "model" for a in mlflow.MlflowClient().list_artifacts(runs[0]["run_id"]))