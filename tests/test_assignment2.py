import os
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

from data import drop_edges
from experiments import GRID, error_analysis, make, mean_cos_sim, tune
from models import KINDS
from train import fit, n_params, seed_all, test_metrics as final_test_metrics   # alias: see step 2

CFG = dict(lr=0.01, wd=5e-4, hidden=16, dropout=0.5)


@pytest.fixture
def tiny():
    g = torch.Generator().manual_seed(0)
    n = 40
    e = to_undirected(remove_self_loops(torch.randint(0, n, (2, 100), generator=g))[0])
    d = Data(x=torch.rand(n, 16, generator=g), y=torch.randint(0, 3, (n,), generator=g), edge_index=e)
    for name, sl in (("train", slice(0, 15)), ("val", slice(15, 27)), ("test", slice(27, 40))):
        m = torch.zeros(n, dtype=torch.bool); m[sl] = True; d[f"{name}_mask"] = m
    return d


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("layers", [2, 3, 4])
def test_forward_shapes(tiny, kind, layers):
    m = make(kind, tiny, CFG, layers).eval()
    out, emb = m(tiny.x, tiny.edge_index, return_emb=True)
    assert out.shape == (40, 3) and emb.shape == (40, CFG["hidden"])


def test_mlp_ignores_edges_but_gcn_does_not(tiny):
    seed_all(0); mlp = make("MLP", tiny, CFG).eval()
    seed_all(0); gcn = make("GCN", tiny, CFG).eval()
    none = torch.empty(2, 0, dtype=torch.long)
    assert torch.allclose(mlp(tiny.x, tiny.edge_index), mlp(tiny.x, none))
    assert not torch.allclose(gcn(tiny.x, tiny.edge_index), gcn(tiny.x, none))


def test_drop_edges(tiny):
    e = tiny.edge_index
    assert drop_edges(e, 0.0).size(1) == e.size(1)
    assert drop_edges(e, 1.0).size(1) == 0
    half = drop_edges(e, 0.5)
    assert abs(half.size(1) / e.size(1) - 0.5) < 0.15
    s_full = {tuple(c) for c in e.T.tolist()}
    s_half = {tuple(c) for c in half.T.tolist()}
    assert s_half <= s_full                                   # only removes, never adds
    assert {(b, a) for a, b in s_half} == s_half              # still symmetric


def test_fit_never_uses_test_labels(tiny):
    def run(d):
        seed_all(0)
        return fit(make("GCN", d, CFG), d, epochs=30)["val"]
    other = tiny.clone()
    other.y[other.test_mask] = (other.y[other.test_mask] + 1) % 3     # corrupt TEST labels only
    assert run(tiny) == run(other)


def test_same_seed_same_result(tiny):
    def run():
        seed_all(1)
        return fit(make("GAT", tiny, CFG), tiny, epochs=20)["val"]
    assert run() == run()


def test_mean_cos_sim():
    assert mean_cos_sim(torch.ones(5, 4)) == pytest.approx(1.0, abs=1e-5)
    assert mean_cos_sim(torch.eye(4)) == pytest.approx(0.0, abs=1e-6)


def test_tuning_budget_is_equal(tiny):
    tables = {k: tune(k, tiny, n_trials=3)[1] for k in ("MLP", "GCN")}
    assert len(tables["MLP"]) == len(tables["GCN"]) == 3
    cols = list(GRID)                                          # identical sampled configs per model
    a = tables["MLP"][cols].sort_values(cols).reset_index(drop=True)
    b = tables["GCN"][cols].sort_values(cols).reset_index(drop=True)
    assert a.equals(b)


def test_error_analysis_columns(tiny):
    seed_all(0)
    m, mlp = make("GCN", tiny, CFG).eval(), make("MLP", tiny, CFG).eval()
    df = error_analysis(m, mlp, tiny, k=5)
    assert 0 < len(df) <= 5
    assert {"node", "true", "gnn_pred", "degree", "nbr_same_label", "n_words", "hint"} <= set(df.columns)
    assert (df.true != df.gnn_pred).all() and tiny.test_mask[df.node.values].all()


def test_metrics_and_params(tiny):
    m = make("GraphSAGE", tiny, CFG)
    assert n_params(m) > 0
    acc, f1 = final_test_metrics(m, tiny)
    assert 0 <= acc <= 1 and 0 <= f1 <= 1


def test_cora_standard_split():
    from data import load_cora
    try:
        d = load_cora(root=os.path.join(os.path.dirname(__file__), "..", "data"))
    except Exception:
        pytest.skip("Cora not downloadable")
    assert (d.num_nodes, d.num_features, int(d.y.max()) + 1) == (2708, 1433, 7)
    assert (int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum())) == (140, 500, 1000)
    assert not (d.train_mask & d.val_mask).any() and not (d.train_mask & d.test_mask).any() and not (d.val_mask & d.test_mask).any()