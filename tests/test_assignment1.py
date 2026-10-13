"""Automated checks for Assignment 1. Run:  pytest -q"""
import numpy as np
import pytest
import torch

from data import make_graph, make_splits, edge_homophily
from models import GCN, MLP, add_self_loops, gcn_norm, propagate
from train import train
from checks import (manual_layer_check, permutation_check, random_edges,
                    eval_with_edges, ablation)

N = 200


@pytest.fixture(scope="module")
def data():
    x, y, ei = make_graph(n=N, seed=0)
    return x, y, ei, make_splits(N, seed=0)


# ---------- dataset ----------
def test_dataset_size_features_classes(data):
    x, y, ei, _ = data
    assert x.shape == (N, 3) and N >= 100
    assert set(y.tolist()) == {0, 1}


def test_graph_is_undirected_without_self_loops_or_duplicates(data):
    _, _, ei, _ = data
    edges = set(map(tuple, ei.T.tolist()))
    assert len(edges) == ei.size(1)                              # no duplicates
    assert all((v, u) in edges for u, v in edges)                # both directions present
    assert all(u != v for u, v in edges)                         # no self-loops


def test_edges_carry_label_information(data):
    _, y, ei, _ = data
    assert edge_homophily(ei, y) > 0.7                           # 0.5 would be random


def test_labels_are_not_in_features(data):
    x, y, _, _ = data
    for k in range(3):
        assert not torch.equal(x[:, k], y.float())
        assert abs(np.corrcoef(x[:, k].numpy(), y.numpy())[0, 1]) < 0.9


def test_splits_are_disjoint_and_cover_all_nodes(data):
    tr, va, te = data[3]
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()


# ---------- GCN building blocks ----------
def test_self_loops_added():
    ei = torch.tensor([[0, 1], [1, 0]])
    out = add_self_loops(ei, 3)
    assert out.shape == (2, 5)
    assert out[:, 2:].tolist() == [[0, 1, 2], [0, 1, 2]]


def test_degree_normalisation_on_tiny_graph():
    # path 0-1-2 with self-loops: deg = [2, 3, 2]
    ei = add_self_loops(torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 3)
    w = gcn_norm(ei, 3)
    for (s, d), wt in zip(ei.T.tolist(), w.tolist()):
        deg = [2, 3, 2]
        assert wt == pytest.approx(1 / (deg[s] * deg[d]) ** 0.5)


def test_propagate_equals_dense_matrix_multiplication(data):
    x, _, ei, _ = data
    n = x.size(0)
    e = add_self_loops(ei, n)
    w = gcn_norm(e, n)
    A_hat = torch.zeros(n, n)
    A_hat[e[1], e[0]] = w                                         # A_hat[dst, src]
    assert torch.allclose(propagate(x, e, w), A_hat @ x, atol=1e-5)


# ---------- required correctness checks ----------
def test_manual_node_calculation_matches_layer_output(data):
    x, _, ei, _ = data
    torch.manual_seed(0)
    r = manual_layer_check(GCN(3, 16, 2), x, ei)
    assert r["match"] and len(r["neighbours"]) >= 4              # >= 3 neighbours + itself


def test_permutation_equivariance(data):
    x, _, ei, _ = data
    torch.manual_seed(0)
    ok, err = permutation_check(GCN(3, 16, 2), x, ei)
    assert ok and err < 1e-5


def test_permutation_check_can_fail(data):
    """Negative control: permuting features but NOT the edges must change the output."""
    x, _, ei, _ = data
    torch.manual_seed(0)
    model = GCN(3, 16, 2).eval()
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert not torch.allclose(model(x, ei)[perm], model(x[perm], ei), atol=1e-5)


# ---------- training hygiene ----------
def test_test_labels_do_not_influence_training(data):
    """Changing labels of TEST nodes must not change the learned weights."""
    x, y, ei, masks = data
    y2 = y.clone(); y2[masks[2]] = 1 - y2[masks[2]]
    states = []
    for labels in (y, y2):
        torch.manual_seed(0)
        m = GCN(3, 16, 2)
        train(m, x, ei, labels, masks, epochs=30)
        states.append(m.state_dict())
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k])


# ---------- random edges ----------
def test_random_edges_same_count_valid_and_destroys_homophily(data):
    x, y, ei, _ = data
    r = random_edges(ei, N)
    assert r.size(1) == ei.size(1)                               # same number of edges
    assert (r[0] != r[1]).all()                                   # no self-loops
    assert len(set(map(tuple, r.T.tolist()))) == r.size(1)        # no duplicates
    assert edge_homophily(r, y) < 0.6 < edge_homophily(ei, y)


# ---------- the real question: does the model use the graph? ----------
def test_gcn_beats_mlp_and_depends_on_edges(data):
    x, y, ei, masks = data
    res = ablation(x, y, ei, masks, seeds=range(3))
    m = {k: np.mean(v) for k, v in res.items()}
    assert m["GCN (real edges)"] > m["MLP (no edges)"] + 0.05
    assert m["GCN (real edges)"] > m["GCN (random edges)"] + 0.05
    assert m["GCN (real edges)"] > m["GCN (self-loops only)"] + 0.05


def test_trained_gcn_degrades_when_edges_removed_at_test_time(data):
    x, y, ei, masks = data
    torch.manual_seed(0)
    model = GCN(3, 16, 2)
    train(model, x, ei, y, masks)
    real = eval_with_edges(model, x, ei, y, masks[2])
    rand = eval_with_edges(model, x, random_edges(ei, N), y, masks[2])
    none = eval_with_edges(model, x, torch.empty(2, 0, dtype=torch.long), y, masks[2])
    assert real > rand and real > none