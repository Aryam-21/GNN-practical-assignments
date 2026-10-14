"""Run:  pytest tests -q      (tiny synthetic graph, no downloads)"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torch_geometric.utils import remove_self_loops, to_undirected

from checks import assert_no_leak, assert_symmetric_decoder, assert_valid_split
from data import both, leaky_mp_ei, pair_keys, sample_negatives, split_edges
from experiments import common_neighbor_scores, metrics, run_baseline, run_gnn, top_candidates
from models import LinkPredictor

N = 60


@pytest.fixture
def graph():
    g = torch.Generator().manual_seed(0)
    e = to_undirected(remove_self_loops(torch.randint(0, N, (2, 200), generator=g))[0])
    return torch.rand(N, 8, generator=g), e


@pytest.fixture
def split(graph):
    return split_edges(graph[1], N, val=0.1, test=0.2, seed=0)


def test_split_is_valid_and_complete(graph, split):
    assert assert_valid_split(split)
    total = split.train_pos.size(1) + split.val_pos.size(1) + split.test_pos.size(1)
    assert total == int((graph[1][0] < graph[1][1]).sum())          # every unique link in exactly one split


def test_message_passing_graph_is_train_only_both_directions(split):
    assert split.mp_ei.size(1) == 2 * split.train_pos.size(1)
    assert_no_leak(split)


def test_leak_check_detects_leaky_graph(split):
    with pytest.raises(AssertionError):
        assert_no_leak(split, leaky_mp_ei(split))
    one_dir = torch.cat([split.mp_ei, split.test_pos[:, :1]], 1)    # only ONE direction of a test link
    with pytest.raises(AssertionError):
        assert_no_leak(split, one_dir)
    reverse_only = torch.cat([split.mp_ei, split.val_pos[:, :1].flip(0)], 1)
    with pytest.raises(AssertionError):
        assert_no_leak(split, reverse_only)


def test_negatives_valid_and_fixed(graph, split):
    again = split_edges(graph[1], N, val=0.1, test=0.2, seed=0)
    assert torch.equal(split.val_neg, again.val_neg) and torch.equal(split.test_neg, again.test_neg)
    for neg in (split.val_neg, split.test_neg):
        assert (neg[0] != neg[1]).all()
        assert not torch.isin(pair_keys(neg, N), split.all_pos_keys).any()


def test_training_negatives_avoid_everything_forbidden(split):
    g = torch.Generator().manual_seed(1)
    neg = sample_negatives(N, split.train_forbidden, 100, g)
    assert not torch.isin(pair_keys(neg, N), split.train_forbidden).any()
    assert (neg[0] != neg[1]).all()


@pytest.mark.parametrize("dec", ["dot", "mlp"])
def test_decoders_symmetric(dec):
    torch.manual_seed(0)
    m = LinkPredictor(8, decoder=dec)
    z = torch.randn(20, 32)
    pairs = torch.randint(0, 20, (2, 50))
    assert assert_symmetric_decoder(m, z, pairs)


def test_common_neighbors_hand_example():
    # 0-1, 0-2, 1-2, 2-3  -> CN(1,3)=1 (node 2), CN(0,3)=1, CN(0,1)=1, CN(1,1)... 
    mp = both(torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]))
    s = common_neighbor_scores(mp, torch.tensor([[1, 0, 0], [3, 3, 1]]), 4)
    assert s.tolist() == [1.0, 1.0, 1.0]
    assert common_neighbor_scores(mp, torch.tensor([[3], [1]]), 4).item() == 1.0     # symmetric


def test_metrics_perfect_and_random():
    auc, ap = metrics(torch.tensor([0.9, 0.8]), torch.tensor([0.1, 0.2]))
    assert auc == 1.0 and ap == 1.0


@pytest.mark.parametrize("dec", ["dot", "mlp"])
def test_run_gnn_returns_valid_metrics(graph, split, dec):
    r = run_gnn(split, graph[0], dec, split.mp_ei, epochs=5)
    assert 0 <= r["test_auc"] <= 1 and 0 <= r["test_ap"] <= 1
    assert len(r["s_pos"]) == split.test_pos.size(1) and len(r["s_neg"]) == split.test_neg.size(1)


def test_baseline_uses_train_graph_only(graph, split):
    r = run_baseline(split, split.mp_ei)
    leak = run_baseline(split, leaky_mp_ei(split))
    assert 0 <= r["test_auc"] <= 1
    # a held-out link (u,v) can never be its own common neighbour pair, but leaky graph adds extra paths
    assert leak["s_pos"].sum() >= r["s_pos"].sum()


def test_top_candidates_are_unobserved(graph, split):
    torch.manual_seed(0)
    m = LinkPredictor(8, decoder="mlp")
    df = top_candidates(m, graph[0], split, k=10, block=16)
    assert len(df) == 10 and (df.u != df.v).all()
    keys = pair_keys(torch.tensor([df.u.tolist(), df.v.tolist()]), N)
    assert not torch.isin(keys, split.all_pos_keys).any()           # no observed link (any split)
    assert torch.unique(keys).numel() == 10                         # no duplicates
    assert df.prob.is_monotonic_decreasing