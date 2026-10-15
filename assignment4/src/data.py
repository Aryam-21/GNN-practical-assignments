"""Cora, inductive hold-out (unseen nodes), and data snapshots for drift monitoring."""
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph


def load_cora(root="../data"):
    return Planetoid(root, "Cora")[0]                  # RAW binary features; preprocessing lives in the package


def hold_out_unseen(d, n_unseen=200, seed=0):
    """Pick test nodes that will be 'introduced after training'."""
    idx = d.test_mask.nonzero().flatten()
    return idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(seed))[:n_unseen]]


def training_graph(d, unseen):
    """Training graph WITHOUT the unseen nodes and WITHOUT their incident edges."""
    keep = torch.ones(d.num_nodes, dtype=torch.bool)
    keep[unseen] = False
    ei, _ = subgraph(keep, d.edge_index, relabel_nodes=True, num_nodes=d.num_nodes)
    return Data(x=d.x[keep], y=d.y[keep], edge_index=ei, train_mask=d.train_mask[keep],
                val_mask=d.val_mask[keep], test_mask=d.test_mask[keep])


def drop_edges(edge_index, frac, seed=0):
    und = edge_index[:, edge_index[0] < edge_index[1]]
    und = und[:, torch.rand(und.size(1), generator=torch.Generator().manual_seed(seed)) >= frac]
    return torch.cat([und, und.flip(0)], 1)


def make_snapshot(d, edge_drop=0.0, feat_drop=0.0, seed=0):
    """A later 'snapshot' of the same nodes: some edges vanish / some words are missing."""
    keep = torch.rand(d.x.shape, generator=torch.Generator().manual_seed(seed)) >= feat_drop
    return d.x * keep, drop_edges(d.edge_index, edge_drop, seed)