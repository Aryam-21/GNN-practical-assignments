"""Cora loading, inspection, and edge removal."""
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_cora(root="../data"):
    return Planetoid(root, "Cora", transform=T.NormalizeFeatures())[0]


def describe(d):
    src, dst = d.edge_index
    deg = torch.bincount(dst, minlength=d.num_nodes)
    return {
        "nodes": d.num_nodes, "undirected edges": d.edge_index.size(1) // 2,
        "features (bag-of-words)": d.num_features, "classes": int(d.y.max()) + 1,
        "train/val/test": (int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum())),
        "avg degree": round(deg.float().mean().item(), 2), "isolated nodes": int((deg == 0).sum()),
        "edge homophily": round((d.y[src] == d.y[dst]).float().mean().item(), 3),
        "has self-loops": bool(d.has_self_loops()),
    }


def drop_edges(edge_index, frac, seed=0):
    """Remove a fraction of UNDIRECTED edges (both directions together)."""
    und = edge_index[:, edge_index[0] < edge_index[1]]
    keep = torch.rand(und.size(1), generator=torch.Generator().manual_seed(seed)) >= frac
    und = und[:, keep]
    return torch.cat([und, und.flip(0)], dim=1)