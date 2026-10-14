"""Cora loading + leakage-safe link split (positives AND negatives)."""
from dataclasses import dataclass
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_cora(root="../data"):
    return Planetoid(root, "Cora", transform=T.NormalizeFeatures())[0]


def both(e):                                    # (u,v) -> (u,v) and (v,u)
    return torch.cat([e, e.flip(0)], dim=1)


def pair_keys(e, n):                            # direction-free id of an undirected pair
    return torch.minimum(e[0], e[1]) * n + torch.maximum(e[0], e[1])


def sample_negatives(n, forbidden_keys, count, g):
    """`count` unique undirected pairs (u<v): no self-links, none in `forbidden_keys`."""
    got = torch.empty(0, dtype=torch.long)
    while got.numel() < count:
        u = torch.randint(0, n, (2 * count,), generator=g)
        v = torch.randint(0, n, (2 * count,), generator=g)
        k = (torch.minimum(u, v) * n + torch.maximum(u, v))[u != v]
        k = k[~torch.isin(k, forbidden_keys)]
        got = torch.unique(torch.cat([got, k]))
    got = got[torch.randperm(got.numel(), generator=g)[:count]]
    return torch.stack([got // n, got % n])


@dataclass
class Split:
    n: int
    train_pos: torch.Tensor      # [2, E_train]  one direction (u<v)
    val_pos: torch.Tensor
    test_pos: torch.Tensor
    val_neg: torch.Tensor        # fixed evaluation negatives
    test_neg: torch.Tensor
    mp_ei: torch.Tensor          # message-passing graph = TRAIN links, both directions
    all_pos_keys: torch.Tensor   # every observed link (train+val+test), sorted
    train_forbidden: torch.Tensor  # pairs training-negative sampling must avoid


def split_edges(edge_index, n, val=0.05, test=0.10, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = edge_index[:, edge_index[0] < edge_index[1]]          # unique undirected links
    pos = pos[:, torch.randperm(pos.size(1), generator=g)]
    nt, nv = int(test * pos.size(1)), int(val * pos.size(1))
    test_pos, val_pos, train_pos = pos[:, :nt], pos[:, nt:nt + nv], pos[:, nt + nv:]
    all_keys = torch.sort(pair_keys(pos, n)).values
    neg = sample_negatives(n, all_keys, nt + nv, g)             # one pool -> val/test can't overlap
    val_neg, test_neg = neg[:, :nv], neg[:, nv:]
    forbidden = torch.unique(torch.cat([all_keys, pair_keys(neg, n)]))
    return Split(n, train_pos, val_pos, test_pos, val_neg, test_neg, both(train_pos), all_keys, forbidden)


def leaky_mp_ei(split):
    """DELIBERATELY WRONG graph: also contains held-out positives (for the leakage experiment only)."""
    return both(torch.cat([split.train_pos, split.val_pos, split.test_pos], 1))