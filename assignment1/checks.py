"""Correctness checks + structure ablation."""
import torch
from data import make_graph, make_splits
from models import GCN, MLP, add_self_loops, gcn_norm
from train import train, accuracy


def manual_layer_check(model, x, edge_index, node=None):
    """Recompute ONE node's layer-1 output with a plain python loop and compare."""
    n = x.size(0)
    raw_deg = torch.bincount(edge_index[1], minlength=n)        # neighbours, from the RAW edge list
    if node is None:                                            # first node with >= 3 neighbours
        node = int((raw_deg >= 3).nonzero()[0])
    deg = (raw_deg + 1).float()                                 # +1 for the self-loop
    nbrs = edge_index[0][edge_index[1] == node].tolist() + [node]   # neighbours + itself

    model.eval()
    with torch.no_grad():
        ei = add_self_loops(edge_index, n)
        auto = model.l1(x, ei, gcn_norm(ei, n))[node]           # what the model computes
        manual = model.l1.b.clone()                             # what we compute by hand
        for j in nbrs:
            manual = manual + (x[j] @ model.l1.W) / torch.sqrt(deg[node] * deg[j])
    return dict(node=node, neighbours=nbrs, auto=auto, manual=manual,
                match=torch.allclose(auto, manual, atol=1e-6))


def permutation_check(model, x, edge_index, seed=0):
    """Relabel nodes (and shuffle edge order) consistently -> outputs must be permuted the same way."""
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    perm = torch.randperm(n, generator=g)              # new node i == old node perm[i]
    inv = torch.empty_like(perm); inv[perm] = torch.arange(n)   # old id -> new id
    ei2 = inv[edge_index][:, torch.randperm(edge_index.size(1), generator=g)]
    model.eval()
    with torch.no_grad():
        out, out2 = model(x, edge_index), model(x[perm], ei2)
    err = (out[perm] - out2).abs().max().item()
    return err < 1e-5, err


def random_edges(edge_index, n, seed=0):
    """Same number of undirected edges, but random endpoints (destroys class information)."""
    g = torch.Generator().manual_seed(seed)
    m = int((edge_index[0] < edge_index[1]).sum())     # number of undirected edges
    iu = torch.triu_indices(n, n, offset=1)            # all possible pairs (no self-loops)
    pick = torch.randperm(iu.size(1), generator=g)[:m] # sample m distinct pairs
    e = iu[:, pick]
    return torch.cat([e, e.flip(0)], dim=1)            # store both directions


@torch.no_grad()
def eval_with_edges(model, x, edge_index, y, mask):
    model.eval()
    return accuracy(model(x, edge_index), y, mask)


def ablation(x, y, edge_index, masks, seeds=range(5), hidden=16):
    """Train a fresh model per seed under each setting; return {name: [test accuracies]}.
    Same features, same split everywhere; only the edges / model type change."""
    n = x.size(0)
    settings = {
        "MLP (no edges)":        (MLP, edge_index),
        "GCN (real edges)":      (GCN, edge_index),
        "GCN (random edges)":    (GCN, random_edges(edge_index, n)),
        "GCN (self-loops only)": (GCN, torch.empty(2, 0, dtype=torch.long)),
    }
    res = {}
    for name, (cls, ei) in settings.items():
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            _, _, test = train(cls(x.size(1), hidden, 2), x, ei, y, masks)
            accs.append(test)
        res[name] = accs
    return res


def ablation_many_graphs(seeds=range(5), n=200, hidden=16):
    """Stronger version: a NEW graph + split per seed, so the std also reflects data randomness."""
    res = {}
    for s in seeds:
        x, y, ei = make_graph(n=n, seed=s)
        r = ablation(x, y, ei, make_splits(n, seed=s), seeds=[s], hidden=hidden)
        for k, v in r.items():
            res.setdefault(k, []).extend(v)
    return res