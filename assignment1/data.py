"""Synthetic graph: 2 classes, 3 features, edges that carry extra label information."""
import torch


def make_graph(n=200, p_in=0.08, p_out=0.01, shift=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Labels: balanced, randomly assigned to nodes
    y = (torch.arange(n) % 2)[torch.randperm(n, generator=g)]
    # Features: class mean + N(0,1) noise. 'shift' is small -> features alone are only weakly informative.
    # Column 3 has the same mean for both classes -> pure noise.
    mu = torch.tensor([[-1.0, 0.5, 0.0], [1.0, -0.5, 0.0]]) * shift
    x = mu[y] + torch.randn(n, 3, generator=g)
    # Edges (stochastic block model): same-class pairs connect with p_in, different-class with p_out
    same = y[:, None] == y[None, :]
    p = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    A = torch.triu((torch.rand(n, n, generator=g) < p).int(), 1)
    A = ((A + A.T) > 0)                               # undirected, no self-loops
    edge_index = A.nonzero().T.contiguous()           # shape [2, E], both directions stored
    return x, y, edge_index


def make_splits(n, train=0.3, val=0.2, seed=0):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a, b = int(n * train), int(n * (train + val))
    masks = []
    for idx in (perm[:a], perm[a:b], perm[b:]):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        masks.append(m)
    return masks  # train, val, test


def edge_homophily(edge_index, y):
    """Fraction of edges that connect nodes of the same class."""
    return (y[edge_index[0]] == y[edge_index[1]]).float().mean().item()