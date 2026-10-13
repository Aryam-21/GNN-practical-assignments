"""GCN from scratch (tensor ops only) + MLP baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_self_loops(edge_index, n):
    """Append edge (i -> i) for every node."""
    loops = torch.arange(n).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def gcn_norm(edge_index, n):
    """Per-edge weight 1/sqrt(deg_src * deg_dst). Degree counts the self-loop."""
    src, dst = edge_index
    deg = torch.zeros(n).index_add_(0, dst, torch.ones(dst.numel()))
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt[src] * inv_sqrt[dst]


def propagate(h, edge_index, w):
    """Message passing: each edge sends w * h[src] to dst; messages are summed at dst."""
    src, dst = edge_index
    out = torch.zeros_like(h)
    return out.index_add_(0, dst, w[:, None] * h[src])


class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, edge_index, w):          # H = A_hat (X W) + b
        return propagate(x @ self.W, edge_index, w) + self.b


class GCN(nn.Module):
    """Two layers: A_hat · ReLU(A_hat · X · W1) · W2.
    Dropout is applied to the hidden layer only (with just 3 input features,
    dropping inputs would throw away half of the little signal we have)."""

    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.p = GCNLayer(d_in, d_hid), GCNLayer(d_hid, d_out), dropout

    def forward(self, x, edge_index, return_hidden=False):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        w = gcn_norm(ei, n)
        h = F.relu(self.l1(x, ei, w))
        out = self.l2(F.dropout(h, self.p, self.training), ei, w)
        return (out, h) if return_hidden else out


class MLP(nn.Module):
    """Same sizes, same features, NO edges. Accepts edge_index only to share the GCN interface."""

    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_out), dropout

    def forward(self, x, edge_index=None, return_hidden=False):
        h = F.relu(self.l1(x))
        out = self.l2(F.dropout(h, self.p, self.training))
        return (out, h) if return_hidden else out