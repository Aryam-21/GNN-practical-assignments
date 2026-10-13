"""Our own model classes. Convolution layers come from PyG; depth/dropout/forward are ours."""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class Base(nn.Module):
    """`layers` message-passing layers; last one outputs class logits.
    forward(..., return_emb=True) also returns the last hidden representation."""
    def __init__(self, d_in, d_hid, d_out, layers=2, dropout=0.5):
        super().__init__()
        assert layers >= 2
        self.p = dropout                       # set BEFORE building the layers (GAT needs it)
        dims = [d_in] + [d_hid] * (layers - 1) + [d_out]
        self.convs = nn.ModuleList([self.layer(dims[i], dims[i + 1], i == layers - 1) for i in range(layers)])

    def forward(self, x, edge_index, return_emb=False):
        for conv in self.convs[:-1]:
            x = F.elu(conv(F.dropout(x, self.p, self.training), edge_index))
        out = self.convs[-1](F.dropout(x, self.p, self.training), edge_index)
        return (out, x) if return_emb else out


class _Lin(nn.Linear):                          # linear layer that ignores edges
    def forward(self, x, edge_index=None):
        return super().forward(x)


class MLP(Base):
    def layer(self, i, o, last):
        return _Lin(i, o)


class GCN(Base):
    def layer(self, i, o, last):
        return GCNConv(i, o)


class SAGE(Base):
    def layer(self, i, o, last):
        return SAGEConv(i, o)


class GAT(Base):
    heads = 8                                   # hidden size is split over heads (hidden % 8 == 0)

    def layer(self, i, o, last):
        if last:
            return GATConv(i, o, heads=1, dropout=self.p)
        return GATConv(i, o // self.heads, heads=self.heads, dropout=self.p)


KINDS = {"MLP": MLP, "GCN": GCN, "GraphSAGE": SAGE, "GAT": GAT}