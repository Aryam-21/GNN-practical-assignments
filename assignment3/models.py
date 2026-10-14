"""GCN encoder + two decoders (both symmetric in the node pair)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DotDecoder(nn.Module):
    def forward(self, zu, zv):
        return (zu * zv).sum(-1)


class MLPDecoder(nn.Module):
    """Input [zu*zv, |zu-zv|] is unchanged when u and v swap -> score(u,v) == score(v,u)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, zu, zv):
        return self.net(torch.cat([zu * zv, (zu - zv).abs()], -1)).squeeze(-1)


class LinkPredictor(nn.Module):
    def __init__(self, d_in, d_hid=64, d_out=32, decoder="dot", dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(d_in, d_hid), GCNConv(d_hid, d_out), dropout
        self.decoder = DotDecoder() if decoder == "dot" else MLPDecoder(d_out)

    def encode(self, x, edge_index):
        x = F.relu(self.c1(F.dropout(x, self.p, self.training), edge_index))
        return self.c2(F.dropout(x, self.p, self.training), edge_index)

    def score(self, z, pairs):                 # pairs: [2, P]
        return self.decoder(z[pairs[0]], z[pairs[1]])