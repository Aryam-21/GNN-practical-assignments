"""Untracked baseline: trains and prints one number. Compare its limits with the MLOps pipeline."""
import torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

d = Planetoid("../data", "Cora", transform=T.NormalizeFeatures())[0]
c1, c2 = GCNConv(d.num_features, 16), GCNConv(16, 7)
opt = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=0.01, weight_decay=5e-4)
f = lambda: c2(F.dropout(F.relu(c1(d.x, d.edge_index)), 0.5), d.edge_index)
for _ in range(200):
    opt.zero_grad(); F.cross_entropy(f()[d.train_mask], d.y[d.train_mask]).backward(); opt.step()
print("test acc:", (f().argmax(1)[d.test_mask] == d.y[d.test_mask]).float().mean().item())
torch.save({"c1": c1.state_dict(), "c2": c2.state_dict()}, "model.pt")   # overwritten on every run