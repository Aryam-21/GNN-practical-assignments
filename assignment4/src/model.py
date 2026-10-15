"""GCN + training loop (used by tracking, the CI gate and tests)."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def normalize_rows(x):
    """Preprocessing (saved with the model): divide each row by its sum (rows summing to 0 stay 0)."""
    return x / x.sum(1, keepdim=True).clamp(min=1.0)


class GCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(d_in, d_hid), GCNConv(d_hid, d_out), dropout

    def forward(self, x, edge_index, return_emb=False):
        h = F.relu(self.c1(F.dropout(x, self.p, self.training), edge_index))
        out = self.c2(F.dropout(h, self.p, self.training), edge_index)
        return (out, h) if return_emb else out


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_model(g, cfg, seed=0):
    """Returns (model, best_val_acc, history[(loss, val_acc)]). Never touches test labels."""
    seed_all(seed)
    x = normalize_rows(g.x)
    model = GCN(x.size(1), cfg["hidden"], cfg["n_classes"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = g.train_mask.nonzero().flatten()[: cfg.get("n_labels") or None]      # n_labels: fewer labels = weaker model
    best, state, hist = -1.0, None, []
    for _ in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, g.edge_index)[tr], g.y[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = (model(x, g.edge_index).argmax(1)[g.val_mask] == g.y[g.val_mask]).float().mean().item()
        hist.append((loss.item(), val))
        if val > best:
            best, state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), best, hist