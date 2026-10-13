"""Own training + evaluation code. `fit` never touches the test set."""
import random, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, data, ei=None):
    model.eval()
    return model(data.x, data.edge_index if ei is None else ei)


def val_acc(model, data, ei=None):
    return (predict(model, data, ei).argmax(1)[data.val_mask] == data.y[data.val_mask]).float().mean().item()


def fit(model, data, ei=None, lr=0.01, wd=5e-4, epochs=200, patience=50):
    """Train on train labels; keep the checkpoint with best VALIDATION accuracy."""
    ei = data.edge_index if ei is None else ei
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad, hist, t0 = -1.0, None, 0, [], time.time()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(data.x, ei)[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        v = val_acc(model, data, ei)
        hist.append((loss.item(), v))
        if v > best:
            best, bad, state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    secs = time.time() - t0
    model.load_state_dict(state)
    return {"model": model, "val": best, "time": secs, "hist": hist}


def test_metrics(model, data, ei=None):
    """Call ONLY for final reporting."""
    pred = predict(model, data, ei).argmax(1)[data.test_mask].numpy()
    true = data.y[data.test_mask].numpy()
    return float((pred == true).mean()), float(f1_score(true, pred, average="macro"))