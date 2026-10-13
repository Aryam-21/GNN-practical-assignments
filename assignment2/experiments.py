"""Tuning, final comparison, depth study, edge-removal study, error analysis."""
import random
from collections import Counter
import pandas as pd
import torch
import torch.nn.functional as F
from data import drop_edges
from models import KINDS
from train import fit, n_params, predict, seed_all, test_metrics

GRID = dict(lr=[0.005, 0.01, 0.02], wd=[5e-4, 1e-3, 5e-3], hidden=[16, 32, 64], dropout=[0.3, 0.5, 0.7])


def make(kind, data, cfg, layers=2):
    return KINDS[kind](data.num_features, cfg["hidden"], int(data.y.max()) + 1, layers, cfg["dropout"])


def tune(kind, data, n_trials=8, seed=0):
    """Same sampled configs (same budget) for every model; chosen by validation accuracy only."""
    rng = random.Random(0)
    configs = [{k: rng.choice(v) for k, v in GRID.items()} for _ in range(n_trials)]
    rows = []
    for cfg in configs:
        seed_all(seed)
        r = fit(make(kind, data, cfg), data, lr=cfg["lr"], wd=cfg["wd"])
        rows.append({**cfg, "val": r["val"]})
    best = max(rows, key=lambda r: r["val"])
    return {k: best[k] for k in GRID}, pd.DataFrame(rows).sort_values("val", ascending=False)


def final(kind, cfg, data, seeds=(0, 1, 2), layers=2, ei=None):
    rows, runs = [], []
    for s in seeds:
        seed_all(s)
        m = make(kind, data, cfg, layers)
        r = fit(m, data, ei, cfg["lr"], cfg["wd"])
        acc, f1 = test_metrics(r["model"], data, ei)
        rows.append(dict(model=kind, seed=s, val=r["val"], test_acc=acc, macro_f1=f1,
                         time_s=r["time"], params=n_params(m)))
        runs.append(r)
    return pd.DataFrame(rows), runs


def summarize(df):
    g = df.groupby("model", sort=False)
    return pd.DataFrame({
        "test_acc": g.test_acc.apply(lambda s: f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"),
        "macro_f1": g.macro_f1.apply(lambda s: f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"),
        "train_time_s": g.time_s.mean().round(2), "params": g.params.first()})


def mean_cos_sim(z):
    """Average cosine similarity between all pairs of node embeddings (1.0 = everything collapsed)."""
    z = F.normalize(z, dim=1)
    s, n = z @ z.T, z.size(0)
    return ((s.sum() - s.diagonal().sum()) / (n * (n - 1))).item()


def depth_study(kinds, cfgs, data, depths=(2, 3, 4, 6, 8), seeds=(0, 1, 2)):
    rows = []
    for kind in kinds:
        for L in depths:
            for s in seeds:
                seed_all(s)
                m = make(kind, data, cfgs[kind], L)
                r = fit(m, data, lr=cfgs[kind]["lr"], wd=cfgs[kind]["wd"])
                acc, _ = test_metrics(r["model"], data)
                with torch.no_grad():
                    _, emb = r["model"].eval()(data.x, data.edge_index, return_emb=True)
                rows.append(dict(model=kind, depth=L, seed=s, test_acc=acc, cos_sim=mean_cos_sim(emb)))
    return pd.DataFrame(rows)


def edge_study(cfgs, data, fracs=(0, 0.25, 0.5, 0.75, 0.9, 1.0), seeds=(0, 1, 2)):
    """Edges removed for training AND evaluation. MLP is the control (should be flat)."""
    rows = []
    for kind, cfg in cfgs.items():
        for f in fracs:
            for s in seeds:
                ei = drop_edges(data.edge_index, f, seed=s)
                seed_all(s)
                r = fit(make(kind, data, cfg), data, ei, cfg["lr"], cfg["wd"])
                acc, _ = test_metrics(r["model"], data, ei)
                rows.append(dict(model=kind, frac_removed=f, seed=s, test_acc=acc))
    return pd.DataFrame(rows)


def error_analysis(model, mlp, data, k=5):
    """The k most confident WRONG test predictions, with feature and neighbourhood evidence."""
    out = predict(model, data)
    prob, pred = out.softmax(1), out.argmax(1)
    mlp_pred = predict(mlp, data).argmax(1)
    idx = ((pred != data.y) & data.test_mask).nonzero().flatten()
    idx = idx[prob[idx].max(1).values.argsort(descending=True)][:k]
    rows = []
    for i in idx.tolist():
        nb = data.edge_index[1][data.edge_index[0] == i]
        same = (data.y[nb] == data.y[i]).float().mean().item() if len(nb) else float("nan")
        if len(nb) <= 1:                 hint = "very low degree: little context"
        elif same < 0.5:                 hint = "neighbours mostly other classes (heterophily)"
        elif mlp_pred[i] == data.y[i]:   hint = "MLP right, GNN misled by neighbourhood"
        else:                            hint = "ambiguous features, neighbours not decisive"
        rows.append(dict(node=i, true=int(data.y[i]), gnn_pred=int(pred[i]), mlp_pred=int(mlp_pred[i]),
                         conf=round(prob[i].max().item(), 2), degree=len(nb),
                         nbr_same_label=round(same, 2), nbr_true_labels=dict(Counter(data.y[nb].tolist())),
                         nbr_pred_labels=dict(Counter(pred[nb].tolist())),
                         n_words=int((data.x[i] > 0).sum()), hint=hint))
    return pd.DataFrame(rows)