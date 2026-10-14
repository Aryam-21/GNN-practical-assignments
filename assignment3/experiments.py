"""Training, baseline, valid-vs-leaky comparison, failure cases, candidate links."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from data import leaky_mp_ei, sample_negatives
from models import LinkPredictor

METHODS = {"dot decoder": "dot", "MLP decoder": "mlp"}


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def metrics(s_pos, s_neg):
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = torch.cat([s_pos, s_neg]).numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)


@torch.no_grad()
def evaluate(model, x, mp_ei, pos, neg):
    model.eval()
    z = model.encode(x, mp_ei)
    sp, sn = model.score(z, pos), model.score(z, neg)
    return (*metrics(sp, sn), sp, sn)


def common_neighbor_scores(mp_ei, pairs, n):
    A = torch.zeros(n, n)
    A[mp_ei[0], mp_ei[1]] = 1.0
    return (A[pairs[0]] * A[pairs[1]]).sum(1)


def run_baseline(split, mp_ei):
    sp, sn = (common_neighbor_scores(mp_ei, p, split.n) for p in (split.test_pos, split.test_neg))
    auc, ap = metrics(sp, sn)
    val_auc = metrics(*(common_neighbor_scores(mp_ei, p, split.n) for p in (split.val_pos, split.val_neg)))[0]
    return dict(val_auc=val_auc, test_auc=auc, test_ap=ap, s_pos=sp, s_neg=sn)


def run_gnn(split, x, decoder, mp_ei, seed=0, epochs=200, lr=0.01, wd=0.0, dropout=0.5):
    """Positives = train links; negatives resampled every epoch; best checkpoint by VAL AUC."""
    seed_all(seed)
    g = torch.Generator().manual_seed(seed)
    model = LinkPredictor(x.size(1), decoder=decoder, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos = split.train_pos
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(pos.size(1))])
    best, state, hist = -1.0, None, []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        neg = sample_negatives(split.n, split.train_forbidden, pos.size(1), g)
        z = model.encode(x, mp_ei)
        loss = F.binary_cross_entropy_with_logits(model.score(z, torch.cat([pos, neg], 1)), labels)
        loss.backward(); opt.step()
        val = evaluate(model, x, mp_ei, split.val_pos, split.val_neg)[0]
        hist.append((loss.item(), val))
        if val > best:
            best, state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    auc, ap, sp, sn = evaluate(model, x, mp_ei, split.test_pos, split.test_neg)   # test: once, at the end
    return dict(model=model, hist=hist, val_auc=best, test_auc=auc, test_ap=ap, s_pos=sp, s_neg=sn)


def compare(split, x, seeds=(0, 1, 2), **kw):
    """Valid experiment (train-only graph) vs deliberately leaky one (held-out links in the graph)."""
    rows, runs = [], {}
    for setting, mp in (("valid", split.mp_ei), ("leaky", leaky_mp_ei(split))):
        r = run_baseline(split, mp); runs[(setting, "common neighbors")] = r
        rows.append(dict(setting=setting, method="common neighbors", seed=0, test_auc=r["test_auc"], test_ap=r["test_ap"]))
        for name, dec in METHODS.items():
            for s in seeds:
                r = run_gnn(split, x, dec, mp, seed=s, **kw)
                rows.append(dict(setting=setting, method=name, seed=s, test_auc=r["test_auc"], test_ap=r["test_ap"]))
                if s == seeds[0]:
                    runs[(setting, name)] = r
    return pd.DataFrame(rows), runs


def summarize(df):
    g = df.groupby(["setting", "method"], sort=False)
    f = lambda c: g[c].apply(lambda s: f"{s.mean():.3f} ± {s.std(ddof=0):.3f}")
    return pd.DataFrame({"test_auc": f("test_auc"), "test_ap": f("test_ap")})


def failure_cases(res, split, k=5):
    """Lowest-scored true test links (missed) and highest-scored test non-links (false alarms)."""
    n, deg = split.n, torch.bincount(split.mp_ei[0], minlength=split.n)
    out = []
    for kind, pairs, idx in (("missed link", split.test_pos, res["s_pos"].argsort()[:k]),
                             ("false alarm", split.test_neg, res["s_neg"].argsort(descending=True)[:k])):
        p = pairs[:, idx]
        s = res["s_pos"][idx] if kind == "missed link" else res["s_neg"][idx]
        cn = common_neighbor_scores(split.mp_ei, p, n)
        for j in range(p.size(1)):
            u, v = int(p[0, j]), int(p[1, j])
            out.append(dict(kind=kind, u=u, v=v, score=round(s[j].item(), 2),
                            train_deg_u=int(deg[u]), train_deg_v=int(deg[v]), common_nbrs=int(cn[j])))
    return pd.DataFrame(out)


@torch.no_grad()
def top_candidates(model, x, split, y=None, k=10, block=128):
    """Highest-scored pairs that are NOT observed anywhere (train/val/test) and not self-pairs."""
    model.eval()
    z = model.encode(x, split.mp_ei)
    n = z.size(0)
    cols, best = torch.arange(n)[None, :], []
    for s in range(0, n, block):
        rows = torch.arange(s, min(s + block, n))[:, None]
        sc = model.decoder(z[rows[:, 0]][:, None, :], z[None, :, :])               # [B, N]
        key = torch.minimum(rows, cols) * n + torch.maximum(rows, cols)
        sc = sc.masked_fill((cols <= rows) | torch.isin(key, split.all_pos_keys), float("-inf"))
        v, i = sc.flatten().topk(min(k, sc.numel()))
        best.append(torch.stack([v, (s + i // n).float(), (i % n).float()]))
    best = torch.cat(best, 1)
    best = best[:, best[0].argsort(descending=True)[:k]]
    u, v = best[1].long(), best[2].long()
    df = pd.DataFrame(dict(u=u.tolist(), v=v.tolist(), prob=torch.sigmoid(best[0]).round(decimals=3).tolist(),
                           common_train_nbrs=common_neighbor_scores(split.mp_ei, torch.stack([u, v]), n).int().tolist()))
    if y is not None:
        df["class_u"], df["class_v"] = y[u].tolist(), y[v].tolist()
        df["same_class"] = df.class_u == df.class_v
    return df