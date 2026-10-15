"""Drift detection with a FIXED model: one graph statistic + one embedding statistic."""
import copy
import torch
import torch.nn.functional as F


def graph_stats(edge_index, n):
    deg = torch.bincount(edge_index[1], minlength=n).float()
    return {"mean_degree": deg.mean().item(), "isolated_frac": (deg == 0).float().mean().item()}


def snapshot_metrics(pred, ref, snap):
    """ref / snap = (x, edge_index) over the SAME node ids.
    degree_change   : relative change of mean degree           (graph structure)
    embedding_shift : 1 - mean cosine(old emb, new emb) per node (hidden layer, same model)"""
    (xr, er), (xs, es) = ref, snap
    _, zr = pred.logits(xr, er, return_emb=True)
    _, zs = pred.logits(xs, es, return_emb=True)
    d0, d1 = graph_stats(er, xr.size(0))["mean_degree"], graph_stats(es, xs.size(0))["mean_degree"]
    return {"degree_change": abs(d1 - d0) / max(d0, 1e-9),
            "embedding_shift": 1 - F.cosine_similarity(zr, zs, dim=1).mean().item()}


def calibrate(normal_runs, k=3.0):
    """Threshold = max(mean + k*std, 1.5*mean) of metrics seen under NORMAL variation:
    above normal noise, but low enough to catch real change."""
    thr = {}
    for key in normal_runs[0]:
        v = torch.tensor([m[key] for m in normal_runs])
        thr[key] = max((v.mean() + k * v.std()).item(), 1.5 * v.mean().item(), 1e-6)
    return thr


def detect(metrics, thr):
    return {k: metrics[k] > thr[k] for k in thr}


def linear_cka(X, Y):
    """Similarity of two embedding spaces that ignores rotation / permutation of coordinates."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return ((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def permute_hidden(model, perm):
    """Function-preserving relabelling of hidden units: same predictions, different raw coordinates."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.c1.lin.weight.copy_(model.c1.lin.weight[perm])
        m.c1.bias.copy_(model.c1.bias[perm])
        m.c2.lin.weight.copy_(model.c2.lin.weight[:, perm])
    return m