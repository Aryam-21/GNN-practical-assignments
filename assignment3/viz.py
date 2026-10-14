"""All plots."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve


def plot_split(split):
    from data import leaky_mp_ei
    names = ["train +", "val +", "test +", "val −", "test −"]
    sizes = [split.train_pos.size(1), split.val_pos.size(1), split.test_pos.size(1),
             split.val_neg.size(1), split.test_neg.size(1)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    ax[0].bar(names, sizes, color=["tab:green"] * 3 + ["tab:red"] * 2); ax[0].set_title("Unique undirected pairs per split")
    ax[1].bar(["valid graph", "leaky graph"], [split.mp_ei.size(1), leaky_mp_ei(split).size(1)], color=["tab:green", "tab:red"])
    ax[1].set_title("Directed edges used for message passing")
    plt.tight_layout(); plt.show()


def plot_curves(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    for name, r in runs.items():
        h = np.array(r["hist"]); ax[0].plot(h[:, 0], label=name); ax[1].plot(h[:, 1], label=name)
    ax[0].set_title("Train loss"); ax[1].set_title("Validation ROC-AUC")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.show()


def plot_roc(runs):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, r in runs.items():
        y = np.r_[np.ones(len(r["s_pos"])), np.zeros(len(r["s_neg"]))]
        fpr, tpr, _ = roc_curve(y, torch.cat([r["s_pos"], r["s_neg"]]).numpy())
        ax.plot(fpr, tpr, label=f"{name} ({r['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Test ROC (valid setting)"); ax.legend()
    plt.show()


def plot_valid_vs_leaky(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for a, col in zip(ax, ("test_auc", "test_ap")):
        m = df.groupby(["method", "setting"], sort=False)[col].mean().unstack()
        m[["valid", "leaky"]].plot.bar(ax=a, color=["tab:green", "tab:red"], rot=15); a.set_ylim(0.5, 1.02); a.set_title(col)
    plt.tight_layout(); plt.show()


def plot_score_dist(valid_run, leaky_run, name="MLP decoder"):
    """Held-out positives look far more 'linked' when their own edge is in the graph."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    for a, r, t in zip(ax, (valid_run, leaky_run), ("valid", "LEAKY")):
        a.hist(torch.sigmoid(r["s_neg"]).numpy(), bins=30, alpha=.6, label="test negatives")
        a.hist(torch.sigmoid(r["s_pos"]).numpy(), bins=30, alpha=.6, label="test positives")
        a.set_title(f"{name}: {t}"); a.set_xlabel("predicted probability"); a.legend()
    plt.tight_layout(); plt.show()