"""All plots."""
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(hists):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    for name, h in hists.items():
        h = np.array(h); ax[0].plot(h[:, 0], label=name); ax[1].plot(h[:, 1], label=name)
    ax[0].set_title("Train loss"); ax[1].set_title("Validation accuracy")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.tight_layout(); plt.show()


def plot_runs(df):
    d = df.sort_values("start_time")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.bar(d["tags.mlflow.runName"], d["metrics.best_val_acc"], color="tab:blue")
    ax.set_ylabel("best validation accuracy"); ax.set_title("Tracked experiments (MLflow)")
    plt.xticks(rotation=15); plt.tight_layout(); plt.show()


def plot_seen_unseen(acc):
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.bar(list(acc), list(acc.values()), color=["tab:green", "tab:orange"])
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy"); ax.set_title("Seen vs unseen (inductive) nodes")
    plt.show()


def plot_hops(res):
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.bar([str(h) for h in res], list(res.values()), color="tab:purple")
    ax.set_xlabel("hops of graph context sent"); ax.set_ylabel("agreement with full-graph prediction"); ax.set_ylim(0, 1.05)
    ax.set_title("Context size vs exactness"); plt.show()


def plot_drift(thr, normal, drift):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for a, k in zip(ax, thr):
        a.bar(["normal", "drifted"], [normal[k], drift[k]], color=["tab:green", "tab:red"])
        a.axhline(thr[k], color="k", ls="--", label="threshold"); a.set_title(k); a.legend()
    plt.tight_layout(); plt.show()