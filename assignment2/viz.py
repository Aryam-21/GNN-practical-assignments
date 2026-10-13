"""All plots."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from train import predict


def plot_dataset(d):
    deg = torch.bincount(d.edge_index[1], minlength=d.num_nodes).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    ax[0].bar(*np.unique(d.y.numpy(), return_counts=True)); ax[0].set_title("Nodes per class"); ax[0].set_xlabel("class")
    ax[1].hist(deg, bins=40, log=True); ax[1].set_title("Degree distribution (log y)")
    ax[2].hist((d.x > 0).sum(1).numpy(), bins=30); ax[2].set_title("Words per document")
    plt.tight_layout(); plt.show()


def plot_tuning(tables):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.boxplot([t.val.values for t in tables.values()])
    ax.set_xticks(range(1, len(tables) + 1), list(tables))
    ax.set_ylabel("validation accuracy"); ax.set_title("Tuning trials (same budget per model)")
    plt.show()

def plot_results(df):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    for a, col, t in zip(ax, ("test_acc", "macro_f1", "time_s"), ("Test accuracy", "Macro F1", "Train time (s)")):
        g = df.groupby("model", sort=False)[col]
        a.bar(g.mean().index, g.mean(), yerr=g.std(ddof=0), capsize=4); a.set_title(t)
        if col != "time_s": a.set_ylim(0.4, 0.9)
    plt.tight_layout(); plt.show()


def plot_curves(runs_by_model):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, r in runs_by_model.items():
        h = np.array(r["hist"])
        ax[0].plot(h[:, 0], label=name); ax[1].plot(h[:, 1], label=name)
    ax[0].set_title("Train loss (seed 0)"); ax[1].set_title("Validation accuracy (seed 0)")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.show()


def plot_depth(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for kind, g in df.groupby("model"):
        m, s = g.groupby("depth").test_acc.mean(), g.groupby("depth").test_acc.std(ddof=0)
        ax[0].errorbar(m.index, m, s, marker="o", label=kind, capsize=3)
        c = g.groupby("depth").cos_sim.mean()
        ax[1].plot(c.index, c.values, marker="o", label=kind)
    ax[0].set_title("Test accuracy vs depth"); ax[1].set_title("Mean pairwise cosine similarity vs depth")
    for a in ax: a.set_xlabel("# layers"); a.legend()
    plt.tight_layout(); plt.show()


def plot_edges(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind, g in df.groupby("model", sort=False):
        m, s = g.groupby("frac_removed").test_acc.mean(), g.groupby("frac_removed").test_acc.std(ddof=0)
        ax.errorbar(m.index, m, s, marker="o", label=kind, capsize=3, ls="--" if kind == "MLP" else "-")
    ax.set_xlabel("fraction of edges removed"); ax.set_ylabel("test accuracy"); ax.legend()
    ax.set_title("Edge removal (MLP = control)"); plt.show()


def plot_ego(d, err_df):
    """1-hop neighbourhood of each misclassified node; colour = TRUE class, black ring = the node."""
    k = len(err_df)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.4), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (_, r) in zip(axes[0], err_df.iterrows()):
        i = r["node"]
        nb = d.edge_index[1][d.edge_index[0] == i].tolist()
        G = nx.star_graph(len(nb)) if nb else nx.empty_graph(1)
        nodes = [i] + nb                      # star_graph: node 0 = centre, 1..k = neighbours
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[300] + [120] * len(nb),
                               node_color=[cmap(int(d.y[n])) for n in nodes],
                               edgecolors=["k"] + ["none"] * len(nb), linewidths=2)
        ax.set_title(f"node {i}: true {r['true']}, pred {r['gnn_pred']}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()


def plot_confusion(model, data, title=""):
    pred = predict(model, data).argmax(1)[data.test_mask]
    true, c = data.y[data.test_mask], int(data.y.max()) + 1
    cm = torch.zeros(c, c, dtype=torch.long)
    for t, p in zip(true, pred): cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
    for i in range(c):
        for j in range(c): ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} confusion (test)")
    plt.tight_layout(); plt.show()


@torch.no_grad()
def plot_embeddings(models, data):          # models = {"MLP": model, "GCN": model, ...}
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4), squeeze=False)
    for ax, (name, m) in zip(axes[0], models.items()):
        _, h = m.eval()(data.x, data.edge_index, return_emb=True)
        h = h - h.mean(0); z = (h @ torch.linalg.svd(h, full_matrices=False)[2][:2].T).numpy()
        ax.scatter(z[:, 0], z[:, 1], c=data.y.numpy(), cmap="tab10", s=4)
        ax.set_title(f"{name}: hidden embeddings (PCA)")
    plt.tight_layout(); plt.show()