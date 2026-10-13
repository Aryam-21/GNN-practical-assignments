"""All plotting helpers."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from models import add_self_loops, gcn_norm

COL = ["tab:blue", "tab:orange"]


def plot_dataset(x, y, edge_index):
    c = [COL[i] for i in y.tolist()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    G = nx.Graph(); G.add_nodes_from(range(len(y))); G.add_edges_from(edge_index.T.tolist())
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax[0])
    nx.draw_networkx_nodes(G, pos, node_color=c, node_size=25, ax=ax[0])
    ax[0].set_title("Graph (colour = class)"); ax[0].axis("off")
    ax[1].scatter(x[:, 0], x[:, 1], c=c, s=15)
    ax[1].set_title("Features f1 vs f2 (classes overlap)"); ax[1].set_xlabel("f1"); ax[1].set_ylabel("f2")
    deg = torch.bincount(edge_index[1], minlength=len(y))
    ax[2].hist(deg.numpy(), bins=20); ax[2].set_title("Degree distribution")
    plt.tight_layout(); plt.show()


def plot_splits(y, masks):
    names = ["train", "val", "test"]
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (m, nm) in enumerate(zip(masks, names)):
        counts = torch.bincount(y[m], minlength=2).numpy()
        ax.bar(np.arange(2) + i * 0.27, counts, 0.27, label=nm)
    ax.set_xticks([0.27, 1.27]); ax.set_xticklabels(["class 0", "class 1"])
    ax.set_title("Nodes per split / class"); ax.legend(); plt.show()


def plot_adjacency(edge_index, y):
    """Raw (A + I) vs normalised matrix, nodes sorted by class to expose the block structure."""
    n = len(y)
    e = add_self_loops(edge_index, n); w = gcn_norm(e, n)
    raw, nrm = torch.zeros(n, n), torch.zeros(n, n)
    raw[e[1], e[0]] = 1.0; nrm[e[1], e[0]] = w
    o = y.argsort()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, M, t in zip(ax, (raw, nrm), ("A + I (self-loops added)", "D^-1/2 (A+I) D^-1/2")):
        im = a.imshow(M[o][:, o], cmap="viridis"); a.set_title(t); plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout(); plt.show()


def plot_curves(hist, title="GCN training"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["loss"]); ax[0].set_title(f"{title}: train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["train"], label="train"); ax[1].plot(hist["val"], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.show()


def plot_ablation(res):
    names = list(res); m = [np.mean(v) for v in res.values()]; s = [np.std(v) for v in res.values()]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names, m, yerr=s, capsize=4, color=["gray", "tab:green", "tab:red", "tab:purple"])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("test accuracy (mean ± std)")
    plt.xticks(rotation=15); plt.tight_layout(); plt.show()


def _pca2(z):
    z = z - z.mean(0)
    _, _, vh = torch.linalg.svd(z, full_matrices=False)
    return (z @ vh[:2].T).numpy()


@torch.no_grad()
def plot_embeddings(model, x, edge_index, y):
    """Raw features vs GCN hidden layer: message passing should pull same-class nodes together."""
    model.eval()
    _, h = model(x, edge_index, return_hidden=True)
    c = [COL[i] for i in y.tolist()]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, z, t in zip(ax, (x[:, :2].numpy(), _pca2(h)), ("Input features (f1, f2)", "GCN hidden layer (PCA)")):
        a.scatter(z[:, 0], z[:, 1], c=c, s=15); a.set_title(t)
    plt.tight_layout(); plt.show()