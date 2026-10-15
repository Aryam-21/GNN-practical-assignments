"""Build the graph context a request needs. A 2-layer GCN needs the 3-hop neighbourhood to reproduce
full-graph predictions exactly (2 hops of messages + degrees of the outermost nodes)."""
import torch
from torch_geometric.utils import k_hop_subgraph


def context_graph(x, edge_index, ids, hops=3):
    ids = torch.as_tensor(ids)
    subset, ei, mapping, _ = k_hop_subgraph(ids, hops, edge_index, relabel_nodes=True, num_nodes=x.size(0))
    return x[subset], ei, mapping, subset            # mapping = local ids of the targets, subset = original ids


def build_request(x, edge_index, ids, hops=3):
    xs, ei, local, subset = context_graph(x, edge_index, ids, hops)
    return {"features": xs.tolist(), "edges": ei.T.tolist(), "target_ids": local.tolist()}, subset