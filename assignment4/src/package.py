"""Model package = weights + architecture + preprocessing + feature schema + label mapping."""
import json
from pathlib import Path
import torch
from model import GCN, normalize_rows


class ModelLoadError(RuntimeError):
    pass


def save_package(path, model, cfg, n_features, labels, dataset_id, extra=None):
    p = Path(path); p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / "weights.pt")
    meta = {"architecture": {"name": "GCN", "d_in": n_features, "d_hid": cfg["hidden"],
                             "d_out": len(labels), "dropout": cfg["dropout"]},
            "preprocessing": "row_normalize: x / max(sum(x), 1)",
            "feature_schema": {"n_features": n_features, "dtype": "float32", "min_value": 0.0},
            "labels": labels, "dataset_id": dataset_id, "torch_version": torch.__version__,
            "version": "unregistered", **(extra or {})}
    (p / "meta.json").write_text(json.dumps(meta, indent=2))
    return p


class Predictor:
    def __init__(self, path):
        try:
            p = Path(path)
            self.meta = json.loads((p / "meta.json").read_text())
            a = self.meta["architecture"]
            self.model = GCN(a["d_in"], a["d_hid"], a["d_out"], a["dropout"])
            self.model.load_state_dict(torch.load(p / "weights.pt", map_location="cpu", weights_only=True))
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"cannot load model from {path}: {e}") from e
        self.version = self.meta["version"]
        self.n_features = self.meta["feature_schema"]["n_features"]
        self.labels = self.meta["labels"]

    @torch.no_grad()
    def logits(self, x, edge_index, return_emb=False):
        return self.model(normalize_rows(x.float()), edge_index, return_emb=return_emb)

    @torch.no_grad()
    def predict(self, x, edge_index, ids):
        probs = self.logits(x, edge_index).softmax(1)[ids]
        deg = torch.bincount(edge_index[1], minlength=x.size(0))[ids]
        conf, lab = probs.max(1)
        return [dict(id=int(i), label=int(l), label_name=self.labels[int(l)], confidence=float(c), isolated=bool(d == 0))
                for i, l, c, d in zip(ids.tolist(), lab, conf, deg)]


def val_accuracy(path, g):
    """Validation accuracy of a saved package on graph g (same data for every model -> fair gate)."""
    pred = Predictor(path).logits(g.x, g.edge_index).argmax(1)
    return (pred[g.val_mask] == g.y[g.val_mask]).float().mean().item()