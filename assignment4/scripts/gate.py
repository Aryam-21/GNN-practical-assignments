"""CI promotion gate: train a small candidate, compare with the approved model on the SAME validation data.
Exit code 1 (job fails) when blocked. Works on a temp COPY of the registry, so CI never changes it."""
import argparse, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import hold_out_unseen, load_cora, training_graph
from assignment4.src.model import train_model
from assignment4.src.package import save_package, val_accuracy
from registry import Registry
from assignment4.src.tracking import fingerprint

ap = argparse.ArgumentParser()
ap.add_argument("--registry", default="registry"); ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--n-labels", type=int, default=None)
ap.add_argument("--min-delta", type=float, default=-0.02, help="negative = allow small noise (CI mode)")
ap.add_argument("--floor", type=float, default=0.40)
a = ap.parse_args()

data = load_cora(str(Path(__file__).resolve().parents[1] / "data"))
g = training_graph(data, hold_out_unseen(data))
cfg = dict(hidden=16, dropout=0.5, lr=0.01, wd=5e-4, epochs=a.epochs, n_labels=a.n_labels, n_classes=7)
model, val, _ = train_model(g, cfg, a.seed)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "registry"
    if Path(a.registry).exists():
        shutil.copytree(a.registry, root)
    pkg = save_package(Path(tmp) / "cand", model, cfg, g.x.size(1), [f"class_{i}" for i in range(7)], fingerprint(g.x, g.edge_index, g.y))
    reg = Registry(root)
    v = reg.register(pkg, {"val_acc": val})
    d = reg.try_promote(v, lambda p: val_accuracy(p, g), a.min_delta, a.floor)
    print(reg.state["versions"][v]["gate"])
    print(d.reason)
sys.exit(0 if d.promote else 1)