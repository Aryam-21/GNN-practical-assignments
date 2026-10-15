"""MLflow experiment tracking: params, seed, code id, dataset id, metrics, model artifact."""
import hashlib
import subprocess
from pathlib import Path
import mlflow
from model import train_model
from package import save_package

TRACKING_URI = "sqlite:///mlflow.db"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def fingerprint(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def run_experiment(name, g, cfg, seed, labels, out_dir, uri=TRACKING_URI, experiment="cora-gcn"):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    ds_id, commit = fingerprint(g.x, g.edge_index, g.y), git_commit()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({**cfg, "seed": seed})
        mlflow.set_tags({"git_commit": commit, "dataset_id": ds_id})
        model, val, hist = train_model(g, cfg, seed)
        for i, (loss, v) in enumerate(hist):
            mlflow.log_metric("train_loss", loss, step=i)
            mlflow.log_metric("val_acc", v, step=i)
        mlflow.log_metric("best_val_acc", val)
        pkg = save_package(Path(out_dir) / name, model, cfg, g.x.size(1), labels, ds_id,
                           extra={"seed": seed, "git_commit": commit, "mlflow_run_id": run.info.run_id})
        mlflow.log_artifacts(str(pkg), artifact_path="model")
    return dict(name=name, run_id=run.info.run_id, package=pkg, val_acc=val, hist=hist, model=model)