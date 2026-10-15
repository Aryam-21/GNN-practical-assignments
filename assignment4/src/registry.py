"""Tiny file-based model registry + validation gate + rollback (pure Python, easy to test)."""
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Decision:
    promote: bool
    reason: str


def _missing(m):
    return m is None or (isinstance(m, float) and math.isnan(m))


def gate(candidate, approved, has_approved, min_delta=0.005, floor=0.40):
    """Metric = validation accuracy of candidate and approved model on the SAME validation set.
    * missing candidate metric          -> BLOCK
    * candidate below absolute floor    -> BLOCK
    * no approved model yet (1st model) -> PROMOTE (floor already passed)
    * approved metric missing           -> BLOCK (cannot compare)
    * gain >= min_delta                 -> PROMOTE
    * 0 <= gain < min_delta (tie/noise) -> BLOCK (keep the approved model)
    * gain < 0                          -> BLOCK (regression)
    min_delta < 0 turns this into a 'no significant regression' check (used in CI)."""
    if _missing(candidate):
        return Decision(False, "BLOCKED: candidate metric missing")
    if candidate < floor:
        return Decision(False, f"BLOCKED: {candidate:.3f} is below the floor {floor}")
    if not has_approved:
        return Decision(True, "PROMOTE: first model and above floor")
    if _missing(approved):
        return Decision(False, "BLOCKED: approved model metric missing, cannot compare")
    gain = candidate - approved
    if gain >= min_delta - 1e-12:
        return Decision(True, f"PROMOTE: {candidate:.3f} vs approved {approved:.3f} (gain {gain:+.3f})")
    if gain >= 0:
        return Decision(False, f"BLOCKED: gain {gain:+.3f} < required {min_delta} (tie/noise), keep approved")
    return Decision(False, f"BLOCKED: regression {candidate:.3f} < approved {approved:.3f}")


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.file = self.root / "registry.json"
        self.state = json.loads(self.file.read_text()) if self.file.exists() else {"versions": {}, "approved": None, "history": []}

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.state, indent=2))

    @property
    def approved(self):
        return self.state["approved"]

    def path(self, version):
        return self.root / "versions" / version

    def register(self, pkg_dir, metrics, run_id=None):
        v = f"v{len(self.state['versions']) + 1}"
        dst = self.path(v)
        shutil.copytree(pkg_dir, dst)
        meta = json.loads((dst / "meta.json").read_text()); meta["version"] = v
        (dst / "meta.json").write_text(json.dumps(meta, indent=2))
        self.state["versions"][v] = {"metrics": metrics, "run_id": run_id}
        self._save()
        return v

    def promote(self, version):
        self.state["approved"] = version
        self.state["history"].append(version)
        self._save()

    def rollback(self):
        h = self.state["history"]
        if len(h) < 2:
            raise RuntimeError("no previous approved version to roll back to")
        h.pop()
        self.state["approved"] = h[-1]
        self._save()
        return h[-1]

    def try_promote(self, version, evaluate, min_delta=0.005, floor=0.40):
        """evaluate(package_path) -> metric. A failing evaluation counts as a missing metric."""
        def safe(path):
            try:
                return evaluate(path)
            except Exception:
                return None
        cur = self.approved
        cand = safe(self.path(version))
        appr = safe(self.path(cur)) if cur else None
        d = gate(cand, appr, cur is not None, min_delta, floor)
        self.state["versions"][version]["gate"] = {"candidate": cand, "approved": appr, "promote": d.promote, "reason": d.reason}
        if d.promote:
            self.promote(version)
        else:
            self._save()
        return d