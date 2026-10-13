"""Lets each tests/test_assignmentN.py import its OWN assignment's code (data.py, models.py, ...).
Runs automatically before pytest imports each test file."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_NAMES = ("data", "models", "train", "viz", "experiments", "checks")   # names reused by several assignments


def _code_dir(n):
    for d in ROOT.iterdir():                                       # assignment-2, assignment_2, Assignment2 ...
        if d.is_dir() and re.fullmatch(rf"assignment[\s_-]*{n}", d.name, re.I):
            for c in (d / "src", d):                               # code in src/ or directly in the folder
                if c.exists() and any(c.glob("*.py")):
                    return c
    return None


def pytest_collectstart(collector):
    path = Path(str(getattr(collector, "path", None) or collector.fspath))
    m = re.fullmatch(r"test_assignment(\d+)\.py", path.name)
    code = _code_dir(m.group(1)) if m else None
    if code:
        for name in SHARED_NAMES:                                  # forget modules loaded for another assignment
            sys.modules.pop(name, None)
        sys.path.insert(0, str(code))