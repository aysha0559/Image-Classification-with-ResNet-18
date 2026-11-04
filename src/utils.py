"""Seeding, device selection, git info, atomic saving, parameter count, env paths, a timer."""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and set cuDNN flags per DECISIONS D4."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: seed Python and NumPy in each worker from its torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU torch.Generator seeded with `seed`, for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(pref: str = "auto") -> torch.device:
    """Return the torch device for `pref` ("auto", "cpu" or "cuda"); auto picks CUDA if present."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if pref not in ("cpu", "cuda"):
        raise ValueError(f"unknown device preference {pref!r}; use auto, cpu or cuda")
    return torch.device(pref)


def _git(*args: str) -> str | None:
    """Run a git command in the repo root; return stripped stdout, or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def get_git_commit() -> str:
    """Return the current git commit hash, or "unknown" if not available."""
    return _git("rev-parse", "HEAD") or "unknown"


def is_git_dirty() -> bool:
    """Return True if tracked files have uncommitted changes (False if git is unavailable)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return bool(status)


def atomic_save(obj: Any, path: str | os.PathLike) -> None:
    """Save `obj` with torch.save to a temp file in the same folder, then rename onto `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ENV_DEFAULTS: dict[str, str] = {
    "DATA_DIR": "./data",
    "RUNS_DIR": "./runs",
    "RESULTS_DIR": "./results",
}


def env_path(name: str, default: str | None = None) -> Path:
    """Return the path in env var `name` (CONTRACTS C2), else `default`, else C2's default."""
    if default is None:
        default = ENV_DEFAULTS.get(name)
    value = os.environ.get(name) or default
    if value is None:
        raise ValueError(f"environment variable {name} is not set and has no default")
    return Path(value)


class Timer:
    """Context manager measuring wall-clock seconds in `.elapsed`."""

    def __init__(self) -> None:
        self.start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - (self.start or 0.0)
