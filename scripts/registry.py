"""Build results/runs.csv from the run folders and result CSVs (CONTRACTS C8).

Usage: python -m scripts.registry

Reads $RUNS_DIR/*/run_meta.json and $RESULTS_DIR/{raw,smoke}/{run_id}.csv.
The file is always rebuilt from those; never edit it by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils import env_path

META_COLUMNS = (
    "run_id",
    "model",
    "strategy",
    "seed",
    "epochs",
    "git_commit",
    "git_dirty",
    "torch_version",
    "torchvision_version",
    "python_version",
    "device_name",
    "n_params",
    "started_at",
    "finished_at",
    "train_time_s",
    "status",
)
COLUMNS = (*META_COLUMNS, "has_results", "clean_accuracy")


def clean_accuracy(results_csv: Path) -> str:
    """Return the clean accuracy in a results CSV as written there, or "" if it has none."""
    with results_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["corruption"] == "clean":
                return row["accuracy"]
    return ""


def registry_row(meta: dict, results_dir: Path) -> dict:
    """Return one registry row for a run's metadata."""
    folder = "smoke" if meta["run_id"].startswith("smoke-") else "raw"
    results_csv = results_dir / folder / f"{meta['run_id']}.csv"
    has_results = results_csv.exists()
    return {
        **meta,
        "has_results": has_results,
        "clean_accuracy": clean_accuracy(results_csv) if has_results else "",
    }


def build_registry(runs_dir: Path, results_dir: Path) -> Path:
    """Write `results_dir`/runs.csv with one row per run, sorted by run id; return its path."""
    metas = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/run_meta.json"))]
    out = results_dir / "runs.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        # extrasaction="ignore": a failed run's `error` message is not a C8 column.
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(registry_row(m, results_dir) for m in metas)
    return out


def main() -> None:
    """Rebuild the registry from RUNS_DIR and RESULTS_DIR."""
    out = build_registry(env_path("RUNS_DIR"), env_path("RESULTS_DIR"))
    with out.open(newline="") as f:
        n_runs = sum(1 for _ in csv.DictReader(f))
    print(f"wrote {out} ({n_runs} runs)")


if __name__ == "__main__":
    main()
