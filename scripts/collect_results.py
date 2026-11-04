"""Import results from a Drive download (Phase 08).

Usage: python -m scripts.collect_results --from PATH

PATH is the downloaded `MyDrive/cifar10-robustness/results/` folder, holding `raw/` and
`run_folders/`, or a folder with `results/` inside it.

Every evaluated run is checked against CONTRACTS C7 and D4. If anything is wrong, or a file differs
from the copy already here, the problems are listed and nothing is copied (exit code 1). Otherwise
result files go to `$RESULTS_DIR/raw/`, each run's small files to `$RUNS_DIR/{run_id}/`, and the
registry is rebuilt, because it is built from those run folders. Existing files are only replaced
in the run folder of an unfinished run, which legitimately moves on between downloads.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.plan import config_epochs, parse_run_id, read_log_rows
from scripts.registry import build_registry
from src.constants import (
    CORRUPTION_TO_FAMILY,
    CORRUPTIONS,
    MODEL_SHORT_NAMES,
    NUM_CLASSES,
    SEEN_CORRUPTIONS,
    SEVERITIES,
)
from src.utils import env_path

# src.evaluate holds these too, but importing it would load the CIFAR-10-C reader (RULES 3).
# tests/test_matrix.py checks they are the same.
C7_COLUMNS = (
    "run_id",
    "model",
    "strategy",
    "seed",
    "corruption",
    "family",
    "group",
    "severity",
    "n_images",
    "accuracy",
    "macro_f1",
    "loss",
    "git_commit",
    "evaluated_at",
)
C6_LOG_COLUMNS = ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s")
N_IMAGES = 10000
RAW_SUFFIXES = (".csv", "_confusion.npz", "_train_log.csv")
RUN_FOLDER_FILES = ("run_meta.json", "config_resolved.yaml", "train_log.csv")
LONG_MODEL = {short: name for name, short in MODEL_SHORT_NAMES.items()}


@dataclass
class Import:
    """What a download holds, once checked."""

    copies: list[tuple[Path, Path]] = field(default_factory=list)
    evaluated: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def expected_conditions() -> set[tuple[str, int]]:
    """Return the 76 (corruption, severity) pairs of D7."""
    return {("clean", 0)} | {(c, s) for c in CORRUPTIONS for s in SEVERITIES}


def _expected_labels(corruption: str) -> tuple[str, str]:
    """Return the (family, group) a condition must carry, or empty strings if it is unknown."""
    if corruption == "clean":
        return "clean", "clean"
    if corruption not in CORRUPTION_TO_FAMILY:
        return "", ""
    group = "seen" if corruption in SEEN_CORRUPTIONS else "unseen"
    return CORRUPTION_TO_FAMILY[corruption], group


def _check_row(row: dict[str, str], run_id: str) -> list[str]:
    """Return what is wrong with one result row."""
    short, strategy, seed = parse_run_id(run_id)
    identity = {
        "run_id": run_id,
        "model": LONG_MODEL[short],
        "strategy": strategy,
        "seed": str(seed),
        "n_images": str(N_IMAGES),
    }
    condition = f"{row['corruption']} s{row['severity']}"
    wrong = [f"{condition}: {k} is {row[k]!r}" for k, v in identity.items() if row[k] != v]
    if (row["family"], row["group"]) != _expected_labels(row["corruption"]):
        wrong.append(f"{condition}: family, group is {row['family']}, {row['group']}")
    for name in ("accuracy", "macro_f1"):
        if not 0.0 <= float(row[name]) <= 1.0:  # also False for NaN
            wrong.append(f"{condition}: {name} is {row[name]}")
    if not math.isfinite(float(row["loss"])):
        wrong.append(f"{condition}: loss is {row['loss']}")
    if not row["git_commit"] or not row["evaluated_at"]:
        wrong.append(f"{condition}: git_commit or evaluated_at is empty")
    return wrong


def check_results_csv(path: Path, run_id: str) -> tuple[list[dict[str, str]], list[str]]:
    """Return (rows, problems) for a C7 results CSV."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = tuple(reader.fieldnames or ())
    if columns != C7_COLUMNS:
        return rows, [f"columns are {columns}"]
    if any(None in row or None in row.values() for row in rows):
        return rows, ["a row has the wrong number of fields"]
    try:
        problems = [p for row in rows for p in _check_row(row, run_id)]
        seen = [(row["corruption"], int(row["severity"])) for row in rows]
    except ValueError as e:
        return rows, [f"a value does not parse: {e}"]
    if len(seen) != len(expected_conditions()) or set(seen) != expected_conditions():
        problems.append(f"{len(seen)} rows, {len(set(seen))} distinct conditions, expected 76")
    return rows, problems


def check_confusion(path: Path, rows: list[dict[str, str]]) -> list[str]:
    """Return what is wrong with a C7 confusion file, read against its CSV rows."""
    with np.load(path, allow_pickle=False) as data:
        matrices = {key: data[key] for key in data.files}
    accuracies = {f"{r['corruption']}__s{r['severity']}": r["accuracy"] for r in rows}
    if set(matrices) != set(accuracies):
        return [f"{len(matrices)} matrices do not match the {len(accuracies)} CSV conditions"]
    problems = []
    for key, matrix in matrices.items():
        if matrix.shape != (NUM_CLASSES, NUM_CLASSES) or matrix.sum() != N_IMAGES:
            problems.append(f"{key}: shape {matrix.shape}, sum {matrix.sum()}")
        elif abs(np.trace(matrix) / N_IMAGES - float(accuracies[key])) > 1e-12:
            problems.append(f"{key}: accuracy differs from the matrix")
    return problems


def check_train_log(path: Path, epochs: int) -> list[str]:
    """Return what is wrong with a C6 train log: it must hold epochs 1..E once each, all finite."""
    rows = read_log_rows(path)
    if not rows or tuple(rows[0]) != C6_LOG_COLUMNS:
        return ["no rows, or the columns are wrong"]
    if [row["epoch"] for row in rows] != [str(e) for e in range(1, epochs + 1)]:
        return [f"{len(rows)} rows do not count epochs 1..{epochs} once each"]
    if not all(math.isfinite(float(v)) for row in rows for v in row.values()):
        return ["a value is NaN or infinite"]
    return []


def check_meta(meta: dict, run_id: str, epochs: int) -> list[str]:
    """Return what is wrong with the metadata of an evaluated run."""
    short, strategy, seed = parse_run_id(run_id)
    expected = {
        "run_id": run_id,
        "model": LONG_MODEL[short],
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "status": "finished",
    }
    return [f"run_meta {k} is {meta.get(k)!r}" for k, v in expected.items() if meta.get(k) != v]


def find_source(path: Path) -> Path:
    """Return the folder under `path` that holds `raw/`; raise FileNotFoundError if none does."""
    for candidate in (path, path / "results"):
        if (candidate / "raw").is_dir():
            return candidate
    raise FileNotFoundError(f"no raw/ folder in {path} or {path / 'results'}")


def _plan_copy(source: Path, dest: Path, may_replace: bool, found: Import) -> None:
    """Add a copy to `found`, or a problem if `dest` differs and must not be replaced."""
    if not dest.exists():
        found.copies.append((source, dest))
    elif not filecmp.cmp(source, dest, shallow=False):
        if may_replace:
            found.copies.append((source, dest))
        else:
            found.problems.append(f"{dest} differs from {source}; not replacing it")


def _plan_run_folder(source: Path, run_id: str, runs_dir: Path, found: Import) -> None:
    """Plan the copy of a run's small files, which the registry reads from the run folder."""
    local_meta = runs_dir / run_id / "run_meta.json"
    finished = local_meta.exists() and json.loads(local_meta.read_text())["status"] == "finished"
    for name in RUN_FOLDER_FILES:
        if (source / name).exists():
            _plan_copy(source / name, runs_dir / run_id / name, not finished, found)


def _check_evaluated_run(
    run_id: str, source: Path, results_dir: Path, runs_dir: Path, found: Import
) -> None:
    """Check one run's three result files and metadata, and plan their copies."""
    raw = source / "raw"
    missing = [s for s in RAW_SUFFIXES if not (raw / f"{run_id}{s}").exists()]
    if missing:
        found.problems.append(f"{run_id}: missing {missing}")
        return
    short, strategy, _ = parse_run_id(run_id)
    epochs = config_epochs(short, strategy)
    rows, problems = check_results_csv(raw / f"{run_id}.csv", run_id)
    if not problems:
        problems = check_confusion(raw / f"{run_id}_confusion.npz", rows)
    problems += check_train_log(raw / f"{run_id}_train_log.csv", epochs)
    meta_paths = [source / "run_folders" / run_id / "run_meta.json"]
    meta_paths.append(runs_dir / run_id / "run_meta.json")
    meta_path = next((p for p in meta_paths if p.exists()), None)
    if meta_path is None:
        problems.append("no run_meta.json in the download or the run folder")
    else:
        problems += check_meta(json.loads(meta_path.read_text()), run_id, epochs)
    found.problems += [f"{run_id}: {p}" for p in problems]
    found.evaluated.append(run_id)
    for suffix in RAW_SUFFIXES:
        name = f"{run_id}{suffix}"
        _plan_copy(raw / name, results_dir / "raw" / name, False, found)


def _matrix_id(name: str) -> str | None:
    """Return `name` if it is a real run id, else None (smoke, timing and invalid runs)."""
    try:
        parse_run_id(name)
    except ValueError:
        return None
    return name


def scan_download(source: Path, results_dir: Path, runs_dir: Path) -> Import:
    """Check everything in a download and work out what to copy; change nothing."""
    found = Import()
    for path in sorted((source / "raw").iterdir()):
        if path.name.endswith(RAW_SUFFIXES[1:]):
            continue
        run_id = _matrix_id(path.stem) if path.suffix == ".csv" else None
        if run_id:
            _check_evaluated_run(run_id, source, results_dir, runs_dir, found)
        else:
            found.notes.append(f"ignored {path.name}")
    for path in sorted((source / "raw").glob("*_train_log.csv")):
        run_id = path.name.removesuffix("_train_log.csv")
        if not (source / "raw" / f"{run_id}.csv").exists():
            found.notes.append(f"{run_id}: trained, not evaluated yet; its results wait")
    folders = source / "run_folders"
    for path in sorted(folders.iterdir()) if folders.is_dir() else ():
        if _matrix_id(path.name):
            _plan_run_folder(path, path.name, runs_dir, found)
        else:
            found.notes.append(f"ignored run_folders/{path.name}")
    return found


def describe_run(run_id: str, runs_dir: Path, results_dir: Path) -> str:
    """Return one line of what Phase 08's checks after each batch look at."""
    meta = json.loads((runs_dir / run_id / "run_meta.json").read_text())
    with (results_dir / "raw" / f"{run_id}.csv").open(newline="") as f:
        clean = next(row["accuracy"] for row in csv.DictReader(f) if row["corruption"] == "clean")
    return (
        f"{run_id}: {meta['status']}, {meta['epochs']} epochs, commit {meta['git_commit'][:7]}, "
        f"dirty {meta['git_dirty']}, {meta['device_name']}, clean {clean}"
    )


def collect(path: Path, results_dir: Path, runs_dir: Path) -> Import:
    """Import a download unless it has problems; rebuild the registry if it was imported."""
    found = scan_download(find_source(path), results_dir, runs_dir)
    if found.problems:
        return found
    for source, dest in found.copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    build_registry(runs_dir, results_dir)
    return found


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Import results from a Drive download.")
    parser.add_argument("--from", dest="source", type=Path, required=True, help="downloaded folder")
    args = parser.parse_args(argv)
    runs_dir, results_dir = env_path("RUNS_DIR"), env_path("RESULTS_DIR")
    found = collect(args.source, results_dir, runs_dir)
    for note in found.notes:
        print(note)
    if found.problems:
        print(f"nothing imported, {len(found.problems)} problems:", file=sys.stderr)
        for problem in found.problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(found.evaluated)} evaluated runs checked, {len(found.copies)} files copied")
    for run_id in found.evaluated:
        print(describe_run(run_id, runs_dir, results_dir))
    print(f"rebuilt {results_dir / 'runs.csv'}")


if __name__ == "__main__":
    main()
