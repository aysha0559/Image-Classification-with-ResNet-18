"""Evaluation entry point: python -m src.evaluate --run-dir DIR [--subset N] [--include-extra]
[--batch-size 512] [--force].

Scores the final-epoch model of a finished run (DECISIONS D3) on the clean test set and on every
CIFAR-10-C (corruption, severity) pair, and writes the CONTRACTS C7 files to `$RESULTS_DIR/raw/`:
`{run_id}.csv`, `{run_id}_confusion.npz` and `{run_id}_train_log.csv`.

`--subset N` scores the first N images of each condition. It adds the `smoke-` prefix to the run
id, and smoke results go only to `$RESULTS_DIR/smoke/` (CONTRACTS C5).

The 4 extra corruptions run only with `--include-extra` (D7). C7 allows only the groups `clean`,
`seen` and `unseen`, and no training ever uses the extras, so they are labelled `unseen`.

The CSV is written last and atomically, so its presence means the whole evaluation finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.constants import (
    CORRUPTION_TO_FAMILY,
    CORRUPTIONS,
    EXTRA_CORRUPTIONS,
    SEEN_CORRUPTIONS,
    SEVERITIES,
)
from src.data.cifar10 import get_test_dataset
from src.data.cifar10c import CIFAR10C
from src.data.transforms import build_eval_transform
from src.metrics import accuracy, confusion_matrix, macro_f1
from src.models import build_model
from src.utils import env_path, get_device, get_git_commit

# Exactly the columns of CONTRACTS C7, in order.
CSV_COLUMNS = (
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
DEFAULT_BATCH_SIZE = 512
CLEAN = "clean"
WORST_SHOWN = 3


def conditions(include_extra: bool = False) -> Iterator[tuple[str, int]]:
    """Yield every (corruption, severity): clean first, then the 15 x 5, then any extras."""
    yield CLEAN, 0
    names = CORRUPTIONS + EXTRA_CORRUPTIONS if include_extra else CORRUPTIONS
    for corruption in names:
        for severity in SEVERITIES:
            yield corruption, severity


def family_of(corruption: str) -> str:
    """Return the C7 `family` of a condition."""
    return CLEAN if corruption == CLEAN else CORRUPTION_TO_FAMILY[corruption]


def group_of(corruption: str) -> str:
    """Return the C7 `group` of a condition: `clean`, `seen` or `unseen` (D6)."""
    if corruption == CLEAN:
        return CLEAN
    return "seen" if corruption in SEEN_CORRUPTIONS else "unseen"


def smoke_run_id(run_id: str) -> str:
    """Return `run_id` with the `smoke-` prefix, added at most once."""
    return run_id if run_id.startswith("smoke-") else f"smoke-{run_id}"


def result_paths(results_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Return a run's (csv, confusion npz) paths; `smoke-` runs never go to `raw/`."""
    folder = results_dir / ("smoke" if run_id.startswith("smoke-") else "raw")
    return folder / f"{run_id}.csv", folder / f"{run_id}_confusion.npz"


def load_finished_run(run_dir: Path) -> tuple[dict, dict]:
    """Return (run_meta, model_final contents); raise unless the run is finished and consistent."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; is {run_dir} a run folder?")
    meta = json.loads(meta_path.read_text())
    if meta["status"] != "finished":
        raise ValueError(f"refusing to evaluate {run_dir}: status is {meta['status']!r}")
    model_path = run_dir / "model_final.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} not found, although the run says it finished")
    final = torch.load(model_path, map_location="cpu", weights_only=True)
    if final["run_id"] != meta["run_id"]:
        raise ValueError(
            f"{run_dir}: model_final.pt is from {final['run_id']!r}, "
            f"run_meta.json from {meta['run_id']!r}"
        )
    return meta, final


def condition_dataset(data_dir: Path, corruption: str, severity: int) -> Dataset:
    """Return the images of one condition with the eval transform (D2)."""
    if corruption == CLEAN:
        return get_test_dataset(str(data_dir))
    return CIFAR10C(corruption, severity, build_eval_transform(), data_dir)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (true labels, predicted labels, mean cross-entropy) over `loader`."""
    model.eval()
    labels_seen, predictions = [], []
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        outputs = outputs.float()  # sum the loss in full precision, also under AMP
        total_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
        labels_seen.append(labels.cpu())
        predictions.append(outputs.argmax(dim=1).cpu())
    y_true = torch.cat(labels_seen).numpy()
    return y_true, torch.cat(predictions).numpy(), total_loss / len(y_true)


def score_condition(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    num_workers: int,
    subset: int | None,
) -> tuple[dict, np.ndarray]:
    """Return ({n_images, accuracy, macro_f1, loss}, confusion matrix) for one condition."""
    if subset is not None:
        dataset = Subset(dataset, range(min(subset, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_true, y_pred, loss = predict(model, loader, device, use_amp)
    scores = {
        "n_images": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "loss": loss,
    }
    return scores, confusion_matrix(y_true, y_pred)


def write_confusion(path: Path, matrices: dict[str, np.ndarray]) -> None:
    """Write the C7 confusion file atomically."""
    tmp = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        np.savez_compressed(f, **matrices)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the C7 CSV atomically."""
    tmp = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def print_summary(run_id: str, rows: list[dict]) -> None:
    """Print clean accuracy, mean corrupted accuracy over the 75 conditions, worst 3 conditions."""
    clean = next(row for row in rows if row["corruption"] == CLEAN)
    # Extras never enter the main averages (D7).
    corrupted = [row for row in rows if row["corruption"] in CORRUPTIONS]
    mean_corrupted = float(np.mean([row["accuracy"] for row in corrupted]))
    print(f"{run_id}: {len(rows)} conditions, {clean['n_images']} images each")
    print(f"clean accuracy          {clean['accuracy']:.4f}")
    print(f"mean corrupted accuracy {mean_corrupted:.4f} (over {len(corrupted)} conditions)")
    print("worst conditions:")
    for row in sorted(corrupted, key=lambda r: r["accuracy"])[:WORST_SHOWN]:
        print(f"  {row['corruption']} s{row['severity']}: {row['accuracy']:.4f}")


def evaluate(
    run_dir: str | Path,
    subset: int | None = None,
    include_extra: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> Path:
    """Evaluate a finished run on every condition, write the C7 files, and return the CSV path.

    Raises, before any evaluation, if the run is not finished or results exist without `force`.
    """
    if subset is not None and subset < 1:
        raise ValueError(f"--subset must be >= 1, got {subset}")
    run_dir = Path(run_dir)
    meta, final = load_finished_run(run_dir)
    run_id = smoke_run_id(meta["run_id"]) if subset is not None else meta["run_id"]
    csv_path, npz_path = result_paths(env_path("RESULTS_DIR"), run_id)
    existing = [str(p) for p in (csv_path, npz_path) if p.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite {existing}; use --force")

    cfg = final["config"]
    device = get_device(cfg["device"])
    use_amp = cfg["amp"] and device.type == "cuda"
    model = build_model(cfg["model"])
    model.load_state_dict(final["state_dict"])
    model.to(device)

    data_dir = env_path("DATA_DIR")
    commit = get_git_commit()
    rows, matrices = [], {}
    for corruption, severity in conditions(include_extra):
        scores, matrix = score_condition(
            model,
            condition_dataset(data_dir, corruption, severity),
            device,
            use_amp,
            batch_size,
            cfg["num_workers"],
            subset,
        )
        matrices[f"{corruption}__s{severity}"] = matrix
        rows.append(
            {
                "run_id": run_id,
                "model": meta["model"],
                "strategy": meta["strategy"],
                "seed": meta["seed"],
                "corruption": corruption,
                "family": family_of(corruption),
                "group": group_of(corruption),
                "severity": severity,
                **scores,
                "git_commit": commit,
                "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    write_confusion(npz_path, matrices)
    # A copy, as C7 asks: on Colab the copy made by training is lost with the session.
    shutil.copyfile(run_dir / "train_log.csv", csv_path.with_name(f"{run_id}_train_log.csv"))
    write_csv(csv_path, rows)
    print_summary(run_id, rows)
    print(f"wrote {csv_path}")
    return csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line of CONTRACTS C5."""
    parser = argparse.ArgumentParser(description="Evaluate a finished run on all conditions.")
    parser.add_argument("--run-dir", required=True, help="the run folder (CONTRACTS C6)")
    parser.add_argument("--subset", type=int, help="first N images per condition; smoke output")
    parser.add_argument("--include-extra", action="store_true", help="add the 4 extra corruptions")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    evaluate(args.run_dir, args.subset, args.include_extra, args.batch_size, args.force)


if __name__ == "__main__":
    main()
